// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This entry routes every request to the best free model that can serve it, in the
// quality table's rating order, and fails over to the next best free model when a
// provider refuses (402, 429 or 5xx). It speaks all three upstream wires (OpenAI,
// Anthropic, Gemini) and answers in the wire the client used, in JSON and in
// server-sent events. A JSON-mode request never rides an anthropic-wire provider:
// that wire cannot carry response_format, and its echo would read json_mode false.
//
// WHAT A TRIAL GIVES YOU
//   The brief at $MAKEFUN_PROMPT_FILE carries the provider directory (ten providers,
//   each with its id, wire, auth scheme, key env var, key and base path), the quality
//   table (every model's rating_milli, capabilities and free flag) and the workload
//   manifest (the twenty requests, in order, with each one's api, stream flag, weight
//   and requirements). The env at $MAKEFUN_ENV_URL serves the providers under their
//   base paths and the control routes /register, /work and /instance.
//
// THE ROUTE
//   For one request, every free model that meets its requirements is ranked by the
//   quality table's rating_milli, best first. The server tries them in that order:
//   the first provider call that answers 2xx serves the request, and a 402, 429 or
//   5xx (or an unreachable provider) moves on to the next candidate. That is the
//   failover column, and it is also how a spent per-trial budget is survived: a
//   provider out of budget refuses, and the next best model answers instead.
//   Providers are never called two at a time for one request, and one provider is
//   never called concurrently: the client sends one request at a time.
//
// THE WIRES
//   Client OpenAI -> provider OpenAI: the body passes through with the model's id.
//   Client OpenAI -> provider Anthropic: openAIToAnthropicRequest.
//   Client OpenAI -> provider Gemini: openAIToGeminiRequest, model in the URL path.
//   Client Anthropic -> provider OpenAI: anthropicToOpenAI.
//   Client Anthropic -> provider Anthropic: the body passes through with the model's id.
//   Client Anthropic -> provider Gemini: anthropicToGeminiRequest, model in the URL path.
//   Every answer is translated back into the client's wire, in JSON or as the
//   client's streaming event sequence. The provider's text, usage and stop reason
//   are carried whole; the request's system prompt, user text, images, tools, stop
//   sequences, max_tokens and JSON mode are passed through untouched, because the
//   provider echoes what it parsed and the client compares that echo with its own.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers:
//   no wall clock, no unseeded randomness. The catalog is built once and printed by
//   --list byte for byte as it is registered.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The rating of one model, from the quality table's row where it has one, else 0. */
function ratingOf(probe: Probe, model: ProbedModel): number {
  const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
  return row === null ? 0 : row.rating_milli;
}

/**
 * Every free model that meets one request's requirements, best rating first.
 * A JSON-mode request never rides an anthropic-wire provider: that wire cannot carry
 * response_format, so its echo would read json_mode false and the request would fail
 * the wire check. Ties break by provider order, then model id, so the order is fixed.
 */
function rankedCandidates(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: ratingOf(probe, model) });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const pa = brief.directory.providers.findIndex((p) => p.id === a.probe.provider.id);
    const pb = brief.directory.providers.findIndex((p) => p.id === b.probe.provider.id);
    if (pa !== pb) return pa - pb;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** Whether one client body asks for JSON mode (an OpenAI response_format). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream URL and body for one candidate, in the provider's own wire. */
function upstreamCall(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): { url: string; init: RequestInit } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    // A streaming client needs a final usage chunk: the provider only sends one when
    // the request asks for it, so ask. stream_options is not part of the echo the
    // client compares, so this changes nothing the wire check reads.
    const upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody) } };
  }
  if (provider.wire === 'anthropic') {
    const upstreamBody = clientWire === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody) } };
  }
  // gemini: the model rides in the URL path, and the body is a generateContent request.
  const upstreamBody = clientWire === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const encoded = encodeURIComponent(candidate.model.id);
  return { url: `${base}/v1beta/models/${encoded}:${action}`, init: { method: 'POST', headers, body: JSON.stringify(upstreamBody) } };
}

/** Whether one upstream answer is a refusal worth failing over from: 402, 429 or any 5xx. */
function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write one provider's SSE stream to the client, translated into the client's wire. */
async function pipeTranslatedStream(candidate: Candidate, clientWire: 'openai' | 'anthropic', upstream: Response, res: ServerResponse, requestedModel: string): Promise<void> {
  const wire = candidate.probe.provider.wire;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstream.body === null) {
    res.end();
    return;
  }
  if (clientWire === 'openai') {
    if (wire === 'openai') await pipeStream(upstream.body, res);
    else if (wire === 'anthropic') for await (const event of openAIEventsFromAnthropic(upstream.body, requestedModel)) res.write(event);
    else for await (const event of openAIEventsFromGemini(upstream.body, requestedModel)) res.write(event);
    res.end();
    return;
  }
  // clientWire === 'anthropic'
  if (wire === 'anthropic') await pipeStream(upstream.body, res);
  else if (wire === 'openai') for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) res.write(event);
  else for await (const event of anthropicEventsFromGemini(upstream.body, requestedModel)) res.write(event);
  res.end();
}

/** One provider's JSON answer, translated into the client's wire. */
async function translatedJson(candidate: Candidate, clientWire: 'openai' | 'anthropic', upstream: Response, requestedModel: string): Promise<Json | null> {
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) return null;
  const wire = candidate.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicToOpenAIResponse(answer, requestedModel);
    return geminiToOpenAIResponse(answer, requestedModel);
  }
  if (wire === 'anthropic') return answer;
  if (wire === 'openai') return openAIToAnthropic(answer, requestedModel);
  return geminiToAnthropicResponse(answer, requestedModel);
}

/**
 * Serve one client request: try the ranked free candidates in rating order until one
 * answers 2xx, translating the answer back into the client's wire. A refusal (402,
 * 429, 5xx) or an unreachable provider moves on to the next candidate. Returns true
 * when an answer was sent, false when every candidate refused.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonMode = clientWire === 'openai' && jsonModeOf(body);
  const candidates = rankedCandidates(requires, jsonMode);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0]?.model.id ?? 'auto';
  for (const candidate of candidates) {
    const { url, init } = upstreamCall(candidate, clientWire, body, stream);
    let upstream: Response;
    try {
      upstream = await fetch(url, init);
    } catch {
      continue;
    }
    if (!upstream.ok) {
      // A refusal spends the provider's budget and is the failover signal: drain the
      // body so the connection is released, then try the next best free model.
      try {
        await upstream.text();
      } catch {
        // The body is gone either way.
      }
      if (isRefusal(upstream.status)) continue;
      // A 4xx that is not a refusal (a bad request, a wrong key) is not a failover
      // signal, but the next candidate may still serve it: try on.
      continue;
    }
    if (stream && upstream.body !== null) {
      await pipeTranslatedStream(candidate, clientWire, upstream, res, requestedModel);
      return true;
    }
    const translated = await translatedJson(candidate, clientWire, upstream, requestedModel);
    if (translated === null) {
      // The provider answered 2xx with a body that is not JSON: try the next candidate.
      continue;
    }
    sendJson(res, 200, translated);
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, res);
  if (!served) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, res);
  if (!served) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
