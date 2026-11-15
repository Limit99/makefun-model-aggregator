// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW THIS ENTRY ROUTES
//   - quality: every free model on a provider that answered its probe, meeting the request's
//     requirements (the manifest's row for the request now arriving), is a candidate, ranked by the
//     quality table's rating_milli, best first. A paid model is never a candidate. An OpenAI client
//     request in JSON mode (response_format) is never routed to an anthropic-wire provider: that wire
//     cannot carry response_format, so its echo would report json_mode absent and lose the request.
//   - budgets: a provider's per-trial budget is never published, so this server counts its own calls
//     per provider — every inference call it makes, refused ones included, is one against the same
//     counter the provider keeps — and lets the ranking spread ties across providers (equal rating:
//     the provider called less goes first). A provider that refuses twice in a row is demoted below
//     every healthy candidate for later requests, but stays reachable as a last resort.
//   - failover: a call answered 402, 429 or 5xx (or one that never answers) is not the client's answer:
//     the request moves to the next candidate. A provider with a clean record — at least one success
//     and no refusal yet — that refuses once is retried once at once with the same model: a one-off
//     failure slot is likelier than a spent budget early in a provider's life, and the retry keeps the
//     request on the best model instead of a weaker one. A provider gets at most those two calls per
//     request, so a spent or out provider costs a request at most one refusal plus one retry.
//   - wires: same-wire calls pass through untouched (an OpenAI streaming call upstream only gains
//     stream_options.include_usage when the client did not ask for it, so the stream carries the usage
//     the client scores; nothing the provider echoes reads that field). Cross-wire calls are translated
//     by lib/wire.ts: requests rewritten so the provider parses the same system text, user text, image,
//     tool names, output cap, stop sequences and JSON mode the client sent; answers parsed into one
//     normalized shape and rendered back in the client's wire, JSON or SSE, with usage and the stop
//     reason mapped across. Cross-wire streams are buffered whole before rendering — the providers
//     answer at once, and a buffered render carries the final usage exactly, in canonical event order.
//   Every choice above is a function of the brief, the requests and the providers' answers seen so
//   far — never of the wall clock — so a replay of the trial makes the same calls in the same order.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client wires and all three upstream wires: translations both ways.
//   LICENSE            MIT.

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
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  normFromAnthropic,
  normFromAnthropicStream,
  normFromGemini,
  normFromGeminiStream,
  normFromOpenAI,
  normFromOpenAIStream,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  type NormAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take before it counts as a refusal, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A provider with this many consecutive refusals is demoted below every healthy candidate. */
const DEMOTE_AFTER_REFUSALS = 3;

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

/** What this server has seen one provider do: its counted calls, its successes, its refusal streak. */
interface ProviderState {
  calls: number;
  successes: number;
  consecRefusals: number;
}

const providerStates = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerStates.get(providerId);
  if (state === undefined) {
    state = { calls: 0, successes: 0, consecRefusals: 0 };
    providerStates.set(providerId, state);
  }
  return state;
}

/** One legal pick for a request: a free model on a provider that answered, with its public rating. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every candidate for one request, best first: demoted providers (a refusal streak at the threshold)
 * below healthy ones, then rating, then the provider called less, then the brief's provider order,
 * then model id — a total order, so the ranking is the same on a replay. An OpenAI JSON-mode request
 * skips anthropic-wire providers outright: their wire cannot carry response_format, so their echo
 * would report json_mode absent and the request would be lost however good the model is.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', body: Json): Candidate[] {
  const jsonMode =
    clientWire === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const order = new Map(brief.directory.providers.map((p, i) => [p.id, i] as const));
  out.sort((a, b) => {
    const demotedA = stateOf(a.probe.provider.id).consecRefusals >= DEMOTE_AFTER_REFUSALS ? 1 : 0;
    const demotedB = stateOf(b.probe.provider.id).consecRefusals >= DEMOTE_AFTER_REFUSALS ? 1 : 0;
    if (demotedA !== demotedB) return demotedA - demotedB;
    if (b.rating !== a.rating) return b.rating - a.rating;
    const calls = stateOf(a.probe.provider.id).calls - stateOf(b.probe.provider.id).calls;
    if (calls !== 0) return calls;
    const providers = (order.get(a.probe.provider.id) ?? 0) - (order.get(b.probe.provider.id) ?? 0);
    if (providers !== 0) return providers;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
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

/** An OpenAI client's body for an openai-wire provider: untouched but for the model, plus usage on streams. */
function openAIUpstreamBody(body: Json, model: string, stream: boolean): Json {
  const out: Json = { ...body, model };
  if (stream && !(isRecord(body.stream_options) && body.stream_options.include_usage === true)) {
    out.stream_options = { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true };
  }
  return out;
}

interface UpstreamCall {
  ok: boolean;
  status: number;
  response: Response | null;
}

/** Whether a status is a provider refusing: over budget, on its failure schedule, out, or unreachable. */
function isRefusal(status: number): boolean {
  return status === 0 || status === 402 || status === 429 || status >= 500;
}

/**
 * One inference call to a candidate, in the provider's own wire, counted against the provider's state
 * either way it lands. The request is the client's, rewritten only as the wire demands: same-wire
 * bodies pass through with the routed model named; cross-wire bodies are lib/wire.ts's translations.
 */
async function callUpstream(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamCall> {
  const provider = candidate.probe.provider;
  const state = stateOf(provider.id);
  state.calls += 1;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientWire === 'openai' ? openAIUpstreamBody(body, candidate.model.id, stream) : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    payload = clientWire === 'openai' ? openAIToGeminiRequest(body, candidate.model.id) : anthropicToGeminiRequest(body, candidate.model.id);
  }
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (response.ok) {
      state.successes += 1;
      state.consecRefusals = 0;
      return { ok: true, status: response.status, response };
    }
    state.consecRefusals += 1;
    try {
      await response.text();
    } catch {
      // The refusal's body does not matter; draining it only frees the socket.
    }
    return { ok: false, status: response.status, response: null };
  } catch {
    state.consecRefusals += 1;
    return { ok: false, status: 0, response: null };
  }
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

/** Answer the client from the one successful upstream call, in the client's wire. */
async function answerWith(candidate: Candidate, upstream: Response, clientWire: 'openai' | 'anthropic', stream: boolean, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  if (wire === clientWire) {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (stream) {
    const text = await upstream.text();
    const answer: NormAnswer =
      wire === 'openai' ? normFromOpenAIStream(text) : wire === 'anthropic' ? normFromAnthropicStream(text) : normFromGeminiStream(text);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(clientWire === 'openai' ? renderOpenAISse(answer, candidate.model.id) : renderAnthropicSse(answer, candidate.model.id));
    return;
  }
  const parsed: unknown = await upstream.json().catch(() => null);
  const answer: NormAnswer = !isRecord(parsed)
    ? { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] }
    : wire === 'openai'
      ? normFromOpenAI(parsed)
      : wire === 'anthropic'
        ? normFromAnthropic(parsed)
        : normFromGemini(parsed);
  sendJson(res, 200, clientWire === 'openai' ? renderOpenAIJson(answer, candidate.model.id) : renderAnthropicJson(answer, candidate.model.id));
}

/**
 * Serve one client request: walk the ranked candidates, failing over on refusals, until one provider
 * answers. Each provider is tried at most twice per request — its first call plus, when its record
 * was clean, one immediate retry of the same model — and never again once another candidate has been
 * reached for this request.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const stream = body.stream === true;
  const candidates = candidatesFor(requires, clientWire, body);
  if (candidates.length === 0) {
    if (clientWire === 'anthropic') return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const attempted = new Set<string>();
  for (const candidate of candidates) {
    const provider = candidate.probe.provider;
    if (attempted.has(provider.id)) continue;
    attempted.add(provider.id);
    const state = stateOf(provider.id);
    const healthy = state.successes > 0 && state.consecRefusals === 0;
    let call = await callUpstream(candidate, clientWire, body, stream);
    if (!call.ok && healthy && isRefusal(call.status)) {
      call = await callUpstream(candidate, clientWire, body, stream);
    }
    if (!call.ok || call.response === null) continue;
    await answerWith(candidate, call.response, clientWire, stream, res);
    return;
  }
  if (clientWire === 'anthropic') return sendError(res, 'anthropic', 503, 'api_error', 'every free provider refused this request');
  return sendError(res, 'openai', 503, 'upstream_refused', 'every free provider refused this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
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
