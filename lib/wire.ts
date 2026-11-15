// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------------------------------
// The other two upstream wires, and normalized answers.
//
// The baseline above speaks to openai-wire providers only. What follows completes the matrix: an
// OpenAI-shaped client body rewritten for an anthropic-wire provider, and both client bodies rewritten
// for a gemini-wire provider. Every translation is built to preserve exactly what a provider reads out
// of a request — its system text, its last user message's text and image, its tool names in order, its
// output cap, its stop sequences and its JSON mode — because the provider echoes those fields and the
// client scores the echo.
//
// Answers come back in one normalized shape (text, usage, stop reason, tool calls) parsed from any of
// the three wires, as JSON or as a buffered event stream, and are rendered into whichever client wire
// asked, with the stop reason mapped across the families (end_turn <-> stop, max_tokens <-> length,
// tool_use <-> tool_calls, refusal <-> content_filter).

/** One tool call in the normalized answer shape; `arguments` is one JSON string. */
export interface NormToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** One provider answer, whichever wire carried it: the whole text, the usage pair, the stop reason. */
export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The stop reason in the upstream wire's own spelling ('stop', 'end_turn', 'STOP', ...). */
  stop: string | null;
  toolCalls: NormToolCall[];
}

const TO_OPENAI_STOP: Record<string, string> = {
  stop: 'stop', end_turn: 'stop', STOP: 'stop',
  length: 'length', max_tokens: 'length', MAX_TOKENS: 'length',
  tool_calls: 'tool_calls', function_call: 'tool_calls', tool_use: 'tool_calls',
  content_filter: 'content_filter', refusal: 'content_filter', SAFETY: 'content_filter',
};

/** A stop reason from any wire as OpenAI's finish_reason. */
export function toOpenAIStop(stop: string | null): string {
  return (stop !== null && TO_OPENAI_STOP[stop]) || 'stop';
}

const TO_ANTHROPIC_STOP: Record<string, string> = {
  stop: 'end_turn', STOP: 'end_turn', end_turn: 'end_turn',
  length: 'max_tokens', MAX_TOKENS: 'max_tokens', max_tokens: 'max_tokens',
  tool_calls: 'tool_use', function_call: 'tool_use', tool_use: 'tool_use',
  content_filter: 'refusal', SAFETY: 'refusal', refusal: 'refusal',
};

/** A stop reason from any wire as Anthropic's stop_reason. */
export function toAnthropicStop(stop: string | null): string {
  return (stop !== null && TO_ANTHROPIC_STOP[stop]) || 'end_turn';
}

/** The whole text of an answer's content: a string as itself, else its text blocks joined end to end. */
function wholeText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** A Chat Completions JSON answer in the normalized shape. */
export function normFromOpenAI(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: NormToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: wholeText(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    toolCalls,
  };
}

/** A Messages JSON answer in the normalized shape. */
export function normFromAnthropic(body: Json): NormAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: NormToolCall[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    toolCalls,
  };
}

/** A Gemini generateContent JSON answer in the normalized shape. */
export function normFromGemini(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: NormToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    toolCalls,
  };
}

interface SseFrame {
  event: string | null;
  data: string;
}

/** The frames of one whole SSE body: event name (null when unnamed) and its data lines joined. */
function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function frameJson(data: string): Json | null {
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A whole Chat Completions event stream in the normalized shape. */
export function normFromOpenAIStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const toolCalls: NormToolCall[] = [];
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = frameJson(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : toolCalls.length;
      while (toolCalls.length <= index) toolCalls.push({ id: '', name: '', arguments: '' });
      const entry = toolCalls[index] as NormToolCall;
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** A whole Messages event stream in the normalized shape. */
export function normFromAnthropicStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const toolByIndex = new Map<number, NormToolCall>();
  for (const frame of sseFrames(text)) {
    const data = frameJson(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : toolByIndex.size;
      toolByIndex.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        arguments: '',
      });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const entry = typeof data.index === 'number' ? toolByIndex.get(data.index) : undefined;
        if (entry !== undefined) entry.arguments += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  return {
    text: out,
    inputTokens,
    outputTokens,
    stop,
    toolCalls: [...toolByIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v),
  };
}

/** A whole Gemini streamGenerateContent event stream in the normalized shape. */
export function normFromGeminiStream(text: string): NormAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const toolCalls: NormToolCall[] = [];
  for (const frame of sseFrames(text)) {
    const body = frameJson(frame.data);
    if (body === null) continue;
    const piece = normFromGemini(body);
    out += piece.text;
    toolCalls.push(...piece.toolCalls);
    if (piece.stop !== null) stop = piece.stop;
    if (isRecord(body.usageMetadata)) {
      inputTokens = piece.inputTokens;
      outputTokens = piece.outputTokens;
    }
  }
  return { text: out, inputTokens, outputTokens, stop, toolCalls };
}

/** A normalized answer as a Chat Completions JSON body naming `model`. */
export function renderOpenAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toOpenAIStop(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalized answer as a Messages JSON body naming `model`. */
export function renderAnthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregate', name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toAnthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalized answer as a Chat Completions event stream: text, finish, usage, [DONE]. */
export function renderOpenAISse(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  if (answer.text !== '') frames.push(sseEvent(null, chunk({ content: answer.text }, null)));
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(null, chunk({ tool_calls: [{ index, id: call.id !== '' ? call.id : `call_${index}`, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null)),
    );
  });
  frames.push(sseEvent(null, chunk({}, toOpenAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as a Messages event stream, in the canonical event order. */
export function renderAnthropicSse(answer: NormAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    if (answer.text !== '') frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } }));
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: toAnthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** A message content's text the way a provider reads it: a string as itself, else its text blocks joined by newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The system text of an OpenAI request: its system messages' texts, the empty ones dropped, joined by newlines. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = blocksText(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** A `data:` URL's media type and base64 payload, or null when `url` is not a base64 data URL. */
function dataUrl(url: string): { media: string; data: string } | null {
  const match = /^data:([^,]*),(.*)$/s.exec(url);
  if (match === null || !(match[1] as string).endsWith(';base64')) return null;
  const media = (match[1] as string).slice(0, -';base64'.length);
  return { media: media !== '' ? media : 'image/png', data: match[2] as string };
}

/** A stop field as a list: a lone string is a one-item list. */
function stopListOf(stop: unknown): string[] {
  if (typeof stop === 'string') return [stop];
  if (Array.isArray(stop)) return stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** An OpenAI request's output cap: max_tokens, else max_completion_tokens, else none. */
function openAIMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: system messages
 * fold into the system prompt, image_url parts become image blocks (base64 data URLs stay base64, other
 * URLs stay URLs), tool calls become tool_use blocks, tool answers become tool_result blocks folded
 * into the preceding user message (so the last user text a provider reads is the client's own), and
 * the output cap, stop sequences, tools and tool choice are rewritten in the Messages spelling.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const out: Json = { model, messages: [] };
  const system = openAISystemText(body);
  if (system !== '') out.system = system;
  const messages: Json[] = [];
  const pushUserBlocks = (blocks: Json[]): void => {
    const last = messages[messages.length - 1];
    if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...blocks);
      return;
    }
    if (last !== undefined && last.role === 'user' && typeof last.content === 'string') {
      last.content = [{ type: 'text', text: last.content }, ...blocks];
      return;
    }
    messages.push({ role: 'user', content: blocks });
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'user') {
      if (typeof raw.content === 'string') {
        messages.push({ role: 'user', content: raw.content });
        continue;
      }
      const blocks: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const parsed = dataUrl(part.image_url.url);
            if (parsed !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: parsed.media, data: parsed.data } });
            else blocks.push({ type: 'image', source: { type: 'url', url: part.image_url.url } });
          }
        }
      }
      messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      pushUserBlocks([{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: blocksText(raw.content) }]);
    }
  }
  out.messages = messages;
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text, or an image inline (data URL) or by reference; null for the rest. */
function geminiPartFromOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const parsed = dataUrl(part.image_url.url);
    if (parsed !== null) return { inlineData: { mimeType: parsed.media, data: parsed.data } };
    return { fileData: { fileUri: part.image_url.url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request (the model rides in the
 * route, not the body): system messages fold into systemInstruction, assistant turns become model
 * contents, tool calls become functionCall parts, tool answers become functionResponse parts folded
 * into the preceding user content (so the last user text a provider reads is the client's own), and
 * the output cap, stop sequences and JSON mode land in generationConfig — response_format becomes
 * responseMimeType application/json, the one JSON-mode spelling the Gemini wire carries.
 */
export function openAIToGeminiRequest(body: Json, _model: string): Json {
  const out: Json = {};
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const toolNameById = new Map<string, string>();
  let lastUser: Json | null = null;
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') parts.push({ text: raw.content });
      else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const mapped = geminiPartFromOpenAI(part);
          if (mapped !== null) parts.push(mapped);
        }
      }
      const content: Json = { role: 'user', parts };
      contents.push(content);
      lastUser = content;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = blocksText(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        if (typeof call.id === 'string' && typeof call.function.name === 'string') toolNameById.set(call.id, call.function.name);
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const name = (typeof raw.tool_call_id === 'string' && toolNameById.get(raw.tool_call_id)) || 'tool';
      const part: Json = { functionResponse: { name, response: { result: blocksText(raw.content) } } };
      if (lastUser !== null && Array.isArray(lastUser.parts)) (lastUser.parts as Json[]).push(part);
      else {
        const content: Json = { role: 'user', parts: [part] };
        contents.push(content);
        lastUser = content;
      }
    }
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  const config: Json = {};
  const maxTokens = openAIMaxTokens(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = stopListOf(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request (the model rides in the route):
 * the system prompt becomes systemInstruction, image blocks become inlineData or fileData parts,
 * tool_use blocks become functionCall parts, tool_result blocks become functionResponse parts in
 * their own user content (a tool-only user turn reads as empty text on both wires), and the output
 * cap and stop sequences land in generationConfig. No responseMimeType is ever set: a Messages
 * request carries no JSON mode for a provider to echo.
 */
export function anthropicToGeminiRequest(body: Json, _model: string): Json {
  const out: Json = {};
  const system = blocksText(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const toolNameById = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') parts.push({ text: raw.content });
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (!assistant && block.type === 'image' && isRecord(block.source)) {
          const sourceBlock = block.source;
          if (sourceBlock.type === 'base64' && typeof sourceBlock.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof sourceBlock.media_type === 'string' ? sourceBlock.media_type : 'image/png', data: sourceBlock.data } });
          } else if (sourceBlock.type === 'url' && typeof sourceBlock.url === 'string') {
            parts.push({ fileData: { fileUri: sourceBlock.url } });
          }
        } else if (assistant && block.type === 'tool_use') {
          if (typeof block.id === 'string' && typeof block.name === 'string') toolNameById.set(block.id, block.name);
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (!assistant && block.type === 'tool_result') {
          const name = (typeof block.tool_use_id === 'string' && toolNameById.get(block.tool_use_id)) || 'tool';
          parts.push({ functionResponse: { name, response: { result: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (choice.type === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (choice.type === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) {
    const stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (stops.length > 0) config.stopSequences = stops;
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}
