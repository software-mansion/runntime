/** Text generation task: an instruction-tuned decoder LLM (Gemma 3, Qwen 3)
 *  on the initRunntime() device. Chat messages or a prompt in, generated text out,
 *  with streaming, sampling, per-step logit masking and KV-cache reuse across
 *  calls that share a prefix (the turns of one conversation). */

import {
  cachedRangeSource,
  chunkedSource,
  defaultRoot,
  httpRangeSource,
  inGpuErrorScopes,
  parseSafetensorsHeader,
  RunntimeError,
  warmUp,
  type RangeSource,
} from '../../../core/index.ts';
import { asLoadError, checkpointMismatch, rethrowRunError } from '../../errors.ts';
import { fetchJson, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import type { DecoderConfig, Json } from '../../decoder/config.ts';
import { loadDecoderEngine, MAX_BURST, type DecoderEngine } from '../../decoder/engine.ts';
import { rng, sampleToken } from '../../decoder/sampling.ts';
import {
  createTextTokenizer,
  type ChatMessage,
  type ChatTemplateOptions,
  type TextTokenizer,
} from '../../decoder/tokenizer.ts';
import { gemma3ConfigFromJson, type Gemma3Config } from '../../gemma3/config.ts';
import { qwen3ConfigFromJson, type Qwen3Config } from '../../qwen3/config.ts';
import { models } from '../../models.ts';

export type {
  ChatMessage,
  ChatTemplateOptions,
  DecoderConfig,
  Gemma3Config,
  Qwen3Config,
  TextTokenizer,
};

/** Parses a Hugging Face config.json of a supported decoder (model_type
 *  gemma3 / gemma3_text / qwen3). Throws CHECKPOINT_MISMATCH for any other. */
export function decoderConfigFromJson(json: Json): DecoderConfig {
  const type = json.model_type;
  if (type === 'qwen3') return qwen3ConfigFromJson(json);
  if (type === 'gemma3' || type === 'gemma3_text' || type === undefined) {
    return gemma3ConfigFromJson(json);
  }
  throw checkpointMismatch('text generation config', `model_type '${String(type)}' unsupported`);
}

/** What to load. */
export interface TextGeneratorModel {
  /** The safetensors checkpoint (bf16, f16 or f32; uploaded as bf16). */
  readonly modelPath: ModelPath;
  /** The Hugging Face tokenizer.json. */
  readonly tokenizerPath: string;
  /** Architecture: a parsed config, or the URL of a config.json. */
  readonly config: DecoderConfig | string;
  /** Cache key prefix for the files; change it when the files change. */
  readonly cacheId?: string;
}

export interface TextGeneratorOptions extends LoadOptions {
  /** Longest prompt + output, in tokens, the KV cache holds. Default 4096. */
  maxContext?: number;
  /** Prompt tokens per prefill step. Default 128. */
  prefillChunk?: number;
}

export interface LogitsContext {
  /** 0 for the first generated token. */
  step: number;
  /** Ids generated so far in this call. */
  generatedIds: readonly number[];
  /** Text generated so far in this call. */
  text: string;
}

export interface GenerateOptions {
  /** Default 256. */
  maxNewTokens?: number;
  /** 0 (default) is greedy decoding. */
  temperature?: number;
  /** Sampling only: keep the k most likely tokens. Default 64. */
  topK?: number;
  /** Sampling only: nucleus cut-off. Default 0.95. */
  topP?: number;
  /** Sampling only: seed for a reproducible draw. */
  seed?: number;
  signal?: AbortSignal;
  /** Called with each newly decoded piece of text. Return false to stop. */
  onToken?: (text: string, tokenId: number) => void | boolean;
  /** Generation stops when the text contains one of these; it is cut off. */
  stop?: readonly string[];
  /** Edits the next-token logits in place (set -Infinity to forbid a token),
   *  e.g. for JSON-constrained decoding. Runs before sampling, each step. */
  logitsProcessor?: (logits: Float32Array, ctx: LogitsContext) => void;
  /** Restricts the next token to these ids, each step (a fixed set or a
   *  per-step callback). */
  allowedTokenIds?: Iterable<number> | ((ctx: LogitsContext) => Iterable<number> | undefined);
  /** A string prompt is wrapped in the chat template as one user turn unless
   *  `raw` is true (then it is tokenized as-is, with `<bos>` prepended where
   *  the model has one). */
  raw?: boolean;
  /** Options of the chat template (Qwen 3: `enableThinking`, `tools`). */
  chatTemplate?: ChatTemplateOptions;
  /** Reuse the KV cache of the previous call's shared prefix. Default true. */
  reuseCache?: boolean;
  /** Greedy decoding without logitsProcessor/allowedTokenIds picks tokens on
   *  the GPU and runs this many steps per GPU→CPU round trip (up to 32). A
   *  burst may overshoot a stop token; the extra steps are discarded.
   *  Default: 1 where a round trip is cheap, 16 where it is slow (Firefox). */
  decodeBurst?: number;
}

export interface GenerationStats {
  promptTokens: number;
  /** Prompt tokens served from the previous call's KV cache. */
  reusedTokens: number;
  prefillMs: number;
  generatedTokens: number;
  /** From the first generated token to the last. */
  decodeMs: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
  finishReason: 'stop' | 'length' | 'context' | 'callback' | 'aborted';
}

export interface TextGenerator {
  readonly tokenizer: TextTokenizer;
  readonly config: DecoderConfig;
  readonly maxContext: number;
  /** Generates a reply. Calls run one at a time. */
  generate(input: string | readonly ChatMessage[], opts?: GenerateOptions): Promise<string>;
  /** Next-token logits for a prompt (chat-formatted like generate()), for
   *  inspection and testing. */
  logits(
    input: string | readonly ChatMessage[],
    opts?: { raw?: boolean; chatTemplate?: ChatTemplateOptions },
  ): Promise<Float32Array>;
  /** Stats of the last finished generate(). */
  readonly lastStats: GenerationStats | undefined;
  /** Forgets the KV cache, so the next call prefills from scratch. */
  resetCache(): void;
  dispose(): void;
}

const isMessages = (v: unknown): v is readonly ChatMessage[] =>
  Array.isArray(v) &&
  v.every(
    (m: unknown) =>
      typeof m === 'object' &&
      m !== null &&
      typeof (m as ChatMessage).role === 'string' &&
      typeof (m as ChatMessage).content === 'string',
  );

/** Length of the longest end of `text` that a stop string starts with
 *  (shorter than the whole stop string, which would have matched). */
export function stopPrefixLength(text: string, stop: readonly string[] | undefined): number {
  let n = 0;
  for (const s of stop ?? []) {
    for (let k = Math.min(s.length - 1, text.length); k > n; k--) {
      if (text.endsWith(s.slice(0, k))) {
        n = k;
        break;
      }
    }
  }
  return n;
}

/** Opens a safetensors file for byte-range reads, with the same network
 *  coalescing and cache layout as the other zoo loaders. */
async function openSafetensors(modelPath: ModelPath, opts: LoadOptions, cacheId: string) {
  const raw: RangeSource =
    typeof modelPath === 'string'
      ? httpRangeSource(modelPath, { onBytes: opts.onBytes })
      : modelPath;
  const cached = (inner: RangeSource) =>
    opts.cache ? cachedRangeSource(inner, opts.cache, cacheId) : inner;
  const index = await parseSafetensorsHeader(cached(raw));
  let payloadEnd = 0;
  for (const info of index.tensors.values()) payloadEnd = Math.max(payloadEnd, info.end);
  return { tensors: index.tensors, source: cached(chunkedSource(raw, payloadEnd)) };
}

/** Loads a decoder LLM (weights, tokenizer), warms the GPU path once and
 *  returns the generator. Default: Gemma 3 1B instruction-tuned, bf16 (2 GB). */
export async function createTextGenerator(
  model: TextGeneratorModel = models.textGeneration.GEMMA3_1B_IT,
  opts: TextGeneratorOptions = {},
): Promise<TextGenerator> {
  try {
    return await loadTextGenerator(model, opts);
  } catch (err) {
    throw asLoadError(err, opts.signal);
  }
}

/** Checks a size option: undefined, or an integer in min..max. */
function checkSize(name: string, value: number | undefined, min: number, max = Infinity) {
  if (value === undefined) return;
  if (!Number.isInteger(value) || value < min || value > max) {
    const range = max === Infinity ? `≥ ${min}` : `in ${min}..${max}`;
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `${name} must be an integer ${range}, got ${value}`,
    );
  }
}

async function loadTextGenerator(
  model: TextGeneratorModel,
  opts: TextGeneratorOptions,
): Promise<TextGenerator> {
  checkSize('maxContext', opts.maxContext, 1);
  checkSize('prefillChunk', opts.prefillChunk, 1);
  const device = defaultRoot().device;
  const base =
    opts.cacheId ??
    model.cacheId ??
    (typeof model.modelPath === 'string'
      ? model.modelPath.replace(/^https?:\/\//, '').replace(/\/model\.safetensors$/, '')
      : undefined);
  if (opts.cache && !base) {
    throw new RunntimeError('INVALID_ARGUMENT', 'createTextGenerator: a cache needs a cacheId');
  }
  const cfg =
    typeof model.config === 'string'
      ? decoderConfigFromJson(
          await fetchJson(model.config, { ...opts, cacheId: base && `${base}/config.json` }),
        )
      : model.config;
  const tokenizer = createTextTokenizer(
    await fetchJson(model.tokenizerPath, { ...opts, cacheId: base && `${base}/tokenizer.json` }),
    { chatFormat: cfg.arch === 'gemma3' ? 'gemma' : 'qwen3' },
  );
  throwIfAborted(opts.signal);
  const weights = await openSafetensors(model.modelPath, opts, `${base}/model.safetensors`);
  const engine: DecoderEngine = await inGpuErrorScopes(device, `${cfg.arch} load`, () =>
    loadDecoderEngine(device, cfg, weights, {
      maxContext: opts.maxContext,
      maxChunk: opts.prefillChunk,
      onProgress: opts.onProgress,
      signal: opts.signal,
    }),
  );

  // Ids whose K/V sit in the cache at positions 0..cached.length-1.
  let cached: number[] = [];
  // One past the furthest position written since the last full refill.
  // Writes past cached.length (a burst's overshoot) still evict ring slots
  // of the sliding-window layers, so rewinds are bounded from here.
  let highWater = 0;
  let defaultBurst = 1;
  try {
    const first = tokenizer.bosId ?? tokenizer.eosId;
    throwIfAborted(opts.signal);
    await warmUp(device, () => engine.forward([first], 0, true), `${cfg.arch} warm-up`);
    cached = [first];
    highWater = 1;
    // GPU→CPU round-trip latency decides the default greedy burst: ~0.1 ms
    // in Chrome, ~100 ms in Firefox, where per-token readback would dominate.
    let rt = Infinity;
    for (let i = 0; i < 3; i++) {
      const t = performance.now();
      device.queue.submit([]);
      await device.queue.onSubmittedWorkDone();
      rt = Math.min(rt, performance.now() - t);
    }
    defaultBurst = rt < 2 ? 1 : 16;
  } catch (err) {
    engine.dispose();
    throw err;
  }

  const softcap = cfg.finalLogitSoftcap;
  const toIds = (
    input: string | readonly ChatMessage[],
    raw?: boolean,
    template?: ChatTemplateOptions,
  ): number[] => {
    if (typeof input === 'string' && raw) return tokenizer.encode(input, { addBos: true });
    const messages: readonly ChatMessage[] =
      typeof input === 'string' ? [{ role: 'user', content: input }] : input;
    return tokenizer.encode(tokenizer.applyChatTemplate(messages, template));
  };

  /** Feeds ids so the cache holds exactly `ids`, reusing the shared prefix;
   *  resolves to the logits after the last id. */
  const prefill = async (ids: number[], reuse: boolean, signal?: AbortSignal) => {
    if (ids.length > engine.maxContext) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `generate: prompt is ${ids.length} tokens, maxContext is ${engine.maxContext}`,
      );
    }
    let p = 0;
    if (reuse) {
      while (p < cached.length && p < ids.length && cached[p] === ids[p]) p++;
      if (p === ids.length) p--; // the last id must run to produce logits
      if (highWater - p > engine.maxRewind) p = 0;
    }
    const reused = p;
    cached.length = p;
    if (p === 0) highWater = 0;
    let logits: Float32Array | undefined;
    while (p < ids.length) {
      if (signal?.aborted) throw signal.reason;
      const part = ids.slice(p, p + engine.maxChunk);
      const last = p + part.length >= ids.length;
      if (last) logits = await engine.forward(part, p, true);
      else await engine.forward(part, p, false);
      cached.push(...part);
      p += part.length;
      highWater = Math.max(highWater, p);
    }
    return { logits: logits!, reused };
  };

  const prepare = (logits: Float32Array) => {
    if (softcap > 0) {
      for (let i = 0; i < logits.length; i++) logits[i] = Math.tanh(logits[i]! / softcap) * softcap;
    }
  };

  let lastStats: GenerationStats | undefined;

  const run = async (
    input: string | readonly ChatMessage[],
    o: GenerateOptions,
  ): Promise<string> => {
    checkSize('generate: maxNewTokens', o.maxNewTokens, 1);
    checkSize('generate: topK', o.topK, 1);
    checkSize('generate: decodeBurst', o.decodeBurst, 1, MAX_BURST);
    if (o.temperature !== undefined && !(o.temperature >= 0)) {
      throw new RunntimeError('INVALID_ARGUMENT', 'generate: temperature must be ≥ 0');
    }
    if (o.topP !== undefined && !(o.topP > 0 && o.topP <= 1)) {
      throw new RunntimeError('INVALID_ARGUMENT', 'generate: topP must be in (0, 1]');
    }
    const ids = toIds(input, o.raw, o.chatTemplate);
    const maxNew = o.maxNewTokens ?? 256;
    const temperature = o.temperature ?? 0;
    const random = o.seed !== undefined ? rng(o.seed) : Math.random;
    const stopIds = new Set(tokenizer.stopIds);
    const stream = tokenizer.createStreamDecoder();
    const t0 = performance.now();
    let { logits, reused } = await prefill(ids, o.reuseCache ?? true, o.signal);
    const t1 = performance.now();
    const generated: number[] = [];
    let text = '';
    // Chars of `text` passed to onToken so far.
    let streamed = 0;
    let cutAtStop = false;
    let finishReason: GenerationStats['finishReason'] = 'length';
    let tFirst = t1;
    // Greedy with no CPU-side logit hooks: argmax on the GPU, in bursts.
    const gpuGreedy = temperature <= 0 && !o.logitsProcessor && o.allowedTokenIds === undefined;
    const burst = o.decodeBurst ?? defaultBurst;
    // Burst output not consumed yet; `fed` ids already have their K/V cached.
    let pending: { id: number; fed: boolean }[] = [];
    try {
      for (let step = 0; step < maxNew; step++) {
        if (o.signal?.aborted) {
          finishReason = 'aborted';
          break;
        }
        let tok: number;
        let fed = false;
        if (step > 0 && gpuGreedy) {
          const next = pending.shift()!;
          tok = next.id;
          fed = next.fed;
        } else {
          prepare(logits);
          const ctx: LogitsContext = { step, generatedIds: generated, text };
          o.logitsProcessor?.(logits, ctx);
          const allowed =
            typeof o.allowedTokenIds === 'function' ? o.allowedTokenIds(ctx) : o.allowedTokenIds;
          const cands = allowed === undefined ? undefined : Array.from(allowed);
          tok = sampleToken(logits, cands, temperature, o.topK ?? 64, o.topP ?? 0.95, random);
        }
        if (step === 0) tFirst = performance.now();
        if (stopIds.has(tok)) {
          finishReason = 'stop';
          break;
        }
        generated.push(tok);
        const piece = stream.push(tok);
        text += piece;
        if (o.stop?.length) {
          const hit = o.stop
            .map((s) => text.indexOf(s, Math.max(0, text.length - piece.length - s.length)))
            .filter((i) => i >= 0);
          if (hit.length > 0) {
            text = text.slice(0, Math.min(...hit));
            cutAtStop = true;
            if (text.length > streamed) o.onToken?.(text.slice(streamed), tok);
            finishReason = 'stop';
            break;
          }
        }
        // Text that may begin a stop string waits until the next pieces
        // settle it, so the streamed text always equals the returned one.
        const upTo = text.length - stopPrefixLength(text, o.stop);
        if (upTo > streamed) {
          const out = text.slice(streamed, upTo);
          streamed = upTo;
          if (o.onToken?.(out, tok) === false) {
            finishReason = 'callback';
            break;
          }
        }
        if (cached.length >= engine.maxContext) {
          finishReason = 'context';
          break;
        }
        if (step + 1 >= maxNew) break; // no need to run the last token
        if (fed) {
          cached.push(tok);
        } else if (gpuGreedy) {
          const pos = cached.length;
          const n = Math.min(burst, maxNew - step - 1, engine.maxContext - pos);
          const out = await engine.decodeGreedy(tok, pos, n);
          highWater = Math.max(highWater, pos + n);
          cached.push(tok);
          pending = out.map((id, k) => ({ id, fed: k < n - 1 }));
        } else {
          logits = await engine.forward([tok], cached.length, true);
          cached.push(tok);
          highWater = Math.max(highWater, cached.length);
        }
      }
      if (!cutAtStop) text += stream.flush();
      const last = generated[generated.length - 1];
      const streaming = finishReason !== 'callback' && finishReason !== 'aborted';
      if (streaming && text.length > streamed && last !== undefined) {
        o.onToken?.(text.slice(streamed), last);
      }
    } finally {
      const t2 = performance.now();
      const n = generated.length;
      lastStats = {
        promptTokens: ids.length,
        reusedTokens: reused,
        prefillMs: tFirst - t0,
        generatedTokens: n,
        decodeMs: t2 - tFirst,
        prefillTokensPerSecond: ((ids.length - reused) / Math.max(1e-3, tFirst - t0)) * 1000,
        decodeTokensPerSecond: n > 1 ? ((n - 1) / Math.max(1e-3, t2 - tFirst)) * 1000 : 0,
        finishReason,
      };
    }
    if (finishReason === 'aborted') throw o.signal!.reason;
    return text;
  };

  let inFlight: Promise<unknown> = Promise.resolve();
  let disposed = false;
  // Calls run one after another: they share the KV cache. Errors become
  // RunntimeErrors, except an abort, which rejects with the signal's reason
  // like fetch() does.
  const serialize = async <T>(
    input: unknown,
    job: () => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> => {
    if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'text generator is disposed');
    if (typeof input !== 'string' && !isMessages(input)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        'generate: input must be a string or an array of { role, content } messages',
      );
    }
    const p = inFlight.then(job);
    inFlight = p.catch(() => undefined);
    return p.catch((err: unknown) => {
      if (signal?.aborted && err === signal.reason) throw err;
      return rethrowRunError(err);
    });
  };

  const generator: TextGenerator = {
    tokenizer,
    config: cfg,
    maxContext: engine.maxContext,
    generate: (input, o = {}) => serialize(input, () => run(input, o), o.signal),
    logits: (input, o = {}) =>
      serialize(input, async () => {
        const l = (await prefill(toIds(input, o.raw, o.chatTemplate), true)).logits;
        prepare(l);
        return l;
      }),
    get lastStats() {
      return lastStats;
    },
    resetCache() {
      cached = [];
      highWater = 0;
    },
    dispose() {
      disposed = true;
      void inFlight.then(() => engine.dispose());
    },
  };
  // Undocumented debugging hook (per-op profiling), not part of the API.
  Object.defineProperty(generator, '_engine', { value: engine, enumerable: false });
  return generator;
}
