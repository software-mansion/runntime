/** Reverb generation: a prefix pass fills the key/value cache, the main and
 *  polish passes over the target frames pick the codes, the codec decoder
 *  turns them into audio.
 *  Each pass reads back the best code and its log-probability per slot, and
 *  the sampler on the CPU picks from those. */

import type { TgpuRoot } from 'typegpu';
import {
  cat,
  createStorageFor,
  evalValues,
  gpuExecutor,
  logitStats,
  materialized,
  RunntimeError,
  tensor,
  toArray,
  type Value,
  type ValueMeta,
} from '../../core/index.ts';
import { ropeTables } from '../shared/rope.ts';
import type { HiggsDecoder } from '../higgs/decoder.ts';
import { COARSE_CODEBOOKS, MASK, type ReverbLanguage } from './config.ts';
import type { ReverbCall, ReverbModel } from './model.ts';
import { headRows } from './stateDictHooks.ts';
import {
  mainPassUpdate,
  POLISH_BIAS,
  POLISH_GUIDANCE,
  polishFirst,
  polishSecond,
  seededRandom,
  type SlotStats,
} from './sampler.ts';
import { reverbPrompt, estimateFrames } from './text.ts';
import type { ReverbTokenizer } from './tokenizer.ts';

/** Main passes per generation. */
export const DEFAULT_STEPS = 8;

/** A voice: a reference clip's codec codes [codebooks, frames], row-major,
 *  with its transcript and language. */
export interface Voice {
  readonly codes: Int32Array;
  readonly frames: number;
  readonly transcript: string;
  /** The language of the clip and of the text it speaks. */
  readonly language: ReverbLanguage;
  /** How many of those frames the speaker spends speaking: no silent edges,
   *  pauses capped at 0.5 s (speechFrames() in text.ts). The target's length
   *  follows it; without it, the whole clip. */
  readonly speechFrames?: number;
}

export interface ReverbGenerateOptions {
  /** Main passes. Default 8. */
  readonly steps?: number;
  /** Seeds the commit noise. Default 0. */
  readonly seed?: number;
}

/** One prompt's filled cache, and passes over its target frames. */
export interface ReverbSession {
  readonly frames: number;
  /** Best codes and their log-probabilities [books, frames] for target
   *  tokens [codebooks, frames]: codebooks 0–1, or with `fine` 2–7 and the
   *  polish bias towards the reference frames. A fine pass keeps its logits:
   *  the next `guided` one picks against them. */
  pass(tokens: Int32Array, opts: { fine: boolean; guided?: boolean }): Promise<SlotStats>;
  dispose(): void;
}

export interface ReverbSpeaker {
  /** Fills the cache with the prefix of `text` in `voice`. */
  open(text: string, voice: Voice): ReverbSession;
  /** Codes [codebooks, frames] of `text` spoken in `voice`. */
  generate(
    text: string,
    voice: Voice,
    opts?: ReverbGenerateOptions,
  ): Promise<{ tokens: Int32Array; frames: number }>;
  /** The codec's samples for codes [codebooks, frames]. */
  decode(tokens: Int32Array, frames: number): Promise<Float32Array>;
  dispose(): void;
}

export function createReverbSpeaker(
  root: TgpuRoot,
  model: ReverbModel,
  codec: HiggsDecoder,
  tokenizer: ReverbTokenizer,
): ReverbSpeaker {
  const { heads, kvHeads, headDim, ffn, codebooks, codes, ropeTheta } = model.cfg;
  const act = model.llm.norm.weight.shape.dtype === 'f16' ? 'f16' : 'f32';
  // Read off the bias: a quantized conv weight is typed quantW.
  const codecAct = codec.acoustic_decoder.conv1.bias!.shape.dtype === 'f16' ? 'f16' : 'f32';
  // A pass's widest buffer has one row per input row: the fine logits, the
  // up and gate rows or the q and k rows. Counted in f32, it must stay
  // bindable.
  const rowBytes =
    4 *
    Math.max(headRows(codebooks - COARSE_CODEBOOKS, codes), 2 * ffn, (heads + kvHeads) * headDim);
  const limit = root.device.limits.maxStorageBufferBindingSize;
  const maxRows = Math.floor(limit / rowBytes);
  // The codec decoder's widest buffer is its last block's transposed conv
  // before the slice: T · (the earlier strides) + 1 rows of stride · channels.
  const { channels, strides } = codec.cfg;
  const stride = strides.at(-1)!;
  const upsample = strides.slice(0, -1).reduce((a, b) => a * b, 1);
  const convRowBytes = (codecAct === 'f16' ? 2 : 4) * stride * (channels / 2 ** strides.length);
  const maxTarget = Math.min(
    maxRows,
    Math.floor((Math.floor(limit / convRowBytes) - 1) / upsample),
  );

  // Lengths change from call to call, so each session and decode ends with
  // a trim: the pool keeps only the buffers of the last length.
  const ex = gpuExecutor(root);
  const codecEx = gpuExecutor(root);
  const open = (text: string, voice: Voice): ReverbSession => {
    const promptIds = tokenizer.encode(reverbPrompt(voice.language, voice.transcript, text));
    const refFrames = voice.frames;
    const P = promptIds.length + refFrames;
    const pace =
      voice.speechFrames === undefined
        ? refFrames
        : Math.max(1, Math.min(voice.speechFrames, refFrames));
    const T = estimateFrames(text, voice.transcript, pace);
    const L = P + T;
    // The polish passes' key bias needs at least two target rows.
    if (T < 2) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `synthesize: the text gives ${T} frame for this voice, fewer than 2`,
      );
    }
    if (T > maxTarget) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `synthesize: the text needs ${T} frames, more than the ${maxTarget} this device runs at once`,
      );
    }
    if (P > maxRows) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `synthesize: the voice and prompt take ${P} rows, more than the ${maxRows} this device runs at once`,
      );
    }

    const owned: Value[] = [];
    // The last fine pass's logits, for the guided pass after it.
    let lastFine: Value | undefined;
    const free = () => {
      for (const v of owned.splice(0)) v.buffer.destroy();
      if (lastFine) ex.releaseValues([lastFine]);
      lastFine = undefined;
      ex.trim();
    };
    const upload = (data: Float32Array, rows: number, dtype: 'f32' | 'f16' = act) => {
      const v = tensor(root, data, { elems: data.length, dtype, dims: [rows, data.length / rows] });
      owned.push(v);
      return v;
    };
    try {
      // Every cache row is written before it is read, so it starts empty.
      const empty = () => {
        const shape: ValueMeta = {
          elems: L * kvHeads * headDim,
          dtype: act,
          dims: [L, kvHeads * headDim],
        };
        const v = materialized(shape, createStorageFor(root, shape));
        owned.push(v);
        return v;
      };
      const cache = Array.from({ length: model.llm.layers.items.length }, () => ({
        k: empty(),
        v: empty(),
      }));
      const { cos, sin } = ropeTables(L, headDim, ropeTheta);
      const ropeRows = (from: number, to: number) => ({
        cos: upload(cos.slice(from * headDim, to * headDim), to - from, 'f32'),
        sin: upload(sin.slice(from * headDim, to * headDim), to - from, 'f32'),
      });

      // Prefix: prompt rows, then reference frames. Only its cache writes run.
      const textRows = upload(model.llm.embed_tokens.table.gather(promptIds), promptIds.length);
      const x0 = cat(
        [
          model.llm.embed_tokens.project.forward(textRows),
          model.audioRows(upload(model.audioIds(voice.codes, refFrames), refFrames, 'f32')),
        ],
        0,
      );
      const written: Value[] = [];
      model.forward(x0, { ...ropeRows(0, P), cache, start: 0, written });
      evalValues(written, ex);

      const target = ropeRows(P, L);
      // The polish passes lean on the reference frames, the prefix's last rows.
      const refBias = { start: P - refFrames, end: P, value: POLISH_BIAS };
      const booksOf = (fine: boolean) => (fine ? codebooks - COARSE_CODEBOOKS : COARSE_CODEBOOKS);

      return {
        frames: T,
        async pass(tokens, { fine, guided = false }) {
          const call: ReverbCall = {
            ...target,
            cache,
            start: P,
            keyBias: fine ? refBias : undefined,
          };
          const ids = tensor(root, model.audioIds(tokens, T), [T * codebooks, 1]);
          try {
            const h = model.forward(model.audioRows(ids), call);
            const logits = (fine ? model.audio_heads.fine : model.audio_heads.coarse).forward(h);
            const guide = fine && guided ? lastFine : undefined;
            const stats = logitStats(logits, {
              blocks: booksOf(fine),
              block: codes,
              argmaxCols: MASK,
              guide,
              guideWeight: POLISH_GUIDANCE,
            });
            let out: Float32Array;
            try {
              // A coarse pass reads its stats back in the eval's own submit. A
              // fine pass also keeps its logits, so it evaluates both first.
              if (fine) evalValues([stats, logits], ex);
              out = await toArray(stats, ex);
            } finally {
              ex.releaseValues([stats, ...(guide ? [guide] : [])]);
              if (guide) lastFine = undefined;
              if (fine && !guided) {
                if (lastFine) ex.releaseValues([lastFine]);
                lastFine = logits;
              } else ex.releaseValues([logits]);
            }
            const n = booksOf(fine) * T;
            const best = new Int32Array(n);
            const confidence = new Float32Array(n);
            for (let i = 0; i < n; i++) {
              best[i] = out[2 * i]!;
              confidence[i] = out[2 * i + 1]!;
            }
            return { best, confidence };
          } finally {
            ids.buffer.destroy();
          }
        },
        dispose: free,
      };
    } catch (err) {
      free();
      throw err;
    }
  };

  return {
    open,

    async generate(text, voice, opts = {}) {
      const steps = opts.steps ?? DEFAULT_STEPS;
      const random = seededRandom(opts.seed ?? 0);
      const session = open(text, voice);
      try {
        const T = session.frames;
        const tokens = new Int32Array(codebooks * T).fill(MASK);
        for (let step = 0; step < steps; step++) {
          const coarse = await session.pass(tokens, { fine: false });
          mainPassUpdate(tokens, T, coarse, step, steps, random);
        }
        polishFirst(tokens, T, await session.pass(tokens, { fine: true }));
        polishSecond(tokens, T, await session.pass(tokens, { fine: true, guided: true }));
        return { tokens, frames: T };
      } finally {
        session.dispose();
      }
    },

    async decode(tokens, frames) {
      const { latent } = codec.cfg;
      const x = tensor(root, codec.latents(tokens, frames), {
        elems: frames * latent,
        dtype: codecAct,
        dims: [frames, latent],
      });
      let y: Value | undefined;
      try {
        y = codec.forward(x);
        return await toArray(y, codecEx);
      } finally {
        if (y) codecEx.releaseValues([y]);
        x.buffer.destroy();
        codecEx.trim();
      }
    },

    dispose() {
      ex.dispose();
      codecEx.dispose();
      model.dispose();
      codec.dispose();
    },
  };
}
