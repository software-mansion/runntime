/** d1-omni's System One call on the initRunntime() device: named questions
 *  over a text state and optional images, one trunk pass per question, no
 *  generated tokens. */

import {
  cat,
  defaultExecutor,
  evalValues,
  slice,
  tensor,
  toArray,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { preprocessImage, resizeAA, type RgbImage } from './image.ts';
import { D1, D1Model } from './model.ts';
import {
  encodeQuestion,
  optionCount,
  QTYPES,
  softmaxT,
  temperatureKey,
  toAnswer,
  YES_NO,
  type Answer,
  type D1Tokenizer,
  type Question,
} from './prompt.ts';

/** Total positions the trunk was trained on: media plus text. */
const MAX_LENGTH = 16384;
/** Text positions after an image prefix, as trained. */
const IMAGE_TEXT_LENGTH = 896;

export interface D1Decider {
  /** Answers each named question; `images` makes the call an image call. */
  systemOne(
    state: string,
    questions: Record<string, Question>,
    images?: readonly RgbImage[],
  ): Promise<{ answers: Record<string, Answer>; inputTokens: number }>;
  dispose(): void;
}

/** Pair-duplicated RoPE tables [T, headDim] for positions 0..T−1. */
function ropeTables(t: number): { cos: Float32Array; sin: Float32Array } {
  const hd = D1.headDim;
  const cos = new Float32Array(t * hd);
  const sin = new Float32Array(t * hd);
  for (let i = 0; i < hd / 2; i++) {
    const invFreq = Math.fround(1 / Math.pow(D1.ropeTheta, (2 * i) / hd));
    for (let p = 0; p < t; p++) {
      const a = Math.fround(p * invFreq);
      cos[p * hd + 2 * i] = cos[p * hd + 2 * i + 1] = Math.cos(a);
      sin[p * hd + 2 * i] = sin[p * hd + 2 * i + 1] = Math.sin(a);
    }
  }
  return { cos, sin };
}

export async function createD1Decider(
  sd: LazyStateDict,
  tokenizer: D1Tokenizer,
  opts: {
    temperatures?: Record<string, number>;
    onProgress?: (name: string, done: number, total: number) => void;
  } = {},
): Promise<D1Decider> {
  const model = new D1Model();
  await model.loadStateDict(sd, { onProgress: opts.onProgress });
  const ex = defaultExecutor();
  const temperatures = opts.temperatures ?? {};
  const zeroRow = tensor(new Float32Array(D1.hidden), [1, D1.hidden]);

  const posTable = model.vision.position_embedding.data;
  if (!(posTable instanceof Float32Array)) throw new Error('d1: position table must be f32');

  /** One image's crops through the tower and projector, rows concatenated. */
  const encodeImage = (img: RgbImage): { prefix: Value; uploads: Value[] } => {
    const uploads: Value[] = [];
    const parts = preprocessImage(img).map(({ ph, pw, patches }) => {
      const pos = resizeAA(posTable, D1.posGrid, D1.posGrid, D1.visionHidden, ph, pw);
      const p = tensor(patches, [ph * pw, 3 * 16 * 16]);
      const q = tensor(pos, [ph * pw, D1.visionHidden]);
      uploads.push(p, q);
      return model.vision.forward(p, q, ph, pw);
    });
    return { prefix: parts.length === 1 ? parts[0]! : cat(parts, 0), uploads };
  };

  const runQuestion = async (
    prefix: Value | undefined,
    ids: number[],
    q: Question,
    markers: number[],
  ) => {
    const p = prefix ? prefix.shape.dims![0]! : 0;
    const t = p + ids.length;
    const emb = model.encoder.embed_tokens.forward(ids);
    const { cos: c, sin: s } = ropeTables(t);
    const cos = tensor(c, [t, D1.headDim]);
    const sin = tensor(s, [t, D1.headDim]);
    const uploads = [emb, cos, sin];
    let segments: Value | undefined;
    let keepRight: Value | undefined;
    if (p > 0) {
      const seg = new Float32Array(2 * t);
      const keep = new Float32Array(t).fill(1);
      for (let i = 0; i < t; i++) seg[2 * i + 1] = i < p ? p : t;
      keep[p - 1] = 0;
      segments = tensor(seg, [t, 2]);
      keepRight = tensor(keep, [t, 1]);
      uploads.push(segments, keepRight);
    }
    const typeRow = model.head.type_emb.forward([QTYPES[q.type]]);
    uploads.push(typeRow);
    const h = model.encoder.forward(prefix ? cat([prefix, emb], 0) : emb, {
      cos,
      sin,
      segments,
      keepRight,
      zeroRow,
    });
    const text = p > 0 ? slice(h, 0, p, t) : h;
    const logits = await toArray(model.head.forward(text, typeRow, markers), ex);
    for (const u of uploads) u.buffer.destroy();
    return Array.from(logits.subarray(0, optionCount(q)));
  };

  const systemOne: D1Decider['systemOne'] = async (state, questions, images) => {
    let prefix: Value | undefined;
    let imageUploads: Value[] = [];
    if (images && images.length > 0) {
      const encoded = images.map(encodeImage);
      prefix =
        encoded.length === 1
          ? encoded[0]!.prefix
          : cat(
              encoded.map((e) => e.prefix),
              0,
            );
      imageUploads = encoded.flatMap((e) => e.uploads);
      evalValues([prefix], ex);
    }
    const p = prefix ? prefix.shape.dims![0]! : 0;
    const maxLen = Math.min(prefix ? IMAGE_TEXT_LENGTH : MAX_LENGTH, MAX_LENGTH - p);
    const answers: Record<string, Answer> = {};
    let inputTokens = 0;
    for (const [name, q] of Object.entries(questions)) {
      const { ids, markers } = encodeQuestion(
        tokenizer,
        state,
        q,
        maxLen,
        prefix ? YES_NO : undefined,
      );
      inputTokens += p + ids.length;
      const logits = await runQuestion(prefix, ids, q, markers);
      const temperature = prefix
        ? 1
        : (temperatures[temperatureKey(q)] ?? temperatures[q.type] ?? 1);
      const probs = softmaxT(logits, temperature);
      // The model reads a noul as [false, true]; answers want [yes, no].
      answers[name] = toAnswer(q, q.type === 'noul' ? probs.reverse() : probs);
    }
    if (prefix) ex.releaseValues([prefix]);
    for (const u of imageUploads) u.buffer.destroy();
    return { answers, inputTokens };
  };

  return {
    systemOne,
    dispose: () => {
      zeroRow.buffer.destroy();
      model.dispose();
    },
  };
}
