/** Reverb's sampler, on the CPU, over the per-slot stats each pass reads back.
 *
 *  Main passes fill codebooks 0–1. Each commits the best scored masked slots
 *  (confidence − codebook penalty + Gumbel noise), most in the last pass.
 *  Every pass but the last then re-masks a quarter as many of the least
 *  confident earlier commits.
 *
 *  Two polish passes fill the rest: codebooks 2–3 from the first, 4–7 from
 *  the second guided against the first. */

import { COARSE_CODEBOOKS, MASK } from './config.ts';

/** Warps the commit schedule: few commits early, most in the last passes. */
const T_SHIFT = 0.1;
/** Per codebook index, so codebook 0 tends to commit before 1. */
const CODEBOOK_PENALTY = 5;
/** Scales confidence and penalty against the Gumbel noise. */
const POSITION_TEMPERATURE = 5;
/** Earlier commits re-masked per new commit. */
const REMASK_RATIO = 0.25;
/** Codebooks 4–7: weight of (second polish pass − first). */
export const POLISH_GUIDANCE = 0.5;
/** Fine codebooks (2–7) the first polish pass fills: 2 and 3. */
const FIRST_POLISH_BOOKS = 2;
/** Attention bias towards the reference frames in the polish passes. */
export const POLISH_BIAS = 1;

/** A pass's best non-mask code per slot and its log-probability, both
 *  [books, frames]. */
export interface SlotStats {
  best: Int32Array;
  confidence: Float32Array;
}

/** Rounds halves to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** mulberry32: a seeded uniform in [0, 1). */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Commits per main pass for `frames` target frames over `steps` passes:
 *  ceil(2·frames·Δt) on the warped schedule, the last pass taking the rest.
 *  t is f32, as torch.linspace(0, 1, steps + 1) and the warp give it: each
 *  point is start + step·i from the near end, rounded once. */
function commitSchedule(steps: number, frames: number): number[] {
  const f = Math.fround;
  const n = steps + 1;
  const step = f(1 / steps);
  const t = Array.from({ length: n }, (_, i) => {
    const x = f(i < Math.floor(n / 2) ? step * i : 1 - step * (n - 1 - i));
    return f(f(f(T_SHIFT) * x) / f(1 - f(f(1 - T_SHIFT) * x)));
  });
  return Array.from({ length: steps }, (_, i) =>
    Math.ceil(COARSE_CODEBOOKS * frames * (t[i + 1]! - t[i]!)),
  );
}

/** Indices of the k largest eligible scores, largest first. */
function topIndices(
  score: Float64Array | Float32Array,
  k: number,
  eligible: (i: number) => boolean,
) {
  const idx: number[] = [];
  for (let i = 0; i < score.length; i++) if (eligible(i)) idx.push(i);
  idx.sort((x, y) => score[y]! - score[x]!);
  return idx.slice(0, k);
}

/** Runs main pass `step` of `steps` on tokens [codebooks, frames]: commits,
 *  then re-masks. Draws one noise value per coarse slot, masked or not, so a
 *  recorded run replays exactly. */
export function mainPassUpdate(
  tokens: Int32Array,
  frames: number,
  coarse: SlotStats,
  step: number,
  steps: number,
  random: () => number,
): void {
  const slots = COARSE_CODEBOOKS * frames;
  const last = step === steps - 1;
  let masked = 0;
  for (let i = 0; i < slots; i++) if (tokens[i] === MASK) masked++;
  const k = last ? masked : Math.min(masked, commitSchedule(steps, frames)[step]!);
  const score = new Float64Array(slots);
  for (let i = 0; i < slots; i++) {
    const book = Math.floor(i / frames);
    const u = random();
    const gumbel = -Math.log(-Math.log(u + 1e-10) + 1e-10);
    score[i] = (coarse.confidence[i]! - CODEBOOK_PENALTY * book) / POSITION_TEMPERATURE + gumbel;
  }
  const revealed = new Uint8Array(slots);
  for (let i = 0; i < slots; i++) revealed[i] = tokens[i] === MASK ? 0 : 1;
  for (const i of topIndices(score, k, (i) => !revealed[i])) tokens[i] = coarse.best[i]!;
  if (last) return;
  let revealedCount = 0;
  for (let i = 0; i < slots; i++) revealedCount += revealed[i]!;
  const count = Math.min(roundHalfEven(REMASK_RATIO * k), revealedCount);
  const negConfidence = Float32Array.from(coarse.confidence, (c) => -c);
  for (const i of topIndices(negConfidence, count, (i) => revealed[i] === 1)) tokens[i] = MASK;
}

/** Copies fine codebooks [from, to) of `best` [books, frames] into tokens. */
function fillFine(tokens: Int32Array, frames: number, best: Int32Array, from: number, to: number) {
  for (let b = from; b < to; b++) {
    for (let t = 0; t < frames; t++) {
      tokens[(COARSE_CODEBOOKS + b) * frames + t] = best[b * frames + t]!;
    }
  }
}

/** Fills codebooks 2–3 from the first polish pass. */
export function polishFirst(tokens: Int32Array, frames: number, fine: { best: Int32Array }): void {
  fillFine(tokens, frames, fine.best, 0, FIRST_POLISH_BOOKS);
}

/** Fills codebooks 4–7 from the guided second polish pass. */
export function polishSecond(tokens: Int32Array, frames: number, fine: { best: Int32Array }): void {
  fillFine(tokens, frames, fine.best, FIRST_POLISH_BOOKS, fine.best.length / frames);
}
