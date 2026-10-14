/** The numbers side of the kernel A/B: calls per round, the verdict and the
 *  markdown table. Pure, so the Node unit tests cover it. */

/** GPU time one round should take. Chrome rounds timestamps to 65.536 µs on
 *  Apple GPUs, so a 10 ms round keeps that error under 1%. */
export const TARGET_ROUND_US = 10_000;

/** Below this relative difference a change counts as noise, however quiet
 *  the rounds were: GPU clocks drift by about this much between rounds. */
export const MIN_NOISE = 0.03;

/** Most K outputs alive at once, in bytes, so a round never exhausts memory. */
export const MAX_ROUND_BYTES = 256 * 2 ** 20;

/** Calls per round: enough for TARGET_ROUND_US, at least 2, at most 256, and
 *  few enough that K outputs fit in MAX_ROUND_BYTES. */
export function callsPerRound(oneCallUs: number, outBytes: number): number {
  const byTime = Math.round(TARGET_ROUND_US / Math.max(oneCallUs, 1));
  const byMemory = Math.floor(MAX_ROUND_BYTES / Math.max(outBytes, 1));
  return Math.max(2, Math.min(256, byTime, byMemory));
}

export function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** Interquartile range relative to the median: the middle half of the rounds,
 *  so one or two rounds hit by a clock dip do not widen it. */
export function spread(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const q1 = median(s.slice(0, s.length >> 1));
  const q3 = median(s.slice((s.length + 1) >> 1));
  return (q3 - q1) / median(s);
}

export type Verdict = 'faster' | 'slower' | 'same';

export interface Comparison {
  aUs: number;
  bUs: number;
  /** (B − A) / A: negative when B is faster. */
  delta: number;
  /** The |delta| a real change has to exceed. */
  noise: number;
  verdict: Verdict;
}

/** Compares per-call times from A's and B's rounds. A change counts only when
 *  it is larger than both sides' own round-to-round spread and MIN_NOISE. */
export function compare(a: readonly number[], b: readonly number[]): Comparison {
  const aUs = median(a);
  const bUs = median(b);
  const delta = (bUs - aUs) / aUs;
  const noise = Math.max(MIN_NOISE, spread(a), spread(b));
  const verdict = Math.abs(delta) <= noise ? 'same' : delta < 0 ? 'faster' : 'slower';
  return { aUs, bUs, delta, noise, verdict };
}

/** Largest |a − b| over two outputs. NaN on either side, or a length
 *  mismatch, is Infinity so it can never read as equal. */
export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (!(d <= max)) max = Number.isNaN(d) ? Infinity : d;
  }
  return max;
}

export interface CaseResult {
  kernel: string;
  case: string;
  /** Why the case has no comparison: skipped, or failed on a side. */
  note?: string;
  comparison?: Comparison;
  maxAbsDiff?: number;
  calls?: number;
}

const us = (x: number) => (x >= 100 ? x.toFixed(0) : x.toFixed(1));
const pct = (x: number) => `${x > 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;

/** The report as a markdown table, one row per case. */
export function formatTable(results: readonly CaseResult[]): string {
  const rows = results.map((r) => {
    if (!r.comparison) return `| ${r.kernel} | ${r.case} | | | | | ${r.note} | |`;
    const { aUs, bUs, delta, noise, verdict } = r.comparison;
    const diff = r.maxAbsDiff === 0 ? '0' : r.maxAbsDiff!.toExponential(1);
    return `| ${r.kernel} | ${r.case} | ${us(aUs)} | ${us(bUs)} | ${pct(delta)} | ±${(noise * 100).toFixed(1)}% | ${verdict} | ${diff} |`;
  });
  return [
    '| kernel | case | A (µs) | B (µs) | Δ | noise | verdict | max abs(A−B) |',
    '| --- | --- | --: | --: | --: | --: | --- | --: |',
    ...rows,
  ].join('\n');
}
