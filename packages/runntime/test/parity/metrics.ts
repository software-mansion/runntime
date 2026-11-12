/** The numbers a parity test compares: how far an output is from its
 *  reference, and how a ranking differs from the reference ranking. */

/** Cosine similarity of two vectors of the same length. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    aa += a[i]! ** 2;
    bb += b[i]! ** 2;
  }
  return dot / Math.sqrt(aa * bb);
}

/** The largest `|got - want|` over two arrays of the same length. */
export function maxAbsError(got: ArrayLike<number>, want: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < want.length; i++) max = Math.max(max, Math.abs(got[i]! - want[i]!));
  return max;
}

export interface Summary {
  readonly mean: number;
  /** The smallest value that at least 99% of the values do not exceed. */
  readonly p99: number;
  readonly max: number;
}

/** Mean, 99th percentile and max of a non-empty list. */
export function summarize(values: readonly number[]): Summary {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    mean: sorted.reduce((sum, v) => sum + v, 0) / sorted.length,
    p99: sorted[Math.ceil(0.99 * sorted.length) - 1]!,
    max: sorted[sorted.length - 1]!,
  };
}

/** Indices of the `k` largest scores, best first. Equal scores keep their
 *  index order. */
export function topK(scores: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: scores.length }, (_, i) => i)
    .sort((a, b) => scores[b]! - scores[a]! || a - b)
    .slice(0, k);
}

/** How many positions of two rankings of the same length hold different
 *  items. */
export function rankMismatches(got: ArrayLike<number>, want: ArrayLike<number>): number {
  let count = 0;
  for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) count++;
  return count;
}

/** How many items of `want` are missing from `got`, in any order. */
export function missing(got: ArrayLike<number>, want: ArrayLike<number>): number {
  const have = new Set(Array.from(got));
  let count = 0;
  for (let i = 0; i < want.length; i++) if (!have.has(want[i]!)) count++;
  return count;
}
