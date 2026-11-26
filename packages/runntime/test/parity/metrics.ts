/** The numbers a parity test compares: how far an output is from its
 *  reference, how a ranking differs from the reference ranking, and how
 *  texts and the spans found in them differ from the reference's. */

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

/** Differences from a reference ranking, split by how far apart the
 *  reference itself scores the items involved. */
export interface Differences {
  /** Differences where the reference scores are more than the margin apart. */
  readonly count: number;
  /** Differences inside the margin: the reference barely tells the items
   *  apart. */
  readonly nearTies: number;
}

/** Splits differences by their reference score gap. A NaN gap or margin
 *  counts as a difference. */
function split(gaps: readonly number[], margin: number): Differences {
  const nearTies = gaps.filter((gap) => gap <= margin).length;
  return { count: gaps.length - nearTies, nearTies };
}

/** The positions of two rankings of the same length that hold different
 *  items. `scores` are the reference scores by item: the gap of a position
 *  is between its two items. */
export function rankMismatches(
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  scores: ArrayLike<number>,
  margin: number,
): Differences {
  const gaps: number[] = [];
  for (let i = 0; i < want.length; i++) {
    if (got[i] !== want[i]) gaps.push(Math.abs(scores[want[i]!]! - scores[got[i]!]!));
  }
  return split(gaps, margin);
}

/** The items of `want` missing from `got`, in any order. `scores` are the
 *  reference scores by item: the gap of a missing item is to the best item
 *  `got` holds in its place. */
export function missing(
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  scores: ArrayLike<number>,
  margin: number,
): Differences {
  const have = new Set(Array.from(got));
  const wanted = new Set(Array.from(want));
  let best = -Infinity;
  for (const item of have) if (!wanted.has(item)) best = Math.max(best, scores[item]!);
  const gaps: number[] = [];
  for (const item of wanted) if (!have.has(item)) gaps.push(scores[item]! - best);
  return split(gaps, margin);
}

/** Vectors on the two sides of a ranking: each asking vector ranks the
 *  ranked ones by dot product. */
export interface RankingVectors {
  readonly asking: readonly ArrayLike<number>[];
  readonly ranked: readonly ArrayLike<number>[];
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
};

/** How the top-k rankings `got` gives differ from the reference's, summed
 *  over the asking vectors: the positions that hold another item, and the
 *  items missing from the top k. `ranking` is the reference top k of each
 *  asking vector, [asking · k]. The gaps come from the scores of `want`. */
export function rankingDifferences(
  got: RankingVectors,
  want: RankingVectors,
  ranking: ArrayLike<number>,
  k: number,
  margin: number,
): { positions: Differences; items: Differences } {
  const total = { positions: { count: 0, nearTies: 0 }, items: { count: 0, nearTies: 0 } };
  got.asking.forEach((asking, q) => {
    const gotTopK = topK(
      got.ranked.map((item) => dot(asking, item)),
      k,
    );
    const wantTopK = Array.from({ length: k }, (_, i) => ranking[q * k + i]!);
    const wantScores = want.ranked.map((item) => dot(want.asking[q]!, item));
    const positions = rankMismatches(gotTopK, wantTopK, wantScores, margin);
    const items = missing(gotTopK, wantTopK, wantScores, margin);
    total.positions.count += positions.count;
    total.positions.nearTies += positions.nearTies;
    total.items.count += items.count;
    total.items.nearTies += items.nearTies;
  });
  return total;
}

/** The words of `text` as WER counts them: lower case, with punctuation
 *  dropped and apostrophes kept. */
export function words(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}' ]+/gu, ' ')
    .split(' ')
    .filter((word) => word !== '');
}

/** The fewest insertions, deletions and substitutions that turn `a` into `b`. */
export function editDistance<T>(a: readonly T[], b: readonly T[]): number {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++) {
      next.push(
        Math.min(row[j]! + 1, next[j - 1]! + 1, row[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)),
      );
    }
    row = next;
  }
  return row[b.length]!;
}

/** Word error rate of `hypotheses` against `references`, text by text: the
 *  word edits over the reference words, both summed over the texts. When
 *  the references hold no word it is NaN, or Infinity if a hypothesis holds
 *  one. */
export function wer(hypotheses: readonly string[], references: readonly string[]): number {
  if (hypotheses.length !== references.length) {
    throw new Error(`wer: ${hypotheses.length} hypotheses for ${references.length} references`);
  }
  let edits = 0;
  let total = 0;
  references.forEach((reference, i) => {
    const want = words(reference);
    edits += editDistance(words(hypotheses[i]!), want);
    total += want.length;
  });
  return edits / total;
}

/** A labeled stretch of a text, from `start` up to `end`, both UTF-16
 *  string indices. */
export interface Span {
  readonly label: string;
  readonly start: number;
  readonly end: number;
}

/** F1 of the spans in `got` against those in `want`, text by text: a span
 *  matches one with the same label, start and end, and each span matches
 *  once. The matches and the spans of both sides are summed over the texts.
 *  NaN when neither side holds a span. */
export function spanF1(
  got: readonly (readonly Span[])[],
  want: readonly (readonly Span[])[],
): number {
  if (got.length !== want.length) {
    throw new Error(`spanF1: spans of ${got.length} texts for ${want.length} references`);
  }
  const key = ({ label, start, end }: Span) => `${start}:${end}:${label}`;
  let matches = 0;
  let total = 0;
  want.forEach((spans, i) => {
    const left = new Map<string, number>();
    for (const span of spans) left.set(key(span), (left.get(key(span)) ?? 0) + 1);
    for (const span of got[i]!) {
      const count = left.get(key(span)) ?? 0;
      if (count > 0) matches++;
      left.set(key(span), count - 1);
    }
    total += got[i]!.length + spans.length;
  });
  return (2 * matches) / total;
}
