/** The numbers side of the kernel A/B: calls per round, the verdict and the
 *  markdown table. Pure, so the Node unit tests cover it. */

/** GPU time one round should take. Chrome rounds timestamps to 65.536 µs on
 *  Apple GPUs, so a 10 ms round keeps that error under 1%. */
export const TARGET_ROUND_US = 10_000;

/** Below this relative difference a change counts as noise */
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

/** Merges a flagged comparison with a second, independent measurement. The
 *  flag stands only if both runs agree on its direction; a lone flag is the
 *  noise that hits about one row in 60. */
export function confirm(first: Comparison, second: Comparison): Comparison {
  const verdict = first.verdict === second.verdict ? first.verdict : 'same';
  return {
    aUs: (first.aUs + second.aUs) / 2,
    bUs: (first.bUs + second.bUs) / 2,
    delta: (first.delta + second.delta) / 2,
    noise: Math.max(first.noise, second.noise),
    verdict,
  };
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

/** A time in µs: no decimals from 100 up, where the decimal is noise. */
const formatTime = (x: number) => (x >= 100 ? x.toFixed(0) : x.toFixed(1));
/** A fraction as a signed percent: 0.673 → '+67.3%'. */
const formatPercent = (x: number) => `${x > 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;

type Measured = CaseResult & { comparison: Comparison };

/** The report's columns, in order. A number column lines its values up on the
 *  last digit, the way a spreadsheet does; the others line up on the first
 *  letter. */
const COLUMNS: { name: string; isNumber?: boolean; cell: (r: Measured) => string }[] = [
  { name: 'kernel', cell: (r) => r.kernel },
  { name: 'case', cell: (r) => r.case },
  { name: 'A (µs)', isNumber: true, cell: (r) => formatTime(r.comparison.aUs) },
  { name: 'B (µs)', isNumber: true, cell: (r) => formatTime(r.comparison.bUs) },
  { name: 'Δ', isNumber: true, cell: (r) => formatPercent(r.comparison.delta) },
  { name: 'noise', isNumber: true, cell: (r) => `±${(r.comparison.noise * 100).toFixed(1)}%` },
  { name: 'verdict', cell: (r) => r.comparison.verdict },
  {
    name: 'max abs(A−B)',
    isNumber: true,
    cell: (r) => (r.maxAbsDiff === 0 ? '0' : r.maxAbsDiff!.toExponential(1)),
  },
];
const HEADER = COLUMNS.map((c) => c.name);
const VERDICT = HEADER.indexOf('verdict');

const cells = (r: Measured) => COLUMNS.map((c) => c.cell(r));

/** A skipped or failed case: kernel and case, the note under verdict. */
const noteCells = (r: CaseResult) =>
  HEADER.map((_, j) => (j === 0 ? r.kernel : j === 1 ? r.case : j === VERDICT ? r.note! : ''));

const markdownRow = (c: readonly string[]) => `|${c.map((s) => (s ? ` ${s} ` : ' ')).join('|')}|`;

/** The report as a markdown table for the pull request, one row per case. */
export function formatTable(results: readonly CaseResult[]): string {
  return [
    markdownRow(HEADER),
    markdownRow(COLUMNS.map((c) => (c.isNumber ? '--:' : '---'))),
    ...results.map((r) => markdownRow(r.comparison ? cells(r as Measured) : noteCells(r))),
  ].join('\n');
}

/** The report for the terminal: a boxed table of the measured cases, a rule
 *  between kernels, then one line per skipped or failed case. `paint` colors
 *  the verdict cell after padding, so escape codes never shift a column. */
export function formatTerminal(
  results: readonly CaseResult[],
  paint: (verdict: Verdict, text: string) => string = (_, text) => text,
): string {
  const measured = results.filter((r): r is Measured => r.comparison !== undefined);
  const rows = measured.map(cells);
  const widths = HEADER.map((h, j) => Math.max(h.length, ...rows.map((c) => c[j]!.length)));
  const rule = (l: string, m: string, r: string) =>
    l + widths.map((w) => '─'.repeat(w + 2)).join(m) + r;
  const line = (c: readonly string[], verdict?: Verdict) => {
    const padded = c.map((s, j) =>
      COLUMNS[j]!.isNumber ? s.padStart(widths[j]!) : s.padEnd(widths[j]!),
    );
    if (verdict) padded[VERDICT] = paint(verdict, padded[VERDICT]!);
    return `│ ${padded.join(' │ ')} │`;
  };

  const out = [rule('┌', '┬', '┐'), line(HEADER), rule('├', '┼', '┤')];
  measured.forEach((r, i) => {
    const prev = measured[i - 1];
    if (prev && prev.kernel !== r.kernel) out.push(rule('├', '┼', '┤'));
    out.push(line(rows[i]!, r.comparison.verdict));
  });
  out.push(rule('└', '┴', '┘'));
  const notes = results.filter((r) => !r.comparison);
  if (notes.length) out.push('', ...notes.map((r) => `${r.kernel} · ${r.case}: ${r.note}`));
  return out.join('\n');
}
