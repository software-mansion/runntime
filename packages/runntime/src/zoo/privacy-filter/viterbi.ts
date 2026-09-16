import { type BoundaryTag, type LabelInfo, type ViterbiBiases, ZERO_BIASES } from './config.ts';

const NEG_INF = -1e9;

export function logSoftmaxRows(logits: Float32Array, width: number): Float32Array {
  const rows = logits.length / width;
  const out = new Float32Array(logits.length);
  for (let r = 0; r < rows; r++) {
    const base = r * width;
    let max = -Infinity;
    for (let c = 0; c < width; c++) max = Math.max(max, logits[base + c]!);
    let sum = 0;
    for (let c = 0; c < width; c++) sum += Math.exp(logits[base + c]! - max);
    const logSum = Math.log(sum) + max;
    for (let c = 0; c < width; c++) out[base + c] = logits[base + c]! - logSum;
  }
  return out;
}

export class ViterbiDecoder {
  readonly #start: Float64Array;
  readonly #end: Float64Array;
  readonly #trans: Float64Array; // [n × n], prev-major
  readonly #n: number;

  constructor(info: LabelInfo, biases: ViterbiBiases = ZERO_BIASES) {
    const n = info.numClasses;
    this.#n = n;
    this.#start = new Float64Array(n).fill(NEG_INF);
    this.#end = new Float64Array(n).fill(NEG_INF);
    this.#trans = new Float64Array(n * n).fill(NEG_INF);

    const bg = info.backgroundTokenLabel;
    const bgSpan = info.backgroundSpanLabel;
    const tag = (i: number) => info.tokenBoundaryTags.get(i) ?? null;
    const span = (i: number) => info.tokenToSpanLabel.get(i);
    const isBg = (i: number) => span(i) === bgSpan || i === bg;

    const valid = (prev: number, next: number): boolean => {
      const nextIsBg = isBg(next);
      const nt = tag(next);
      if ((span(next) === undefined || nt === null) && !nextIsBg) return false;
      const pt = tag(prev);
      if (span(prev) === undefined || pt === null) return nextIsBg || nt === 'B' || nt === 'S';
      if (isBg(prev)) return nextIsBg || nt === 'B' || nt === 'S';
      if (pt === 'E' || pt === 'S') return nextIsBg || nt === 'B' || nt === 'S';
      if (pt === 'B' || pt === 'I') {
        return span(prev) === span(next) && (nt === 'I' || nt === 'E');
      }
      return false;
    };

    const bias = (prev: number, next: number): number => {
      const pt = tag(prev) as BoundaryTag | null;
      const nt = tag(next) as BoundaryTag | null;
      const prevIsBg = isBg(prev);
      const nextIsBg = isBg(next);
      if (prevIsBg) {
        if (nextIsBg) return biases.backgroundStay;
        if (nt === 'B' || nt === 'S') return biases.backgroundToStart;
        return 0;
      }
      if (pt === 'B' || pt === 'I') {
        if (nt === 'I' && span(prev) === span(next)) return biases.insideToContinue;
        if (nt === 'E' && span(prev) === span(next)) return biases.insideToEnd;
        return 0;
      }
      if (pt === 'E' || pt === 'S') {
        if (nextIsBg) return biases.endToBackground;
        if (nt === 'B' || nt === 'S') return biases.endToStart;
        return 0;
      }
      return 0;
    };

    for (let i = 0; i < n; i++) {
      const t = tag(i);
      if (t === 'B' || t === 'S' || i === bg) this.#start[i] = 0;
      if (t === 'E' || t === 'S' || i === bg) this.#end[i] = 0;
      for (let j = 0; j < n; j++) {
        if (valid(i, j)) this.#trans[i * n + j] = bias(i, j);
      }
    }
  }

  decode(logProbs: Float32Array, width: number): number[] {
    const n = this.#n;
    if (width !== n) throw new Error(`Expected width ${n}, got ${width}`);
    const seqLen = logProbs.length / n;
    if (seqLen === 0) return [];

    let scores = new Float64Array(n);
    for (let j = 0; j < n; j++) scores[j] = logProbs[j]! + this.#start[j]!;
    const backpointers = new Int32Array((seqLen - 1) * n);

    let next = new Float64Array(n);
    for (let t = 1; t < seqLen; t++) {
      for (let j = 0; j < n; j++) {
        let best = -Infinity;
        let bestI = 0;
        for (let i = 0; i < n; i++) {
          const s = scores[i]! + this.#trans[i * n + j]!;
          if (s > best) {
            best = s;
            bestI = i;
          }
        }
        next[j] = best + logProbs[t * n + j]!;
        backpointers[(t - 1) * n + j] = bestI;
      }
      [scores, next] = [next, scores];
    }

    if (![...scores].some(Number.isFinite)) {
      // Mirror reference fallback: per-token argmax.
      const out: number[] = [];
      for (let t = 0; t < seqLen; t++) {
        let best = -Infinity;
        let bestJ = 0;
        for (let j = 0; j < n; j++) {
          if (logProbs[t * n + j]! > best) {
            best = logProbs[t * n + j]!;
            bestJ = j;
          }
        }
        out.push(bestJ);
      }
      return out;
    }

    let last = 0;
    let best = -Infinity;
    for (let j = 0; j < n; j++) {
      const s = scores[j]! + this.#end[j]!;
      if (s > best) {
        best = s;
        last = j;
      }
    }
    const path = new Array<number>(seqLen);
    path[seqLen - 1] = last;
    for (let t = seqLen - 2; t >= 0; t--) {
      last = backpointers[t * n + path[t + 1]!]!;
      path[t] = last;
    }
    return path;
  }
}
