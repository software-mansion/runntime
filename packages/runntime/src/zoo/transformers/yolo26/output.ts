import type { RawLevel } from '../../yolo26/pipeline.ts';

/** `a` ranks above `b`: higher score, index breaks ties. */
const beats = (score: Float32Array, a: number, b: number) =>
  score[a]! > score[b]! || (score[a] === score[b] && a < b);

export function topIndices(score: Float32Array, k: number): number[] {
  const n = score.length;
  const size = Math.min(k, n);
  const heap = new Int32Array(size);
  let len = 0;
  const siftDown = (start: number) => {
    let i = start;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let worst = i;
      if (l < len && beats(score, heap[worst]!, heap[l]!)) worst = l;
      if (r < len && beats(score, heap[worst]!, heap[r]!)) worst = r;
      if (worst === i) return;
      const t = heap[i]!;
      heap[i] = heap[worst]!;
      heap[worst] = t;
      i = worst;
    }
  };
  for (let i = 0; i < n; i++) {
    if (len < size) {
      heap[len] = i;
      let j = len++;
      while (j > 0) {
        const parent = (j - 1) >> 1;
        if (!beats(score, heap[parent]!, heap[j]!)) break;
        const t = heap[j]!;
        heap[j] = heap[parent]!;
        heap[parent] = t;
        j = parent;
      }
    } else if (beats(score, i, heap[0]!)) {
      heap[0] = i;
      siftDown(0);
    }
  }
  return Array.from(heap.subarray(0, len)).sort((a, b) => (beats(score, a, b) ? -1 : 1));
}

export interface TopAnchors {
  /** `count` x (numClasses + 1): the class logits, then the background 0. */
  logits: Float32Array;
  boxes: Float32Array;
  count: number;
  /** The class count the logits actually carry, background included. */
  outClasses: number;
}

export function selectTopAnchors(
  levels: readonly RawLevel[],
  opts: { numClasses: number; maxDet: number; inputSize: number },
): TopAnchors {
  const { numClasses, maxDet, inputSize } = opts;
  const total = levels.reduce((n, l) => n + l.h * l.w, 0);
  // Best class logit per anchor; the class loop is outermost so each pass
  // reads one channel plane sequentially.
  const best = new Float32Array(total).fill(-Infinity);
  let base = 0;
  for (const { data, h, w } of levels) {
    const hw = h * w;
    for (let c = 0; c < numClasses; c++) {
      const plane = (4 + c) * hw;
      for (let i = 0; i < hw; i++) {
        const v = data[plane + i]!;
        if (v > best[base + i]!) best[base + i] = v;
      }
    }
    base += hw;
  }
  const order = topIndices(best, maxDet);
  const count = Math.min(maxDet, total);
  // A trailing background class at 0: post_process_object_detection softmaxes
  // across classes and drops anchors whose best class is last, giving sigmoid.
  const outClasses = numClasses + 1;
  const logits = new Float32Array(count * outClasses);
  const boxes = new Float32Array(count * 4);
  for (let k = 0; k < count; k++) {
    let idx = order[k]!;
    let level = 0;
    while (idx >= levels[level]!.h * levels[level]!.w) {
      idx -= levels[level]!.h * levels[level]!.w;
      level++;
    }
    const { data, h, w, stride } = levels[level]!;
    const hw = h * w;
    for (let c = 0; c < numClasses; c++) logits[k * outClasses + c] = data[(4 + c) * hw + idx]!;
    const ax = (idx % w) + 0.5;
    const ay = Math.floor(idx / w) + 0.5;
    const l = data[idx]!;
    const t = data[hw + idx]!;
    const r = data[2 * hw + idx]!;
    const b = data[3 * hw + idx]!;
    const scale = stride / inputSize;
    boxes[k * 4] = (ax + (r - l) / 2) * scale;
    boxes[k * 4 + 1] = (ay + (b - t) / 2) * scale;
    boxes[k * 4 + 2] = (l + r) * scale;
    boxes[k * 4 + 3] = (t + b) * scale;
  }
  return { logits, boxes, count, outClasses };
}
