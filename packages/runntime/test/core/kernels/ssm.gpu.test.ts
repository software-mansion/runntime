import { describe, expect, it } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import {
  SSM_DIRS,
  SSM_STATE,
  ssmScanMerge,
  ssmScanProject,
  ssmSelectiveScan,
} from '../../../src/core/graph/ops.ts';
import { tensor3d } from '../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, upload } from '../../gpu.ts';

const { root, ex } = await gpu();

interface Shape {
  c: number;
  h: number;
  w: number;
  rank: number;
}

/** The pixel that VMamba's cross-scan visits at step t in direction dir:
 *  row-major, column-major, and both reversed. */
function scanPixel(dir: number, t: number, h: number, w: number): number {
  const s = dir >= 2 ? h * w - 1 - t : t;
  return dir === 1 || dir === 3 ? (s % h) * w + Math.floor(s / h) : s;
}

/** A flat f32 upload of `data`. */
const flatOf = (data: Float32Array) => upload(root, data, { elems: data.length, dtype: 'f32' });

/** x_proj over every direction's pixel order, then dt_proj: rows [0, 4C) are
 *  the pre-softplus delta per (direction, channel), then 4 × 8 B rows and
 *  4 × 8 C rows, each [P] in scan order. In f64. */
function projectRef(x: Float32Array, xw: Float32Array, dw: Float32Array, s: Shape): Float64Array {
  const { c, h, w, rank } = s;
  const p = h * w;
  const rows = rank + 2 * SSM_STATE;
  const out = new Float64Array((SSM_DIRS * c + 2 * SSM_DIRS * SSM_STATE) * p);
  for (let dir = 0; dir < SSM_DIRS; dir++) {
    for (let t = 0; t < p; t++) {
      const pixel = scanPixel(dir, t, h, w);
      const proj = Array.from({ length: rows }, (_, r) => {
        let sum = 0;
        for (let ch = 0; ch < c; ch++) sum += xw[(dir * rows + r) * c + ch]! * x[ch * p + pixel]!;
        return sum;
      });
      for (let co = 0; co < c; co++) {
        let delta = 0;
        for (let r = 0; r < rank; r++) delta += dw[(dir * c + co) * rank + r]! * proj[r]!;
        out[(dir * c + co) * p + t] = delta;
      }
      for (let st = 0; st < SSM_STATE; st++) {
        out[(SSM_DIRS * c + dir * SSM_STATE + st) * p + t] = proj[rank + st]!;
        out[(SSM_DIRS * c + SSM_DIRS * SSM_STATE + dir * SSM_STATE + st) * p + t] =
          proj[rank + SSM_STATE + st]!;
      }
    }
  }
  return out;
}

/** The selective-scan recurrence per (direction, channel), in f64:
 *  Δ = softplus(delta + bias), h ← exp(Δ·A)·h + Δ·B·u, y = C·h + D·u. */
function scanRef(
  x: Float32Array,
  proj: ArrayLike<number>,
  a: Float32Array,
  dSkip: Float32Array,
  deltaBias: Float32Array,
  { c, h, w }: Shape,
): Float64Array {
  const p = h * w;
  const out = new Float64Array(SSM_DIRS * c * p);
  for (let dir = 0; dir < SSM_DIRS; dir++) {
    for (let ch = 0; ch < c; ch++) {
      const seq = dir * c + ch;
      const state = new Float64Array(SSM_STATE);
      for (let t = 0; t < p; t++) {
        const u = x[ch * p + scanPixel(dir, t, h, w)]!;
        const delta = Math.log1p(Math.exp(proj[seq * p + t]! + deltaBias[seq]!));
        let y = 0;
        for (let st = 0; st < SSM_STATE; st++) {
          const b = proj[(SSM_DIRS * c + dir * SSM_STATE + st) * p + t]!;
          const cv = proj[(SSM_DIRS * c + SSM_DIRS * SSM_STATE + dir * SSM_STATE + st) * p + t]!;
          state[st] = Math.exp(delta * a[seq * SSM_STATE + st]!) * state[st]! + delta * b * u;
          y += cv * state[st]!;
        }
        out[seq * p + t] = y + dSkip[seq]! * u;
      }
    }
  }
  return out;
}

/** Sums each pixel's four directional outputs, read at that pixel's step in
 *  each scan order. In f64. */
function mergeRef(directional: ArrayLike<number>, { c, h, w }: Shape): Float64Array {
  const p = h * w;
  const out = new Float64Array(c * p);
  for (let dir = 0; dir < SSM_DIRS; dir++) {
    for (let t = 0; t < p; t++) {
      const pixel = scanPixel(dir, t, h, w);
      for (let ch = 0; ch < c; ch++) {
        out[ch * p + pixel] = out[ch * p + pixel]! + directional[(dir * c + ch) * p + t]!;
      }
    }
  }
  return out;
}

/** Rows of the projection the scan reads: delta, B and C. */
const scanRows = (c: number) => SSM_DIRS * c + 2 * SSM_DIRS * SSM_STATE;

function scanInputs(s: Shape) {
  const { c } = s;
  return {
    // A = −exp(A_log) in the model, so negative: the state decays.
    a: randomF32(SSM_DIRS * c * SSM_STATE, 5).map((v) => -Math.abs(v) - 0.1),
    dSkip: randomF32(SSM_DIRS * c, 6),
    deltaBias: randomF32(SSM_DIRS * c, 7),
  };
}

const TOLERANCE = { atol: 1e-5, rtol: 1e-4 };

describe('ssm kernels', () => {
  // One route each, a thread per output. Non-square maps catch an H/W swap in
  // the column-major orders.
  const SHAPES: { name: string; s: Shape }[] = [
    { name: '1×1 map, one channel', s: { c: 1, h: 1, w: 1, rank: 1 } },
    { name: 'a wide map', s: { c: 3, h: 4, w: 5, rank: 2 } },
    { name: 'a tall map', s: { c: 8, h: 7, w: 3, rank: 6 } },
    { name: 'past a workgroup of pixels', s: { c: 2, h: 9, w: 11, rank: 3 } },
  ];

  it.for(SHAPES)('ssmScanProject, $name', async ({ s }) => {
    const p = s.h * s.w;
    const x = randomF32(s.c * p, 1);
    const xw = randomF32(SSM_DIRS * (s.rank + 2 * SSM_STATE) * s.c, 2);
    const dw = randomF32(SSM_DIRS * s.c * s.rank, 3);
    const out = ssmScanProject(
      upload(root, x, tensor3d(s.c, s.h, s.w)),
      upload(root, xw, { elems: xw.length, dtype: 'f32' }),
      upload(root, dw, { elems: dw.length, dtype: 'f32' }),
      { rank: s.rank },
    );
    const got = await toArray(out, ex);
    // Rows past the C rows are dt scratch that no consumer reads.
    expectClose(got.subarray(0, scanRows(s.c) * p), projectRef(x, xw, dw, s), TOLERANCE);
  });

  it.for(SHAPES)('ssmSelectiveScan, $name', async ({ s }) => {
    const p = s.h * s.w;
    const x = randomF32(s.c * p, 1);
    const proj = randomF32(scanRows(s.c) * p, 2);
    const { a, dSkip, deltaBias } = scanInputs(s);
    const got = await toArray(
      ssmSelectiveScan(
        upload(root, x, tensor3d(s.c, s.h, s.w)),
        upload(root, proj, [scanRows(s.c), p]),
        upload(root, a, { elems: a.length, dtype: 'f32' }),
        upload(root, dSkip, { elems: dSkip.length, dtype: 'f32' }),
        upload(root, deltaBias, { elems: deltaBias.length, dtype: 'f32' }),
      ),
      ex,
    );
    expectClose(got, scanRef(x, proj, a, dSkip, deltaBias, s), TOLERANCE);
  });

  it.for(SHAPES)('ssmScanMerge, $name', async ({ s }) => {
    const p = s.h * s.w;
    const directional = randomF32(SSM_DIRS * s.c * p, 1);
    const got = await toArray(
      ssmScanMerge(upload(root, directional, [SSM_DIRS * s.c, p]), { c: s.c, h: s.h, w: s.w }),
      ex,
    );
    expectClose(got, mergeRef(directional, s), TOLERANCE);
  });

  // softplus switches form past ±20; the kernel takes exp(v) below −20 and v
  // above 20, both within f32 of log(1 + e^v). At 100, e^v overflows f32, so
  // only the v branch gives a finite result. Below −20 the two forms differ by
  // less than any tolerance, so that case checks the result, not the branch.
  it.each([
    { name: 'far below 0', bias: -30 },
    { name: 'far above 0', bias: 100 },
  ])('ssmSelectiveScan with a delta $name', async ({ bias }) => {
    const s = { c: 2, h: 3, w: 3, rank: 1 };
    const p = 9;
    const x = randomF32(s.c * p, 1);
    const proj = randomF32(scanRows(s.c) * p, 2);
    const { a, dSkip } = scanInputs(s);
    const deltaBias = new Float32Array(SSM_DIRS * s.c).fill(bias);
    const got = await toArray(
      ssmSelectiveScan(
        upload(root, x, tensor3d(s.c, s.h, s.w)),
        upload(root, proj, [scanRows(s.c), p]),
        upload(root, a, { elems: a.length, dtype: 'f32' }),
        upload(root, dSkip, { elems: dSkip.length, dtype: 'f32' }),
        upload(root, deltaBias, { elems: deltaBias.length, dtype: 'f32' }),
      ),
      ex,
    );
    expectClose(got, scanRef(x, proj, a, dSkip, deltaBias, s), { atol: 1e-4, rtol: 1e-4 });
  });

  it('project, scan and merge chained', async () => {
    const s = { c: 4, h: 5, w: 6, rank: 2 };
    const p = s.h * s.w;
    const x = randomF32(s.c * p, 1);
    const xw = randomF32(SSM_DIRS * (s.rank + 2 * SSM_STATE) * s.c, 2).map((v) => 0.5 * v);
    const dw = randomF32(SSM_DIRS * s.c * s.rank, 3);
    const { a, dSkip, deltaBias } = scanInputs(s);
    const xv = upload(root, x, tensor3d(s.c, s.h, s.w));
    const proj = ssmScanProject(xv, flatOf(xw), flatOf(dw), { rank: s.rank });
    const scanned = ssmSelectiveScan(xv, proj, flatOf(a), flatOf(dSkip), flatOf(deltaBias));
    const got = await toArray(ssmScanMerge(scanned, s), ex);
    const want = mergeRef(scanRef(x, projectRef(x, xw, dw, s), a, dSkip, deltaBias, s), s);
    expectClose(got, want, { atol: 1e-4, rtol: 1e-4 });
  });

  // Past GRID_Y_STRIDE threads the executor splits the grid in two.
  it('ssmScanProject past GRID_Y_STRIDE threads', async () => {
    const s = { c: 1, h: 256, w: 260, rank: 1 };
    const p = s.h * s.w;
    expect(SSM_DIRS * (s.rank + 2 * SSM_STATE) * p).toBeGreaterThan(GRID_Y_STRIDE);
    const x = randomF32(p, 1);
    const xw = randomF32(SSM_DIRS * (s.rank + 2 * SSM_STATE), 2);
    const dw = randomF32(SSM_DIRS, 3);
    const got = await toArray(
      ssmScanProject(upload(root, x, tensor3d(1, s.h, s.w)), flatOf(xw), flatOf(dw), s),
      ex,
    );
    expectClose(got.subarray(0, scanRows(1) * p), projectRef(x, xw, dw, s), TOLERANCE);
  });

  it('ssmScanMerge past GRID_Y_STRIDE threads', async () => {
    const s = { c: 16, h: 512, w: 520, rank: 1 };
    const p = s.h * s.w;
    expect(s.c * p).toBeGreaterThan(GRID_Y_STRIDE);
    const directional = Float32Array.from({ length: SSM_DIRS * s.c * p }, (_, i) => i % 7);
    const got = await toArray(ssmScanMerge(upload(root, directional, [SSM_DIRS * s.c, p]), s), ex);
    expectClose(got, mergeRef(directional, s), { atol: 0, rtol: 0 });
  });

  // Known bug: the kernels split the flat thread index with d.u32(thread / P),
  // which TypeGPU compiles to u32(f32(thread) / f32(P)). f32 counts exactly
  // only up to 2^24, so past 2^24 delta threads (4·C·P) some land on the
  // wrong channel. Turn into a plain `it` with the fix.
  it.fails('ssmScanProject past 2^24 delta threads', async () => {
    const s = { c: 64, h: 200, w: 330, rank: 1 };
    const p = s.h * s.w;
    expect(SSM_DIRS * s.c * p).toBeGreaterThan(2 ** 24);
    const x = randomF32(s.c * p, 1);
    const xw = randomF32(SSM_DIRS * (s.rank + 2 * SSM_STATE) * s.c, 2);
    const dw = randomF32(SSM_DIRS * s.c * s.rank, 3);
    const got = await toArray(
      ssmScanProject(upload(root, x, tensor3d(s.c, s.h, s.w)), flatOf(xw), flatOf(dw), s),
      ex,
    );
    expectClose(
      got.subarray(0, SSM_DIRS * s.c * p),
      projectRef(x, xw, dw, s).subarray(0, SSM_DIRS * s.c * p),
      {
        atol: 1e-4,
        rtol: 1e-4,
      },
    );
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), tensor3d(0, 2, 2))).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('ssmScanProject refuses weights of the wrong size', () => {
    const x = upload(root, new Float32Array(2 * 4), tensor3d(2, 2, 2));
    const dw = flatOf(new Float32Array(SSM_DIRS * 2));
    expect(() => ssmScanProject(x, flatOf(new Float32Array(10)), dw, { rank: 1 })).toThrow(
      /xw must be f32 with 136 elems, got 10/,
    );
    expect(() =>
      ssmScanProject(x, flatOf(new Float32Array(136)), flatOf(new Float32Array(3)), { rank: 1 }),
    ).toThrow(/dw must be f32 with 8 elems, got 3/);
    expect(() => ssmScanProject(x, flatOf(new Float32Array(136)), dw, { rank: 0 })).toThrow(
      /rank must be a positive integer, got 0/,
    );
  });

  it('ssmSelectiveScan refuses a projection too short for the scan', () => {
    const x = upload(root, new Float32Array(2 * 4), tensor3d(2, 2, 2));
    expect(() =>
      ssmSelectiveScan(
        x,
        flatOf(new Float32Array(10)),
        flatOf(new Float32Array(64)),
        flatOf(new Float32Array(8)),
        flatOf(new Float32Array(8)),
      ),
    ).toThrow(/proj must be a scan-project output/);
  });

  it('ssmScanMerge refuses another shape', () => {
    const directional = upload(root, new Float32Array(16), [4, 4]);
    expect(() => ssmScanMerge(directional, { c: 2, h: 2, w: 2 })).toThrow(
      /input must be f32 \[4·2, 4\]/,
    );
  });
});
