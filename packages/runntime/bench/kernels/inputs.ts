/** Input builders shared by the kernel A/B cases. Random data comes from
 *  `Side.rand`, so A and B get the same numbers. */
import type { Side } from '../ab.ts';
import type { EagerDtype, Value, ValueMeta } from '../../src/core/index.ts';

type Float = 'f32' | 'f16';

export function mat(rows: number, cols: number, dtype: EagerDtype = 'f32'): ValueMeta {
  return { elems: rows * cols, dtype, dims: [rows, cols] };
}

export function chw(c: number, h: number, w: number, dtype: Float = 'f32'): ValueMeta {
  return { elems: c * h * w, dtype, dims: [c, h, w] };
}

/** A random [rows, cols] input. */
export function randMat(s: Side, rows: number, cols: number, dtype: Float = 'f32'): Value {
  return s.up(s.rand(rows * cols), mat(rows, cols, dtype));
}

/** A random f16 image already stored hwc4, padding lanes zero. */
export function randHwc4(s: Side, c: number, h: number, w: number): Value {
  const meta = s.lib.hwc4Meta(c, h, w);
  const c4 = meta.elems / (h * w);
  const vals = s.rand(h * w * c);
  const data = new Float32Array(meta.elems);
  for (let p = 0; p < h * w; p++) {
    for (let ch = 0; ch < c; ch++) data[p * c4 + ch] = vals[p * c + ch]!;
  }
  return s.up(data, meta);
}

/** A conv2d weight from random torch-layout [cOut, cIn/groups, k, k], packed
 *  by the side's own packer. */
export function convWeight(s: Side, cOut: number, cIn: number, k: number, groups = 1): Value {
  const dw = groups > 1;
  const bits = s.lib.f32ArrayToF16Bits(s.rand(cOut * (cIn / groups) * k * k)).data;
  const packed = dw
    ? s.lib.packDwHwc4F16(bits, cIn, k, k)
    : s.lib.packConvHwc4F16(bits, cOut, cIn, k, k);
  const meta: ValueMeta = {
    elems: cOut * (cIn / groups) * k * k,
    dtype: 'f16',
    dims: [cOut, (cIn / groups) * k * k],
    layout: 'hwc4',
  };
  return s.keep(s.lib.materialized(meta, s.lib.uploadF16(s.root, packed)));
}

/** A quantized [E·K, O] weight and its f32 scales [E·K/groupSize, O]. */
export function quantWeight(
  s: Side,
  e: number,
  k: number,
  o: number,
  bits: 8 | 4,
  groupSize: number,
): { w: Value; scales: Value } {
  const perByte = bits === 8 ? 1 : 2;
  const qbytes = new Uint8Array((e * k * o) / perByte).map((_, i) => (i * 37 + 11) & 0xff);
  const raw = s.rand((e * k * o) / groupSize).map((x) => Math.abs(x) * 0.01 + 1e-3);
  const packed = s.lib.packQuantColSlice(qbytes, raw, bits, groupSize, e, k, o, 0, o);
  const w = s.keep(
    s.lib.materialized(
      { elems: (e * k * o) / (32 / bits), dtype: 'quantW', dims: [e * k, o] },
      s.lib.uploadU32(s.root, packed.qdata),
    ),
  );
  const scales = s.up(packed.scales, mat((e * k) / groupSize, o));
  return { w, scales };
}

/** Integer ids in [0, n) as f32, the form gatherRowsFrom and matmulGather take. */
export function ids(count: number, n: number): Float32Array {
  return Float32Array.from({ length: count }, (_, i) => (i * 7919) % n);
}
