/** A torch [cOut, cIn, k] conv1d weight as tap-major [k·cIn, cOut]. A pure
 *  permutation, so f16 halfwords move as they are. */
export function convToTapMajor(
  src: Float32Array | Uint16Array,
  cOut: number,
  cIn: number,
  k: number,
): Float32Array | Uint16Array {
  if (src.length !== cOut * cIn * k) {
    throw new Error(`convToTapMajor: length ${src.length} != ${cOut}×${cIn}×${k}`);
  }
  const out =
    src instanceof Uint16Array ? new Uint16Array(src.length) : new Float32Array(src.length);
  for (let cout = 0; cout < cOut; cout++) {
    for (let cin = 0; cin < cIn; cin++) {
      for (let tap = 0; tap < k; tap++) {
        out[(tap * cIn + cin) * cOut + cout] = src[cout * cIn * k + cin * k + tap]!;
      }
    }
  }
  return out;
}

export function bf16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < src.length; i++) u32[i] = src[i]! << 16;
  return out;
}

export function f16BitsToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24; // subnormal
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

export function f16ToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f16BitsToF32(src[i]!);
  return out;
}

const f16Scratch = new DataView(new ArrayBuffer(4));

export function f32ToF16Bits(value: number): number {
  f16Scratch.setFloat32(0, value, true);
  const x = f16Scratch.getUint32(0, true);
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); // NaN / Inf
  let e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7bff; // clamp overflow to f16 max
  if (e <= 0) {
    if (e < -10) return sign; // underflows to zero
    mant |= 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    const rounded = (mant + half - 1 + ((mant >>> shift) & 1)) >>> shift;
    return sign | rounded;
  }
  const half = 0x1000;
  mant = mant + half - 1 + ((mant >>> 13) & 1);
  if (mant & 0x800000) {
    mant = 0;
    e += 1;
    if (e >= 31) return sign | 0x7bff;
  }
  return sign | (e << 10) | (mant >>> 13);
}

export function f32ArrayToF16Bits(src: Float32Array): { data: Uint16Array; clamped: number } {
  const data = new Uint16Array(src.length);
  let clamped = 0;
  for (let i = 0; i < src.length; i++) {
    const value = src[i]!;
    const bits = f32ToF16Bits(value);
    data[i] = bits;
    if ((bits & 0x7fff) === 0x7bff && Math.abs(value) > 65504) clamped++;
  }
  return { data, clamped };
}

export function bf16ToF16Bits(src: Uint16Array): { data: Uint16Array; clamped: number } {
  const data = new Uint16Array(src.length);
  let clamped = 0;
  for (let i = 0; i < src.length; i++) {
    f16Scratch.setUint32(0, src[i]! << 16, true);
    const value = f16Scratch.getFloat32(0, true);
    const bits = f32ToF16Bits(value);
    data[i] = bits;
    if ((bits & 0x7fff) === 0x7bff && Math.abs(value) > 65504) clamped++;
  }
  return { data, clamped };
}

export function permuteConvTransposeF16(
  src: Uint16Array,
  cIn: number,
  cOut: number,
  k: number,
): Uint16Array {
  const kk = k * k;
  const dst = new Uint16Array(cIn * cOut * kk);
  for (let ci = 0; ci < cIn; ci++) {
    const srcBase = ci * cOut * kk;
    for (let co = 0; co < cOut; co++) {
      for (let t = 0; t < kk; t++) {
        dst[(co * kk + t) * cIn + ci] = src[srcBase + co * kk + t]!;
      }
    }
  }
  return dst;
}

export function packConvHwc4F16(
  src: Uint16Array,
  cOut: number,
  cIn: number,
  kH: number,
  kW: number,
): Uint16Array {
  const kk = kH * kW;
  const ob = Math.ceil(cOut / 4);
  const ib = Math.ceil(cIn / 4);
  const dst = new Uint16Array(ob * ib * kk * 16);
  for (let o = 0; o < cOut; o++) {
    for (let i = 0; i < cIn; i++) {
      for (let t = 0; t < kk; t++) {
        const tile = ((o >> 2) * ib + (i >> 2)) * kk + t;
        dst[tile * 16 + (i & 3) * 4 + (o & 3)] = src[(o * cIn + i) * kk + t]!;
      }
    }
  }
  return dst;
}

/** A torch [cOut, cIn, 1, 1] f16 weight for the stride-1 1×1 hwc4 conv: the
 *  same mat4 tiles as packConvHwc4F16, ordered input block first, tile
 *  (iB, oB) at iB·CO4 + oB, so neighbouring output blocks read neighbouring
 *  tiles. */
export function packConv1x1Hwc4F16(src: Uint16Array, cOut: number, cIn: number): Uint16Array {
  const ob = Math.ceil(cOut / 4);
  const ib = Math.ceil(cIn / 4);
  const dst = new Uint16Array(ob * ib * 16);
  for (let o = 0; o < cOut; o++) {
    for (let i = 0; i < cIn; i++) {
      dst[((i >> 2) * ob + (o >> 2)) * 16 + (i & 3) * 4 + (o & 3)] = src[o * cIn + i]!;
    }
  }
  return dst;
}

/** Rows of G for Winograd F(4×4, 3×3). */
const WINOGRAD_G = [
  [1 / 4, 0, 0],
  [-1 / 6, -1 / 6, -1 / 6],
  [-1 / 6, 1 / 6, -1 / 6],
  [1 / 24, 1 / 12, 1 / 6],
  [1 / 24, -1 / 12, 1 / 6],
  [0, 0, 1],
] as const;

/** A torch [cOut, cIn, 3, 3] f16 conv weight as the 36 Winograd planes
 *  G·g·Gᵀ, in the hwc4 conv's mat4 tiles: tile (k, iB, oB) at
 *  (k·CI4 + iB)·CO4 + oB, entry iL holding the output lanes for input lane iL. */
export function packConvWinogradF16(src: Uint16Array, cOut: number, cIn: number): Uint16Array {
  const ob = Math.ceil(cOut / 4);
  const ib = Math.ceil(cIn / 4);
  const dst = new Uint16Array(36 * ib * ob * 16);
  const gg = new Float64Array(18);
  for (let o = 0; o < cOut; o++) {
    for (let i = 0; i < cIn; i++) {
      const base = (o * cIn + i) * 9;
      // G·g, 6×3.
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let t = 0; t < 3; t++)
            s += WINOGRAD_G[r]![t]! * f16BitsToF32(src[base + t * 3 + c]!);
          gg[r * 3 + c] = s;
        }
      }
      // (G·g)·Gᵀ, 6×6.
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let s = 0;
          for (let t = 0; t < 3; t++) s += gg[r * 3 + t]! * WINOGRAD_G[c]![t]!;
          const tile = ((r * 6 + c) * ib + (i >> 2)) * ob + (o >> 2);
          dst[tile * 16 + (i & 3) * 4 + (o & 3)] = f32ToF16Bits(s);
        }
      }
    }
  }
  return dst;
}

export function packDwHwc4F16(src: Uint16Array, c: number, kH: number, kW: number): Uint16Array {
  const kk = kH * kW;
  const dst = new Uint16Array(Math.ceil(c / 4) * kk * 4);
  for (let ch = 0; ch < c; ch++) {
    for (let t = 0; t < kk; t++) {
      dst[((ch >> 2) * kk + t) * 4 + (ch & 3)] = src[ch * kk + t]!;
    }
  }
  return dst;
}
