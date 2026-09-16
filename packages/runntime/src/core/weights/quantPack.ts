/** Pack a column slice [oStart,oEnd) of a raw quantized expert tensor into
 *  eager's blocked-unit quantW layout + the matching scale slice.
 *  Input qbytes: canonical [E, K, O] int stream (int8 = 1 signed byte; int4 =
 *  2/byte, even scalar index = low nibble). Input scales: f32 [E, KG, O],
 *  KG = K/groupSize. Output qdata: the layout kernels/quantCommon.ts documents —
 *  16-value tiles per (expert, 4-col unit, 4-row chunk), col-major within the
 *  tile, packed at `bits` stride (tile = bits/2 words), tiles k4-major per
 *  (e, unit): word base ((e·W/4 + n)·K/4 + k4)·(bits/2) for width W.
 *  Output scales: f32 [E, KG, width] (unchanged layout — a unit's 4 scales
 *  stay adjacent). */
export function packQuantColSlice(
  qbytes: Uint8Array,
  scales: Float32Array,
  bits: 8 | 4,
  groupSize: number,
  E: number,
  K: number,
  O: number,
  oStart: number,
  oEnd: number,
): { qdata: Uint32Array; scales: Float32Array } {
  const width = oEnd - oStart;
  // Tiles are 4 cols × 4 rows: both the slice width and K must tile evenly,
  // or the packing here would silently disagree with the kernels (the real
  // model's width=640/K=640 always do).
  if (width <= 0 || width % 4 !== 0) {
    throw new Error(`packQuantColSlice: slice width ${width} must be a positive multiple of 4`);
  }
  if (K % 4 !== 0)
    throw new Error(`packQuantColSlice: K ${K} must be a multiple of 4 (4-row tiles)`);
  if (K % groupSize !== 0)
    throw new Error(`packQuantColSlice: groupSize ${groupSize} must divide K ${K}`);
  const KG = K / groupSize;
  const expectQ = bits === 8 ? E * K * O : (E * K * O) / 2;
  if (qbytes.length !== expectQ)
    throw new Error(`packQuantColSlice: qbytes ${qbytes.length} != ${expectQ}`);
  if (scales.length !== E * KG * O)
    throw new Error(`packQuantColSlice: scales ${scales.length} != ${E * KG * O}`);
  const mask = bits === 8 ? 0xff : 0xf;
  const perU32 = bits === 8 ? 4 : 8;
  const qdata = new Uint32Array((E * K * width) / perU32);
  const outScales = new Float32Array(E * KG * width);
  const readQ = (s: number): number => {
    if (bits === 8) return (qbytes[s]! << 24) >> 24;
    const nib = s & 1 ? qbytes[s >> 1]! >> 4 : qbytes[s >> 1]! & 0xf;
    return nib >= 8 ? nib - 16 : nib;
  };
  const n4 = width / 4;
  const k4count = K / 4;
  for (let e = 0; e < E; e++) {
    for (let n = 0; n < n4; n++) {
      for (let k4 = 0; k4 < k4count; k4++) {
        const tileBit = ((e * n4 + n) * k4count + k4) * 16 * bits;
        for (let j = 0; j < 4; j++) {
          for (let kk = 0; kk < 4; kk++) {
            const v = readQ((e * K + k4 * 4 + kk) * O + (oStart + n * 4 + j));
            const bit = tileBit + (j * 4 + kk) * bits;
            qdata[bit >> 5] = (qdata[bit >> 5]! | ((v & mask) << (bit & 31))) >>> 0;
          }
        }
      }
    }
    for (let g = 0; g < KG; g++) {
      for (let oo = 0; oo < width; oo++) {
        outScales[(e * KG + g) * width + oo] = scales[(e * KG + g) * O + (oStart + oo)]!;
      }
    }
  }
  return { qdata, scales: outScales };
}
