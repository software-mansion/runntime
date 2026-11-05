/** WGSL kernels of the decoder engine (Gemma 3, Qwen 3).
 *
 *  Gemma's residual stream overflows f16 (activations reach 1e5), so every
 *  activation, accumulator and the KV cache are f32. Weights stay in the
 *  checkpoint's bf16 and are read as packed u32 pairs — the bf16 → f32
 *  widening is one shift, exact, and needs no shader-f16. Linear weights keep
 *  the torch [out, in] layout, so a GEMV row is one contiguous read.
 *
 *  Every kernel binds a static uniform `p` (shapes, fixed per call site) and
 *  the shared step uniform `s` (token count M and first position pos0),
 *  so bind groups are built once at load and a step only rewrites `s`. */

const STEP = /* wgsl */ `
struct Step { M: u32, pos0: u32, k: u32, _b: u32 }
fn bf2(w: u32) -> vec2<f32> {
  return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xffff0000u));
}
`;

/** x[m, :] = bf16 row `ids[m]` of one embedding-table chunk × scale. Rows of
 *  other chunks are left alone; one dispatch per chunk covers every token. */
export const EMBED_WGSL = /* wgsl */ `
${STEP}
struct P { K: u32, rowBase: u32, rows: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> s: Step;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.y;
  if (m >= s.M) { return; }
  let id = ids[m];
  if (id < p.rowBase || id >= p.rowBase + p.rows) { return; }
  let hw = p.K / 2u;
  let i = wg.x * 256u + li;
  if (i >= hw) { return; }
  let v = bf2(table[(id - p.rowBase) * hw + i]) * p.scale;
  x[m * p.K + 2u * i] = v.x;
  x[m * p.K + 2u * i + 1u] = v.y;
}
`;

/** RMSNorm (Gemma's weight pre-folded as (1 + w)):
 *    mode 0: out[m] = norm(in[m]) · w
 *    mode 1: out[m] += norm(in[m]) · w        (post-norm + residual add)
 *  lastOnly reads row M-1 and writes row 0 (the final norm before the
 *  LM head). */
export const RMSNORM_WGSL = /* wgsl */ `
${STEP}
struct P { N: u32, mode: u32, eps: f32, lastOnly: u32 }
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> s: Step;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  var m = wg.x;
  var o = wg.x;
  if (p.lastOnly != 0u) { m = s.M - 1u; o = 0u; }
  if (m >= s.M) { return; }
  let base = m * p.N;
  var ss = 0.0;
  for (var i = li; i < p.N; i += 256u) { let v = inp[base + i]; ss += v * v; }
  red[li] = ss;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.N) + p.eps);
  let ob = o * p.N;
  for (var i = li; i < p.N; i += 256u) {
    let y = inp[base + i] * r * w[i];
    if (p.mode == 0u) { outp[ob + i] = y; } else { outp[ob + i] += y; }
  }
}
`;

/** Residual add fused with the RMSNorm after it (Qwen's pre-norm layers):
 *    x[m] += y[m];  out[m] = norm(x[m]) · w
 *  One workgroup per row; each lane re-reads only the x it wrote. */
export const ADD_RMSNORM_WGSL = /* wgsl */ `
${STEP}
struct P { N: u32, _a: u32, eps: f32, _b: u32 }
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> s: Step;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.x;
  if (m >= s.M) { return; }
  let base = m * p.N;
  var ss = 0.0;
  for (var i = li; i < p.N; i += 256u) {
    let v = x[base + i] + y[base + i];
    x[base + i] = v;
    ss += v * v;
  }
  red[li] = ss;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(p.N) + p.eps);
  for (var i = li; i < p.N; i += 256u) { outp[base + i] = x[base + i] * r * w[i]; }
}
`;

/** Shared binding layout of the two linear kernels:
 *  y[m, yOff + n] (row stride ldy) = x[m, :] · W[n, :], W bf16 [N, K]. */
const LINEAR_HEAD = /* wgsl */ `
${STEP}
struct P { N: u32, K: u32, ldy: u32, yOff: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> s: Step;
`;

/** M = 1: 8 output rows per workgroup, 32 lanes per row streaming the row's
 *  K/8 vec4<u32> words. Dispatch ceil(N / 8). */
export const GEMV_WGSL = /* wgsl */ `
${LINEAR_HEAD}
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let lane = li % 32u;
  let n = wg.x * 8u + li / 32u;
  let K8 = p.K / 8u;
  var acc = 0.0;
  if (n < p.N) {
    let wb = n * K8;
    for (var i = lane; i < K8; i += 32u) {
      let wv = w[wb + i];
      let a = x[2u * i];
      let b = x[2u * i + 1u];
      acc += dot(a, vec4<f32>(bf2(wv.x), bf2(wv.y))) + dot(b, vec4<f32>(bf2(wv.z), bf2(wv.w)));
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) {
    if (lane < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && n < p.N) { y[p.yOff + n] = red[li]; }
}
`;

/** M > 1: 64×64 output tiles, K in steps of 16 through workgroup memory,
 *  4×4 outputs per thread as vec4 outer products. Needs K % 16 == 0.
 *  Dispatch (ceil(N / 64), ceil(M / 64)). Same bindings as GEMV; w is read
 *  as vec2<u32> (4 bf16) here. */
export const GEMM_WGSL = /* wgsl */ `
${STEP}
struct P { N: u32, K: u32, ldy: u32, yOff: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> w: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> s: Step;
// Scalar tiles: per-component writes into shared vec4 slots race on some
// backends (a component store is a read-modify-write of the whole vector).
var<workgroup> xs: array<f32, 1024>; // [k (16)][m (64)]
var<workgroup> ws: array<f32, 1024>; // [k (16)][n (64)]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let n0 = wg.x * 64u;
  let m0 = wg.y * 64u;
  let K4 = p.K / 4u;
  let tm = t / 16u;
  let tn = t % 16u;
  let lr = t / 4u;   // loader row 0..63
  let lc = t % 4u;   // loader k-quad 0..3
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    var xv = vec4<f32>(0.0);
    if (m0 + lr < s.M) { xv = x[(m0 + lr) * K4 + k0 / 4u + lc]; }
    var wv = vec2<u32>(0u);
    if (n0 + lr < p.N) { wv = w[(n0 + lr) * K4 + k0 / 4u + lc]; }
    let wa = bf2(wv.x);
    let wb = bf2(wv.y);
    let kb = 4u * lc * 64u + lr;
    xs[kb] = xv.x;
    xs[kb + 64u] = xv.y;
    xs[kb + 128u] = xv.z;
    xs[kb + 192u] = xv.w;
    ws[kb] = wa.x;
    ws[kb + 64u] = wa.y;
    ws[kb + 128u] = wb.x;
    ws[kb + 192u] = wb.y;
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let xo = kk * 64u + 4u * tm;
      let wo = kk * 64u + 4u * tn;
      let a = vec4<f32>(xs[xo], xs[xo + 1u], xs[xo + 2u], xs[xo + 3u]);
      let b = vec4<f32>(ws[wo], ws[wo + 1u], ws[wo + 2u], ws[wo + 3u]);
      acc0 += a.x * b;
      acc1 += a.y * b;
      acc2 += a.z * b;
      acc3 += a.w * b;
    }
    workgroupBarrier();
  }
  let n = n0 + 4u * tn;
  let mb = m0 + 4u * tm;
  for (var i = 0u; i < 4u; i++) {
    let m = mb + i;
    if (m >= s.M) { break; }
    var v = acc0;
    if (i == 1u) { v = acc1; } else if (i == 2u) { v = acc2; } else if (i == 3u) { v = acc3; }
    let o = m * p.ldy + p.yOff + n;
    for (var j = 0u; j < 4u; j++) {
      if (n + j < p.N) { y[o + j] = v[j]; }
    }
  }
}
`;

/** Splits the fused qkv rows: q and k get their per-head RMSNorm and RoPE
 *  (rotate-half form, tables per position), q is scaled by
 *  query_pre_attn_scalar^-0.5 (head_dim^-0.5 for Qwen), k and v land in the cache ring at
 *  slot = pos % cap. One workgroup per (head slot, token); workgroup size =
 *  headDim. */
export const QKV_WGSL = (headDim: number) => /* wgsl */ `
${STEP}
struct P { H: u32, KVH: u32, cap: u32, eps: f32, qScale: f32, _a: u32, _b: u32, _c: u32 }
const D = ${headDim}u;
const HALF = ${headDim / 2}u;
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read> cosT: array<f32>;
@group(0) @binding(4) var<storage, read> sinT: array<f32>;
@group(0) @binding(5) var<storage, read_write> qout: array<f32>;
@group(0) @binding(6) var<storage, read_write> kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> vc: array<f32>;
@group(0) @binding(8) var<uniform> p: P;
@group(0) @binding(9) var<uniform> s: Step;
var<workgroup> red: array<f32, D>;
var<workgroup> sh: array<f32, D>;
@compute @workgroup_size(D)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let slot = wg.x;
  let m = wg.y;
  if (m >= s.M) { return; }
  let pos = s.pos0 + m;
  let W = p.H + 2u * p.KVH;
  let src = qkv[(m * W + slot) * D + d];
  let row = (pos % p.cap) * p.KVH * D;
  if (slot >= p.H + p.KVH) {
    vc[row + (slot - p.H - p.KVH) * D + d] = src;
    return;
  }
  red[d] = src * src;
  workgroupBarrier();
  for (var st = D / 2u; st > 0u; st >>= 1u) {
    if (d < st) { red[d] += red[d + st]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(red[0] / f32(D) + p.eps);
  var wv = kn[d];
  if (slot < p.H) { wv = qn[d]; }
  let xn = src * r * wv;
  sh[d] = xn;
  workgroupBarrier();
  let i = d % HALF;
  let c = cosT[pos * HALF + i];
  let sn = sinT[pos * HALF + i];
  var rot: f32;
  if (d < HALF) { rot = xn * c - sh[d + HALF] * sn; } else { rot = xn * c + sh[d - HALF] * sn; }
  if (slot < p.H) {
    qout[m * p.H * D + slot * D + d] = rot * p.qScale;
  } else {
    kc[row + (slot - p.H) * D + d] = rot;
  }
}
`;

/** Causal (optionally sliding-window) GQA attention over the cache ring, as
 *  split-K: the visible keys of each (q head, token) are cut into blocks of
 *  BK, Z workgroups each reduce a run of blocks to a partial
 *  (max, sum, acc[D]) with an online softmax, and ATTN_COMBINE merges the Z
 *  partials. Z = min(blocks, ZMAX) follows from the step uniform, so decode
 *  at long context still spreads over many workgroups.
 *
 *  Key j is visible to position pos when j <= pos and, with a window W,
 *  pos - j < W. */
const ATTN_COMMON = (headDim: number) => /* wgsl */ `
${STEP}
struct P { H: u32, KVH: u32, cap: u32, window: u32, softcap: f32, _a: u32, _b: u32, _c: u32 }
const D = ${headDim}u;
const D4 = ${headDim / 4}u;
const BK = 128u;
const ZMAX = ${ATTN_ZMAX}u;
fn keyRange(pos: u32) -> vec2<u32> {
  var js = 0u;
  if (p.window > 0u && pos + 1u > p.window) { js = pos + 1u - p.window; }
  return vec2<u32>(js, pos + 1u);
}
// Blocks per workgroup and workgroup count, uniform over the dispatch.
fn split() -> vec2<u32> {
  var span = s.pos0 + s.M;
  if (p.window > 0u) { span = min(span, p.window); }
  let nblk = (span + BK - 1u) / BK;
  // Prefill already has M·H workgroups; only decode-sized steps split K.
  var z = min(nblk, ZMAX);
  if (s.M > ${ATTN_SPLIT_MAX_M}u) { z = 1u; }
  return vec2<u32>((nblk + z - 1u) / z, z);
}
`;

export const ATTN_ZMAX = 32;
/** Steps with more tokens than this run attention unsplit. */
export const ATTN_SPLIT_MAX_M = 8;
export const ATTN_BK = 128;

export const ATTN_WGSL = (headDim: number) => /* wgsl */ `
${ATTN_COMMON(headDim)}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read_write> pml: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> pacc: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<uniform> s: Step;
var<workgroup> qs: array<vec4<f32>, D4>;
var<workgroup> pdot: array<f32, 256>;
var<workgroup> sc: array<f32, BK>;
var<workgroup> red: array<f32, BK>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let m = wg.y;
  let z = wg.z;
  if (m >= s.M) { return; }
  let pos = s.pos0 + m;
  let kr = keyRange(pos);
  let sp = split();
  let bStart = kr.x + z * sp.x * BK;
  let bEnd = min(kr.y, bStart + sp.x * BK);
  let idx = (m * p.H + h) * sp.y + z;
  if (bStart >= kr.y) {
    if (li == 0u) { pml[idx] = vec2<f32>(-1e30, 0.0); }
    if (li < D) { pacc[idx * D + li] = 0.0; }
    return;
  }
  let kvh = h / (p.H / p.KVH);
  let qb = m * p.H * D + h * D;
  if (li < D4) {
    qs[li] = vec4<f32>(q[qb + 4u * li], q[qb + 4u * li + 1u], q[qb + 4u * li + 2u], q[qb + 4u * li + 3u]);
  }
  workgroupBarrier();
  let rowW = p.KVH * D;
  let key = li / 2u;
  let hf = li % 2u;
  var mRun = -1e30;
  var lRun = 0.0;
  var acc = 0.0;
  for (var b = bStart; b < bEnd; b += BK) {
    // two lanes per key, half of the head dim each
    let j = b + key;
    var part = 0.0;
    if (j < bEnd) {
      let kb = ((j % p.cap) * rowW + kvh * D) / 4u + hf * (D4 / 2u);
      for (var i = 0u; i < D4 / 2u; i++) { part += dot(qs[hf * (D4 / 2u) + i], kc[kb + i]); }
    }
    pdot[li] = part;
    workgroupBarrier();
    if (li < BK) {
      var sv = -1e30;
      if (b + li < bEnd) {
        sv = pdot[2u * li] + pdot[2u * li + 1u];
        if (p.softcap > 0.0) { sv = tanh(sv / p.softcap) * p.softcap; }
      }
      sc[li] = sv;
      red[li] = sv;
    }
    workgroupBarrier();
    for (var st = BK / 2u; st > 0u; st >>= 1u) {
      if (li < st) { red[li] = max(red[li], red[li + st]); }
      workgroupBarrier();
    }
    let newM = max(mRun, red[0]);
    workgroupBarrier();
    if (li < BK) {
      var pv = 0.0;
      if (b + li < bEnd) { pv = exp(sc[li] - newM); }
      sc[li] = pv;
      red[li] = pv;
    }
    workgroupBarrier();
    for (var st = BK / 2u; st > 0u; st >>= 1u) {
      if (li < st) { red[li] += red[li + st]; }
      workgroupBarrier();
    }
    let corr = exp(mRun - newM);
    lRun = lRun * corr + red[0];
    let cnt = min(BK, bEnd - b);
    if (li < D) {
      var a = acc * corr;
      for (var jj = 0u; jj < cnt; jj++) {
        a += sc[jj] * vc[((b + jj) % p.cap) * rowW + kvh * D + li];
      }
      acc = a;
    }
    mRun = newM;
    workgroupBarrier();
  }
  if (li == 0u) { pml[idx] = vec2<f32>(mRun, lRun); }
  if (li < D) { pacc[idx * D + li] = acc; }
}
`;

/** Merges the Z split-K partials of each (q head, token). */
export const ATTN_COMBINE_WGSL = (headDim: number) => /* wgsl */ `
${ATTN_COMMON(headDim)}
@group(0) @binding(0) var<storage, read> pml: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> pacc: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> s: Step;
@compute @workgroup_size(D)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) d: u32) {
  let h = wg.x;
  let m = wg.y;
  if (m >= s.M) { return; }
  let nz = split().y;
  let base = (m * p.H + h) * nz;
  var gm = -1e30;
  for (var z = 0u; z < nz; z++) { gm = max(gm, pml[base + z].x); }
  var num = 0.0;
  var den = 0.0;
  for (var z = 0u; z < nz; z++) {
    let ml = pml[base + z];
    let w = exp(ml.x - gm);
    den += w * ml.y;
    num += w * pacc[(base + z) * D + d];
  }
  outp[m * p.H * D + h * D + d] = num / den;
}
`;

/** h[m, f] = act(gu[m, f]) · gu[m, F + f], act = gelu_tanh (GeGLU, act 0)
 *  or silu (SwiGLU, act 1). */
export const GATED_ACT_WGSL = /* wgsl */ `
${STEP}
struct P { F: u32, act: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> s: Step;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wg.y;
  let f = wg.x * 256u + li;
  if (m >= s.M || f >= p.F) { return; }
  let g = gu[m * 2u * p.F + f];
  let u = gu[m * 2u * p.F + p.F + f];
  if (p.act == 1u) {
    h[m * p.F + f] = g / (1.0 + exp(-g)) * u;
    return;
  }
  let inner = clamp(0.7978845608028654 * (g + 0.044715 * g * g * g), -20.0, 20.0);
  h[m * p.F + f] = 0.5 * g * (1.0 + tanh(inner)) * u;
}
`;

/** Greedy argmax over the logits, pass 1: 256 workgroups each reduce a
 *  contiguous slice to (max, index); ties keep the lower index. A partial is
 *  stored as u32 bits: an index bitcast to f32 is subnormal below 2^23, and
 *  WGSL lets an implementation flush those to zero. */
export const ARGMAX1_WGSL = /* wgsl */ `
${STEP}
struct P { V: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> s: Step;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let chunk = (p.V + 255u) / 256u;
  let start = wg.x * chunk;
  let end = min(p.V, start + chunk);
  var best = -3.0e38;
  var bidx = 0xffffffffu;
  for (var i = start + li; i < end; i += 256u) {
    let v = logits[i];
    if (v > best) { best = v; bidx = i; }
  }
  bv[li] = best;
  bi[li] = bidx;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) {
      let ov = bv[li + st];
      let oi = bi[li + st];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
  }
  if (li == 0u) { part[wg.x] = vec2<u32>(bitcast<u32>(bv[0]), bi[0]); }
}
`;

/** Argmax pass 2: one workgroup merges the 256 partials, writes the id as
 *  the next step's input token and into the burst output at s.k. */
export const ARGMAX2_WGSL = /* wgsl */ `
${STEP}
@group(0) @binding(0) var<storage, read> part: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> outIds: array<u32>;
@group(0) @binding(3) var<uniform> s: Step;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let pv = part[li];
  bv[li] = bitcast<f32>(pv.x);
  bi[li] = pv.y;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) {
      let ov = bv[li + st];
      let oi = bi[li + st];
      if (ov > bv[li] || (ov == bv[li] && oi < bi[li])) { bv[li] = ov; bi[li] = oi; }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    ids[0] = bi[0];
    outIds[s.k] = bi[0];
  }
}
`;
