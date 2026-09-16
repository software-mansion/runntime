/** The privacy filter's attention block, composed from core primitives:
 *
 *      x + out( sdpa( rope(qkv(norm(x))) ) )
 *
 *  qkv's row is q(14×64), k(2×64) and v(2×64) side by side. The q and k slabs
 *  are roped in one dispatch each, covering all head sections at once, and sdpa
 *  runs the banded GQA-with-sinks attention fused, with no per-head fan-out and
 *  no [T,T] scratch.
 *
 *  The caller passes the cos and sin rope tables into forward(), computed once
 *  per forward pass and uploaded as materialized Values.
 *
 *  This block folds the residual add in, which is why the composing block in
 *  model.ts adds nothing of its own. */

import { add, nn, sdpa, slice, type Value } from '../../core/index.ts';
import { applyRope } from './rope.ts';

const HIDDEN = 640;
const QKV_WIDTH = 1152; // q(14×64) ‖ k(2×64) ‖ v(2×64)
const ATTN_OUT = 896; // 14 heads × 64 concatenated back
const Q_HEADS = 14;
const KV_HEADS = 2;
export const HEAD = 64;
const WINDOW = 128; // bidirectional: attend within |i−j| ≤ 128
const K_OFF = Q_HEADS * HEAD; // 896 — k sections start here in the qkv row
const V_OFF = K_OFF + KV_HEADS * HEAD; // 1024 — v sections start here

export class EagerAttnBlock extends nn.Module<[Value, Value, Value], Value> {
  // eps = 1e-5 matches the reference model. RMSNorm's own default of 1e-6
  // would silently diverge from it.
  readonly norm = new nn.RMSNorm(HIDDEN, 1e-5);
  readonly qkv = new nn.Linear(HIDDEN, QKV_WIDTH);
  readonly out = new nn.Linear(ATTN_OUT, HIDDEN);
  readonly sinks = new nn.Parameter(
    { elems: Q_HEADS, dtype: 'f32', dims: [1, Q_HEADS] },
    undefined,
    true, // attention sinks are f32 in every kernel variant
  );

  override forward(x: Value, cos: Value, sin: Value): Value {
    const qkv = this.qkv.forward(this.norm.forward(x)); // [T,1152]

    const qRoped = applyRope(slice(qkv, 1, 0, K_OFF), cos, sin); // [T, 896]
    const kRoped = applyRope(slice(qkv, 1, K_OFF, V_OFF), cos, sin); // [T, 128]
    const vAll = slice(qkv, 1, V_OFF, V_OFF + KV_HEADS * HEAD); // [T, 128]

    const attnOut = sdpa(qRoped, kRoped, vAll, {
      qHeads: Q_HEADS,
      kvHeads: KV_HEADS,
      headDim: HEAD,
      windowLeft: WINDOW,
      windowRight: WINDOW,
      sinks: this.sinks.value,
    });
    return add(x, this.out.forward(attnOut));
  }
}
