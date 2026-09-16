/** The storage dtype a kernel's buffers carry, closed over by the factory
 *  rather than passed through an accessor, so its branches fold at transpile
 *  time.
 *
 *  Only storage varies: accumulators stay f32 everywhere, since an f16
 *  accumulator over a long reduction drifts percent-level. */
import { d } from 'typegpu';

export interface Elem {
  readonly key: 'f32' | 'f16';
  readonly scalar: d.F32 | d.F16;
  readonly vec4: d.Vec4f;
}

export const F32_ELEM: Elem = { key: 'f32', scalar: d.f32, vec4: d.vec4f };
export const F16_ELEM: Elem = {
  key: 'f16',
  scalar: d.f16,
  vec4: d.vec4h as unknown as d.Vec4f,
};

export function elemFor(dtype: string): Elem {
  if (dtype === 'f16') return F16_ELEM;
  if (dtype === 'f32') return F32_ELEM;
  throw new Error(`no kernel element type for dtype '${dtype}'`);
}
