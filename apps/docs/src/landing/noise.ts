/** Hash and value noise shared by the landing page shaders. */

import tgpu, { d, std } from 'typegpu';

/** A pseudo-random value in [0, 1) per point. */
export const hash21 = tgpu.fn(
  [d.vec2f],
  d.f32,
)((p) => {
  'use gpu';
  const q = std.fract(p * d.vec2f(123.34, 456.21));
  const r = q + std.dot(q, q + 45.32);
  return std.fract(r.x * r.y);
});

/** Smooth value noise in [0, 1], one lattice cell per unit. */
export const valueNoise = tgpu.fn(
  [d.vec2f],
  d.f32,
)((p) => {
  'use gpu';
  const i = std.floor(p);
  const f = std.fract(p);
  const u = f * f * (d.vec2f(3) - f * 2);
  const a = hash21(i);
  const b = hash21(i + d.vec2f(1, 0));
  const c = hash21(i + d.vec2f(0, 1));
  const e = hash21(i + d.vec2f(1, 1));
  return std.mix(std.mix(a, b, u.x), std.mix(c, e, u.x), u.y);
});
