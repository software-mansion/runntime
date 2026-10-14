/** Shared setup for `*.gpu.test.ts`: one real WebGPU device per test file,
 *  plus the float comparison every kernel test uses. */
import tgpu, { type TgpuRoot } from 'typegpu';
import { expect, type TestContext } from 'vitest';
import { initRunntime, type RunntimeExecutor } from '../src/core/gpu/eval.ts';

let device: Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> | undefined;

/** The device the library asks for in production, with the same optional
 *  features, so tests take the same kernel routes. */
export function gpu(): Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> {
  device ??= tgpu
    .init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } })
    .then((root) => ({ root, ex: initRunntime(root) }));
  return device;
}

/** Skips the test, with the reason in the report, when the device lacks
 *  `feature`. */
export function requireFeature(ctx: TestContext, root: TgpuRoot, feature: GPUFeatureName): void {
  if (!root.enabledFeatures.has(feature)) ctx.skip(`device has no ${feature}`);
}

/** Fails on the first element where `|got - want| > atol + rtol * |want|`,
 *  or where either side is NaN. */
export function expectClose(
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  { atol = 1e-5, rtol = 1e-5 } = {},
): void {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) {
    const g = got[i]!;
    const w = want[i]!;
    // Written so NaN on either side fails the comparison.
    if (!(Math.abs(g - w) <= atol + rtol * Math.abs(w))) {
      expect.fail(`element ${i}: got ${g}, want ${w} (atol ${atol}, rtol ${rtol})`);
    }
  }
}

/** Deterministic values in [-1, 1), so a failure reproduces. */
export function randomF32(n: number, seed = 1): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = s / 2 ** 31 - 1;
  }
  return out;
}
