/** Shared setup for `*.gpu.test.ts`: one real WebGPU device per test file,
 *  the float comparison every kernel test uses, and a count of live GPU
 *  buffers for the leak checks. */
import tgpu, { type TgpuRoot } from 'typegpu';
import { expect, onTestFinished, type TestContext } from 'vitest';
import { tensor } from '../src/core/gpu/buffers.ts';
import {
  gpuExecutor,
  initRunntime,
  subgroupKernelsOk,
  type RunntimeExecutor,
} from '../src/core/gpu/eval.ts';
import type { Value, ValueMeta } from '../src/core/graph/value.ts';

/** Every GPU buffer created on this page and not destroyed yet. Counting
 *  starts when this module loads, so a leak check compares two snapshots,
 *  never an absolute count. */
const live = new Set<GPUBuffer>();
// Node tests import this file for randomF32 and have no WebGPU to wrap.
if (typeof GPUDevice !== 'undefined') {
  const createBuffer = GPUDevice.prototype.createBuffer;
  const destroyBuffer = GPUBuffer.prototype.destroy;
  GPUDevice.prototype.createBuffer = function (desc) {
    const buf = createBuffer.call(this, desc);
    live.add(buf);
    return buf;
  };
  GPUBuffer.prototype.destroy = function () {
    live.delete(this);
    destroyBuffer.call(this);
  };
}

/** A snapshot of the buffers alive now. */
export function liveBuffers(): ReadonlySet<GPUBuffer> {
  return new Set(live);
}

/** Size and usage of each live buffer missing from `before`, so a failure
 *  names what leaked. */
export function buffersSince(before: ReadonlySet<GPUBuffer>): string[] {
  return [...live].filter((b) => !before.has(b)).map((b) => `${b.size} B, usage ${b.usage}`);
}

/** What `buffersSince` lists after `times` runs that each leave `perRun`. */
export function repeated(perRun: readonly string[], times: number): string[] {
  return Array.from({ length: times }, () => perRun).flat();
}

let device: Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> | undefined;

/** The device the library asks for in production, with the same optional
 *  features, so tests take the same kernel routes. */
export function gpu(): Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> {
  device ??= tgpu
    .init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } })
    .then((root) => ({ root, ex: initRunntime(root) }));
  return device;
}

let deviceWithoutSubgroups: Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> | undefined;

/** A second device with the same features minus `subgroups`, for the routes
 *  only such a device takes. Features are fixed when a device is created, so
 *  it cannot be the one gpu() returns. */
export function gpuWithoutSubgroups(): Promise<{ root: TgpuRoot; ex: RunntimeExecutor }> {
  deviceWithoutSubgroups ??= tgpu
    .init({ device: { optionalFeatures: ['shader-f16'] } })
    .then((root) => ({ root, ex: gpuExecutor(root) }));
  return deviceWithoutSubgroups;
}

/** `tensor()` whose buffer is destroyed when the test ends: uploads are
 *  outside the executor's pool, so the caller frees them. */
export function upload(
  root: TgpuRoot,
  data: Float32Array,
  shape: readonly [number, number] | ValueMeta,
): Value {
  const value = tensor(root, data, shape);
  onTestFinished(() => value.buffer.destroy());
  return value;
}

/** Skips the test, with the reason in the report, when the device lacks
 *  `feature`. `subgroups` uses the executor's own check, so a case written for
 *  a subgroup route never passes on another one. */
export function requireFeature(ctx: TestContext, root: TgpuRoot, feature: GPUFeatureName): void {
  if (feature === 'subgroups' && !subgroupKernelsOk(root.device)) {
    ctx.skip('device has no 32-wide subgroups');
  }
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

/** Rounds each value to the nearest f16, so a reference sees the same inputs
 *  an f16 kernel does. */
export function roundF16(data: Float32Array): Float32Array {
  // Math.f16round ships in Chromium but is missing from the ES2022 lib types.
  const { f16round } = Math as Math & { f16round(x: number): number };
  return data.map(f16round);
}
