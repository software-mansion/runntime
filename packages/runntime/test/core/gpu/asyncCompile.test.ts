/** Async pipeline compile: which submits wait for it, and what
 *  compilePipelines() does with a pipeline that compiles, fails or repeats. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  compilePipelines,
  needsAsyncCompile,
  setAsyncCompile,
  withAsyncCompile,
} from '../../../src/core/gpu/asyncCompile.ts';
import { warmUp } from '../../../src/core/gpu/errorScopes.ts';

/** A device with empty error scopes, all warmUp() and the scope need. */
function fakeDevice(): GPUDevice {
  return {
    pushErrorScope: () => undefined,
    popErrorScope: () => Promise.resolve(null),
  } as unknown as GPUDevice;
}

/** A pipeline that compiles when `finish` is called. */
function fakePipeline() {
  let finish!: () => void;
  let fail!: (err: unknown) => void;
  const done = new Promise<void>((resolve, reject) => {
    finish = resolve;
    fail = reject;
  });
  return { pipeline: { initAsync: vi.fn(() => done) }, finish, fail };
}

afterEach(() => setAsyncCompile(true));

describe('needsAsyncCompile', () => {
  it('is false outside a warm-up, so a regular call compiles on first use', () => {
    expect(needsAsyncCompile(fakeDevice(), [fakePipeline()])).toBe(false);
  });

  it('is true in a warm-up until the pipeline is compiled', async () => {
    const device = fakeDevice();
    const p = fakePipeline();
    p.finish();
    await withAsyncCompile(device, async () => {
      expect(needsAsyncCompile(device, [p])).toBe(true);
      await compilePipelines([p.pipeline]);
      expect(needsAsyncCompile(device, [p])).toBe(false);
    });
  });

  it('is true in a sync warm-up scope and false after it', () => {
    const device = fakeDevice();
    const p = fakePipeline();
    expect(withAsyncCompile(device, () => needsAsyncCompile(device, [p]))).toBe(true);
    expect(needsAsyncCompile(device, [p])).toBe(false);
  });

  it('is false after a warm-up that threw or rejected', async () => {
    const device = fakeDevice();
    const p = fakePipeline();
    expect(() =>
      withAsyncCompile(device, () => {
        throw new Error('sync');
      }),
    ).toThrow('sync');
    await expect(
      withAsyncCompile(device, () => Promise.reject(new Error('async'))),
    ).rejects.toThrow('async');
    expect(needsAsyncCompile(device, [p])).toBe(false);
  });

  it('stays true until the last of two overlapping warm-ups ends', async () => {
    const device = fakeDevice();
    const p = fakePipeline();
    let endFirst!: () => void;
    const first = withAsyncCompile(device, () => new Promise<void>((r) => (endFirst = r)));
    withAsyncCompile(device, () => undefined);
    expect(needsAsyncCompile(device, [p])).toBe(true);
    endFirst();
    await first;
    expect(needsAsyncCompile(device, [p])).toBe(false);
  });

  it('is per device', () => {
    const p = fakePipeline();
    const other = fakeDevice();
    expect(withAsyncCompile(fakeDevice(), () => needsAsyncCompile(other, [p]))).toBe(false);
  });

  it('is true outside a warm-up while the pipeline is mid-compile', async () => {
    const device = fakeDevice();
    const p = fakePipeline();
    const compiling = compilePipelines([p.pipeline]);
    expect(needsAsyncCompile(device, [p])).toBe(true);
    p.finish();
    await compiling;
    expect(needsAsyncCompile(device, [p])).toBe(false);
  });

  it('is true in warmUp() and false after it', async () => {
    const device = fakeDevice();
    const p = fakePipeline();
    let during: boolean | undefined;
    await warmUp(device, () => {
      during = needsAsyncCompile(device, [p]);
      return Promise.resolve(new Float32Array([1]));
    });
    expect(during).toBe(true);
    expect(needsAsyncCompile(device, [p])).toBe(false);
  });
});

describe('compilePipelines', () => {
  it('compiles a repeated pipeline once, and a compiled one never again', async () => {
    const a = fakePipeline();
    const b = fakePipeline();
    a.finish();
    b.finish();
    await compilePipelines([a.pipeline, b.pipeline, a.pipeline]);
    await compilePipelines([a.pipeline]);
    expect(a.pipeline.initAsync).toHaveBeenCalledTimes(1);
    expect(b.pipeline.initAsync).toHaveBeenCalledTimes(1);
  });

  it('starts each compile in a task of its own, before the first one finishes', async () => {
    const a = fakePipeline();
    const b = fakePipeline();
    const done = compilePipelines([a.pipeline, b.pipeline]);
    expect(a.pipeline.initAsync).toHaveBeenCalledTimes(1);
    // Microtasks run in the task that started a: b must not start in it.
    await Promise.resolve();
    expect(b.pipeline.initAsync).not.toHaveBeenCalled();
    // a is still compiling when b starts.
    await vi.waitFor(() => expect(b.pipeline.initAsync).toHaveBeenCalledTimes(1));
    a.finish();
    b.finish();
    await done;
  });

  it('reports a compile that fails while a later one waits to start', async () => {
    const a = fakePipeline();
    const b = fakePipeline();
    a.fail(new Error('bad shader'));
    b.finish();
    await expect(compilePipelines([a.pipeline, b.pipeline])).rejects.toMatchObject({
      code: 'EXECUTION_FAILED',
      message: 'GPU error during pipeline compile: bad shader',
    });
    expect(b.pipeline.initAsync).toHaveBeenCalledTimes(1);
  });

  it.each([
    { name: 'a rejected compile', initAsync: () => Promise.reject(new Error('bad shader')) },
    {
      name: 'a compile that throws',
      initAsync: () => {
        throw new Error('bad shader');
      },
    },
  ])('turns $name into EXECUTION_FAILED and compiles again next time', async ({ initAsync }) => {
    const pipeline = { initAsync: vi.fn<() => Promise<void>>(initAsync) };
    await expect(compilePipelines([pipeline])).rejects.toMatchObject({
      name: 'RunntimeError',
      code: 'EXECUTION_FAILED',
      message: 'GPU error during pipeline compile: bad shader',
      cause: new Error('bad shader'),
    });
    expect(needsAsyncCompile(fakeDevice(), [{ pipeline }])).toBe(false);
    pipeline.initAsync.mockResolvedValue(undefined);
    await compilePipelines([pipeline]);
    expect(pipeline.initAsync).toHaveBeenCalledTimes(2);
  });
});

describe('setAsyncCompile(false)', () => {
  it('leaves a warm-up to compile on first use', async () => {
    setAsyncCompile(false);
    const device = fakeDevice();
    const p = fakePipeline();
    expect(withAsyncCompile(device, () => needsAsyncCompile(device, [p]))).toBe(false);
    await compilePipelines([p.pipeline]);
    expect(p.pipeline.initAsync).not.toHaveBeenCalled();
  });
});
