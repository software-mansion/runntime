/** The kernel A/B runner. A is the base ref's `src` (copied to `.bench/base`
 *  by `bench/cli.ts`), B is this checkout's `src`. Both run in one page on the
 *  same device and inputs, with their rounds interleaved so GPU clock
 *  changes hit both versions alike. */
import tgpu, { type TgpuRoot } from 'typegpu';
import { afterAll, describe, inject, it } from 'vitest';
import { commands } from 'vitest/browser';
import * as libA from '@ab/a/core/index.ts';
import * as libB from '@ab/b/core/index.ts';
import { randomF32 } from '../test/gpu.ts';
import { installProfiler, type Profiler } from './profiler.ts';
import {
  callsPerRound,
  compare,
  confirm,
  maxAbsDiff,
  median,
  spread,
  type CaseResult,
} from './stats.ts';

/** The core API. Both versions are typed as this checkout's; a case that uses
 *  something the base lacks throws on A and is reported as skipped. */
export type Lib = typeof libB;
type Value = libB.Value;
type ValueMeta = libB.ValueMeta;

/** What a case builds with, for one of the two library versions: A (the
 *  base ref) or B (this checkout). */
export interface Version {
  label: 'A' | 'B';
  lib: Lib;
  root: TgpuRoot;
  /** `tensor()` on this version's device, freed after the case. */
  upload: (data: Float32Array, meta: ValueMeta) => Value;
  /** Destroys the Value's buffer when the case ends and returns the Value as
   *  is. For buffers `upload` cannot make (`materialized` + `uploadF16`). */
  freeAfterCase: (value: Value) => Value;
  /** Seeded values in [-1, 1). The seed restarts for each version, so A and B
   *  draw the same numbers in the same order. */
  rand: (n: number) => Float32Array;
}

export interface AbCase {
  /** Route and where the shape comes from, e.g. 'tiled, MiniLM qkv'. */
  name: string;
  /** Device features the route needs. Missing ones skip the case. */
  needs?: GPUFeatureName[];
  /** Runs on a second device without `subgroups`, for the routes that only
   *  exist there. */
  noSubgroups?: boolean;
  /** Uploads the inputs and returns a function that builds one call. */
  build: (version: Version) => () => Value;
}

/** Rounds per version. Alternating ABBA over 8 rounds puts each version first
 *  equally often. */
const ROUNDS = 8;

/** Warm-up runs at least MIN_WARMUP_MS and stops once three rounds in a row
 *  agree within SETTLED_SPREAD, or at MAX_WARMUP_MS. */
const MIN_WARMUP_MS = 300;
const MAX_WARMUP_MS = 5000;
const SETTLED_SPREAD = 0.02;

/** The GPU device a case runs on, and one executor per version on it. */
interface GpuSetup {
  root: TgpuRoot;
  profiler: Profiler;
  executors: [libA.RunntimeExecutor, libB.RunntimeExecutor];
}

/** Keyed by noSubgroups. */
const gpuSetups = new Map<boolean, Promise<GpuSetup>>();

/** Creates the device once and reuses it: the production feature set, or the
 *  same without `subgroups`. A second device is needed for that, since
 *  features are fixed when a device is created. */
function gpuSetup(noSubgroups: boolean): Promise<GpuSetup> {
  let setup = gpuSetups.get(noSubgroups);
  if (!setup) {
    const features: GPUFeatureName[] = ['shader-f16', 'timestamp-query'];
    if (!noSubgroups) features.push('subgroups');
    setup = tgpu.init({ device: { optionalFeatures: features } }).then((root) => ({
      root,
      profiler: installProfiler(root.device),
      executors: [libA.gpuExecutor(root), libB.gpuExecutor(root)],
    }));
    gpuSetups.set(noSubgroups, setup);
  }
  return setup;
}

/** GPU µs per call over one round of `calls` calls in a single submit. */
async function round(gpu: GpuSetup, i: 0 | 1, call: () => Value, calls: number): Promise<number> {
  const lib = i === 0 ? (libA as Lib) : libB;
  const ys = Array.from({ length: calls }, call);
  lib.evalValues(ys, gpu.executors[i]);
  gpu.executors[i].releaseValues(ys);
  await gpu.root.device.queue.onSubmittedWorkDone();
  const ns = (await gpu.profiler.collect()).reduce((a, b) => a + b, 0);
  if (ns === 0) throw new Error(`no GPU time recorded on ${i === 0 ? 'A' : 'B'}`);
  return ns / 1000 / calls;
}

/** Measures a case. If it comes out faster or slower, measures it again on
 *  fresh buffers and keeps that verdict only if both runs agree. */
async function runCase(kernel: string, testCase: AbCase, gpu: GpuSetup): Promise<CaseResult> {
  try {
    const first = await measure(kernel, testCase, gpu, false);
    if (first.comparison?.verdict !== 'same' && first.comparison) {
      // Run again with B's buffers created first. If the first result only
      // came from where the buffers landed in memory, it goes away or flips,
      // and confirm() turns the verdict into 'same'.
      const second = await measure(kernel, testCase, gpu, true);
      if (second.comparison) {
        return { ...first, comparison: confirm(first.comparison, second.comparison) };
      }
    }
    return first;
  } finally {
    // The buffer pools keep a buffer for every output size they have seen.
    // Without freeing them after each case, a whole file's cases use enough
    // GPU memory to crash the browser tab.
    gpu.executors[0].dispose();
    gpu.executors[1].dispose();
  }
}

async function measure(
  kernel: string,
  testCase: AbCase,
  gpu: GpuSetup,
  bFirst: boolean,
): Promise<CaseResult> {
  const owned: Value[] = [];
  const freeAfterCase = (v: Value) => (owned.push(v), v);
  const calls: (() => Value)[] = [];
  try {
    for (const i of bFirst ? [1, 0] : [0, 1]) {
      const label = i === 0 ? 'A' : 'B';
      const lib = i === 0 ? (libA as Lib) : libB;
      const upload = (data: Float32Array, meta: ValueMeta) =>
        freeAfterCase(lib.tensor(gpu.root, data, meta));
      let seed = 1;
      const rand = (n: number) => randomF32(n, seed++);
      try {
        calls[i] = testCase.build({ label, lib, root: gpu.root, upload, freeAfterCase, rand });
      } catch (err) {
        return {
          kernel,
          case: testCase.name,
          note: `skipped, threw on ${label}: ${messageOf(err)}`,
        };
      }
    }
    const [callA, callB] = [calls[0]!, calls[1]!];

    // Output check, which also compiles both versions' pipelines.
    const yA = callA();
    const yB = callB();
    const outA = await readOut(libA as Lib, yA, gpu.executors[0]);
    const outB = await readOut(libB, yB, gpu.executors[1]);
    gpu.executors[0].releaseValues([yA]);
    gpu.executors[1].releaseValues([yB]);
    const outBytes = yB.shape.elems * (yB.shape.dtype === 'f16' ? 2 : 4);

    // Apple GPUs idle at a low clock and ramp up only under sustained load,
    // so warm up with full-size rounds on both versions until B's time settles,
    // and size the timed rounds from there.
    const start = performance.now();
    const recent: number[] = [];
    let n = 4;
    for (;;) {
      await round(gpu, 0, callA, n);
      recent.push(await round(gpu, 1, callB, n));
      n = callsPerRound(recent.at(-1)!, outBytes);
      const elapsed = performance.now() - start;
      const settled = recent.length >= 3 && spread(recent.slice(-3)) <= SETTLED_SPREAD;
      if (elapsed >= MAX_WARMUP_MS || (elapsed >= MIN_WARMUP_MS && settled)) break;
    }
    n = callsPerRound(median(recent.slice(-3)), outBytes);

    const times: [number[], number[]] = [[], []];
    for (let r = 0; r < ROUNDS; r++) {
      const order = r % 4 === 0 || r % 4 === 3 ? ([0, 1] as const) : ([1, 0] as const);
      for (const i of order) times[i].push(await round(gpu, i, calls[i]!, n));
    }
    return {
      kernel,
      case: testCase.name,
      comparison: compare(times[0], times[1]),
      maxAbsDiff: maxAbsDiff(outA, outB),
      calls: n,
    };
  } finally {
    for (const v of owned) if (v.state === 'materialized') v.buffer.destroy();
  }
}

/** Reads an output for the A/B check. hwc4 cannot be read back as is, so it
 *  goes through toChw first; that dispatch is never timed. */
function readOut(lib: Lib, y: Value, executor: libB.RunntimeExecutor): Promise<Float32Array> {
  return lib.toArray(y.shape.layout === 'hwc4' ? lib.toChw(y, 'f32') : y, executor);
}

function messageOf(err: unknown): string {
  return (err instanceof Error ? err.message : String(err)).split('\n')[0]!.slice(0, 120);
}

/** Registers one test per case and writes the kernel's results to
 *  `.bench/results/<kernel>.json` for `bench/cli.ts` to report. */
export function abSuite(kernel: string, cases: readonly AbCase[]): void {
  const results: CaseResult[] = [];
  describe(kernel, () => {
    for (const testCase of cases) {
      it(testCase.name, async (ctx) => {
        const gpu = await gpuSetup(testCase.noSubgroups === true);
        const missing = testCase.needs?.find((f) => !gpu.root.enabledFeatures.has(f));
        let result: CaseResult;
        try {
          result = missing
            ? { kernel, case: testCase.name, note: `skipped, device has no ${missing}` }
            : await runCase(kernel, testCase, gpu);
        } catch (err) {
          // Still a row in the report, and still a failed test.
          results.push({ kernel, case: testCase.name, note: `failed: ${messageOf(err)}` });
          throw err;
        }
        results.push(result);
        if (result.note) ctx.skip(result.note);
      });
    }
  });
  afterAll(async () => {
    await commands.writeFile(
      `${inject('abResultsDir')}/${kernel}.json`,
      JSON.stringify(results, null, 2),
    );
  });
}

declare module 'vitest' {
  export interface ProvidedContext {
    abResultsDir: string;
  }
}
