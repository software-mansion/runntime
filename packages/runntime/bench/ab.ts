/** The kernel A/B runner. A is the base ref's `src` (copied to `.bench/base`
 *  by `bench/cli.ts`), B is this checkout's `src`. Both run in one page on one
 *  device, on the same inputs, with their rounds interleaved so GPU clock
 *  changes hit both sides alike. */
import tgpu, { type TgpuRoot } from 'typegpu';
import { afterAll, describe, inject, it } from 'vitest';
import { commands } from 'vitest/browser';
import * as libA from '@ab/a/core/index.ts';
import * as libB from '@ab/b/core/index.ts';
import { installProfiler } from './profiler.ts';
import { callsPerRound, compare, maxAbsDiff, median, spread, type CaseResult } from './stats.ts';

/** The core API. Both sides are typed as this checkout's; a case that uses
 *  something the base lacks throws on A and is reported as skipped. */
export type Lib = typeof libB;
type Value = libB.Value;
type ValueMeta = libB.ValueMeta;

/** Uploads one input on the side being built. Freed after the case. */
export type Upload = (data: Float32Array, meta: ValueMeta) => Value;

export interface AbCase {
  /** Route and where the shape comes from, e.g. 'tiled, whisper-base fc1'. */
  name: string;
  /** Device features the route needs. Missing ones skip the case. */
  needs?: GPUFeatureName[];
  /** Uploads the inputs and returns a function that builds one call. Seed
   *  any random data, so both sides get the same numbers. */
  build: (lib: Lib, upload: Upload, root: TgpuRoot) => () => Value;
}

/** Rounds per side. Alternating ABBA over 8 rounds puts each side first
 *  equally often. */
const ROUNDS = 8;

/** Warm-up runs at least MIN_WARMUP_MS and stops once three rounds in a row
 *  agree within SETTLED_SPREAD, or at MAX_WARMUP_MS. */
const MIN_WARMUP_MS = 300;
const MAX_WARMUP_MS = 5000;
const SETTLED_SPREAD = 0.02;

const root = await tgpu.init({
  device: { optionalFeatures: ['subgroups', 'shader-f16', 'timestamp-query'] },
});
const profiler = installProfiler(root.device);
const sides = [
  { label: 'A', lib: libA as Lib, ex: libA.initRunntime(root) },
  { label: 'B', lib: libB, ex: libB.initRunntime(root) },
] as const;
type Side = (typeof sides)[number];

/** GPU µs per call over one round of `calls` calls in a single submit. */
async function round(side: Side, call: () => Value, calls: number): Promise<number> {
  const ys = Array.from({ length: calls }, call);
  side.lib.evalValues(ys, side.ex);
  side.ex.releaseValues(ys);
  await root.device.queue.onSubmittedWorkDone();
  const ns = (await profiler.collect()).reduce((a, b) => a + b, 0);
  if (ns === 0) throw new Error(`no GPU time recorded on ${side.label}`);
  return ns / 1000 / calls;
}

async function runCase(kernel: string, c: AbCase): Promise<CaseResult> {
  const uploads: Value[] = [];
  const calls: (() => Value)[] = [];
  try {
    for (const side of sides) {
      const upload: Upload = (data, meta) => {
        const v = side.lib.tensor(root, data, meta);
        uploads.push(v);
        return v;
      };
      try {
        calls.push(c.build(side.lib, upload, root));
      } catch (err) {
        return { kernel, case: c.name, note: `skipped, threw on ${side.label}: ${messageOf(err)}` };
      }
    }
    const [a, b] = [sides[0], sides[1]];
    const [callA, callB] = [calls[0]!, calls[1]!];

    // Output check, which also compiles both sides' pipelines.
    const yA = callA();
    const yB = callB();
    const outA = await a.lib.toArray(yA, a.ex);
    const outB = await b.lib.toArray(yB, b.ex);
    a.ex.releaseValues([yA]);
    b.ex.releaseValues([yB]);
    const outBytes = yB.shape.elems * (yB.shape.dtype === 'f16' ? 2 : 4);

    // Apple GPUs idle at a low clock and take a few hundred ms of work to
    // ramp up, so warm up on both sides before sizing the rounds from B.
    const start = performance.now();
    const recent: number[] = [];
    for (;;) {
      await round(a, callA, 4);
      recent.push(await round(b, callB, 4));
      const elapsed = performance.now() - start;
      const settled = recent.length >= 3 && spread(recent.slice(-3)) <= SETTLED_SPREAD;
      if (elapsed >= MAX_WARMUP_MS || (elapsed >= MIN_WARMUP_MS && settled)) break;
    }
    const n = callsPerRound(median(recent.slice(-3)), outBytes);

    const times: [number[], number[]] = [[], []];
    for (let r = 0; r < ROUNDS; r++) {
      const order = r % 4 === 0 || r % 4 === 3 ? [0, 1] : [1, 0];
      for (const i of order) times[i]!.push(await round(sides[i]!, calls[i]!, n));
    }
    return {
      kernel,
      case: c.name,
      comparison: compare(times[0], times[1]),
      maxAbsDiff: maxAbsDiff(outA, outB),
      calls: n,
    };
  } finally {
    for (const v of uploads) v.buffer.destroy();
  }
}

function messageOf(err: unknown): string {
  return (err instanceof Error ? err.message : String(err)).split('\n')[0]!.slice(0, 120);
}

/** Registers one test per case and writes the kernel's results to
 *  `.bench/results/<kernel>.json` for `bench/cli.ts` to report. */
export function abSuite(kernel: string, cases: readonly AbCase[]): void {
  const results: CaseResult[] = [];
  describe(kernel, () => {
    for (const c of cases) {
      it(c.name, async (ctx) => {
        const missing = c.needs?.find((f) => !root.enabledFeatures.has(f));
        if (missing) {
          results.push({ kernel, case: c.name, note: `skipped, device has no ${missing}` });
          ctx.skip(`device has no ${missing}`);
        }
        try {
          const result = await runCase(kernel, c);
          results.push(result);
          if (result.note) ctx.skip(result.note);
        } catch (err) {
          // Still a row in the report, and still a failed test.
          results.push({ kernel, case: c.name, note: `failed: ${messageOf(err)}` });
          throw err;
        }
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
