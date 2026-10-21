import { describe, expect, it, onTestFinished } from 'vitest';
import { writeF32 } from '../../../src/core/gpu/buffers.ts';
import { evalValues, gpuExecutor, toArray } from '../../../src/core/gpu/eval.ts';
import { matmul } from '../../../src/core/graph/ops.ts';
import { matrix, type Value } from '../../../src/core/graph/value.ts';
import type { F32Buffer } from '../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, upload } from '../../gpu.ts';

const { root } = await gpu();

const K = 8;
const a = randomF32(K * K, 2);
const b = randomF32(K * K, 3);

/** (x · a) · b in f64, x [rows, K]. */
function ref(x: Float32Array, rows: number): Float64Array {
  const mul = (l: ArrayLike<number>, r: ArrayLike<number>) => {
    const out = new Float64Array(rows * K);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < K; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += l[i * K + k]! * r[k * K + j]!;
        out[i * K + j] = s;
      }
    }
    return out;
  };
  return mul(mul(x, a), b);
}

/** (x · a) · b on the GPU. x · a is a pooled intermediate. */
function forward(x: Value): Value {
  return matmul(matmul(x, upload(root, a, matrix(K, K))), upload(root, b, matrix(K, K)));
}

function freshExecutor() {
  const ex = gpuExecutor(root);
  onTestFinished(() => ex.dispose());
  return ex;
}

describe('gpuExecutor trim', () => {
  // Each call trims after it, the way a model with varying input lengths
  // runs. A length unused by the previous call has had its buffers destroyed.
  it.for<{ name: string; lengths: number[] }>([
    { name: 'the same length again', lengths: [5, 5, 5] },
    { name: 'a new length, then the first again', lengths: [5, 9, 5] },
    { name: 'a new length every call', lengths: [3, 7, 12, 1, 20] },
  ])('$name', async ({ lengths }) => {
    const ex = freshExecutor();
    for (const [i, rows] of lengths.entries()) {
      const x = randomF32(rows * K, 10 + i);
      const got = await toArray(forward(upload(root, x, matrix(rows, K))), ex);
      ex.trim();
      expectClose(got, ref(x, rows));
    }
  });

  it('does nothing once captured, so the replay keeps its buffers', async () => {
    const ex = freshExecutor();
    const rows = 4;
    const x = upload(root, randomF32(rows * K, 20), matrix(rows, K));
    const { targets: y, replay } = ex.captureFrame(() => {
      const y = forward(x);
      evalValues([y], ex);
      return y;
    });
    // Two trims: the first keeps what the capture used, the second would
    // destroy the captured intermediate if trim ran on a captured executor.
    ex.trim();
    ex.trim();
    const next = randomF32(rows * K, 21);
    writeF32(root, x.buffer as F32Buffer, next);
    root.device.pushErrorScope('validation');
    const got = ex.readbackOnSubmit(y);
    replay();
    expect(await root.device.popErrorScope()).toBeNull();
    expectClose(await got, ref(next, rows));
  });
});
