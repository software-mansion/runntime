/**
 * Run `fn` inside WebGPU error scopes. OOM and validation errors never throw
 * in JS — the queue silently discards the bad work and readbacks come back
 * all-zero — so without this an entire failed run looks like confident
 * zeros. Scopes are a per-device stack: callers must not overlap two scoped
 * regions on the same device.
 */
export async function inGpuErrorScopes<T>(
  device: GPUDevice,
  stage: string,
  fn: () => Promise<T>,
): Promise<T> {
  // 'internal' matters: per-device shader compiler failures (the kind that
  // only reproduce on one GPU) report as internal, not validation.
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');
  device.pushErrorScope('validation');
  // The three pops are issued back to back so they share one round trip.
  const popAll = async () => {
    const [validation, internal, oom] = await Promise.all([
      device.popErrorScope(),
      device.popErrorScope(),
      device.popErrorScope(),
    ]);
    return oom ?? internal ?? validation;
  };
  let result: T;
  try {
    result = await fn();
  } catch (err) {
    await popAll().catch(() => undefined);
    throw err;
  }
  const gpuError = await popAll();
  if (gpuError) {
    throw new Error(`GPU error during ${stage}: ${gpuError.message}`, { cause: gpuError });
  }
  return result;
}

export async function warmUp(
  device: GPUDevice,
  run: () => Promise<Float32Array>,
  stage = 'pipeline warm-up',
): Promise<Float32Array> {
  const out = await inGpuErrorScopes(device, stage, run);
  if (out.length === 0 || !out.every(Number.isFinite) || out.every((v) => v === 0)) {
    throw new Error(
      `${stage} produced unusable output (non-finite or all zero) — inference fails on this device`,
    );
  }
  return out;
}
