/** Folds a BatchNorm's affine and running stats into the preceding conv, as a
 *  lazy transform — nothing is fetched until the conv's weight is materialized.
 *
 *  Per out-channel: scale = gamma / sqrt(var + eps), weight' = weight · scale,
 *  bias' = beta + (bias − mean) · scale. Grouping does not matter, since the
 *  scaling is per out-channel row either way. */
import { derivedTensor, type LazyStateDict, type LazyTensor } from './safetensors.ts';

export function foldBnIntoConv(
  sd: LazyStateDict,
  convPrefix: string,
  bnPrefix: string,
  eps: number,
): void {
  const bn = (suffix: string): LazyTensor | undefined => sd.tensors.get(`${bnPrefix}.${suffix}`);
  const mean = bn('running_mean');
  if (!mean) return; // already fused
  const gamma = bn('weight');
  const beta = bn('bias');
  const variance = bn('running_var');
  const weight = sd.tensors.get(`${convPrefix}.weight`);
  if (!gamma || !beta || !variance || !weight) {
    throw new Error(
      `foldBnIntoConv: incomplete pair — need ${convPrefix}.weight + ${bnPrefix}.{weight,bias,running_mean,running_var}`,
    );
  }
  const priorBias = sd.tensors.get(`${convPrefix}.bias`);
  const cOut = gamma.shape[0]!;

  let memo: Promise<{ w: Float32Array; b: Float32Array }> | undefined;
  const folded = () =>
    (memo ??= (async () => {
      const [w, g, be, m, v] = await Promise.all([
        weight.f32(),
        gamma.f32(),
        beta.f32(),
        mean.f32(),
        variance.f32(),
      ]);
      const prior = priorBias ? await priorBias.f32() : undefined;
      const perOut = w.length / cOut;
      const b = new Float32Array(cOut);
      for (let o = 0; o < cOut; o++) {
        const scale = g[o]! / Math.sqrt(v[o]! + eps);
        const base = o * perOut;
        // In-place is safe only because `weight` is file-backed, so each f32()
        // decodes a fresh array. A derived tensor would hand back a shared
        // memoized array and this loop would corrupt it for every reader.
        for (let i = 0; i < perOut; i++) w[base + i]! *= scale;
        b[o] = be[o]! + ((prior?.[o] ?? 0) - m[o]!) * scale;
      }
      return { w, b };
    })());

  // All file bytes the fold downloads are attributed to the weight (bias
  // shares the same memoized fetch), keeping progress totals accurate.
  const foldedBytes =
    weight.byteLength +
    gamma.byteLength +
    beta.byteLength +
    mean.byteLength +
    variance.byteLength +
    (priorBias?.byteLength ?? 0);
  sd.tensors.set(
    `${convPrefix}.weight`,
    derivedTensor(weight.shape, async () => (await folded()).w, foldedBytes),
  );
  sd.tensors.set(
    `${convPrefix}.bias`,
    derivedTensor([cOut], async () => (await folded()).b),
  );
  for (const suffix of ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']) {
    sd.tensors.delete(`${bnPrefix}.${suffix}`);
  }
}
