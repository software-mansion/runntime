import { describe, expect, it, vi } from 'vitest';
import { foldBnIntoConv } from '../../../src/core/weights/foldBn.ts';
import { memoryStateDict, type MemoryStateDict } from '../../../src/core/weights/safetensors.ts';

/** Values as bf16 bits. Every value used here is exact in bf16. memoryStateDict
 *  decodes bf16 into a fresh array on each read, like a tensor from a file. */
const bf16 = (values: number[]) =>
  Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

// Two out channels, two weights each. Worked by hand with eps = 1:
//   sqrt(var + eps)   = [2, 4]
//   scale = gamma / … = [2 / 2, 0.5 / 4]     = [1, 0.125]
//   weight' = weight · scale = [1, 2 | 3·0.125, 4·0.125] = [1, 2, 0.375, 0.5]
//   bias'   = beta + (bias − mean) · scale
const EPS = 1;
const conv = { data: bf16([1, 2, 3, 4]), shape: [2, 1, 1, 2] };
const bn = {
  'bn.weight': bf16([2, 0.5]), // gamma
  'bn.bias': bf16([1, -1]), // beta
  'bn.running_mean': bf16([1, 2]),
  'bn.running_var': bf16([3, 15]),
  'bn.num_batches_tracked': new Float32Array([100]),
};

function fold(extra: MemoryStateDict = {}) {
  const sd = memoryStateDict({ 'conv.weight': conv, ...bn, ...extra });
  foldBnIntoConv(sd, 'conv', 'bn', EPS);
  return sd;
}

describe('foldBnIntoConv', () => {
  it('scales the weight per out channel', async () => {
    const sd = fold();
    expect(await sd.tensors.get('conv.weight')!.f32()).toEqual(
      new Float32Array([1, 2, 0.375, 0.5]),
    );
    expect(sd.tensors.get('conv.weight')!.shape).toEqual([2, 1, 1, 2]);
  });

  it('makes a bias when the conv has none', async () => {
    // beta − mean · scale = [1 − 1·1, −1 − 2·0.125]
    const sd = fold();
    expect(await sd.tensors.get('conv.bias')!.f32()).toEqual(new Float32Array([0, -1.25]));
    expect(sd.tensors.get('conv.bias')!.shape).toEqual([2]);
  });

  it('folds into an existing bias', async () => {
    // beta + (bias − mean) · scale = [1 + (3 − 1)·1, −1 + (10 − 2)·0.125]
    const sd = fold({ 'conv.bias': bf16([3, 10]) });
    expect(await sd.tensors.get('conv.bias')!.f32()).toEqual(new Float32Array([3, 0]));
  });

  it('removes the batch norm tensors', () => {
    expect([...fold().tensors.keys()].sort()).toEqual(['conv.bias', 'conv.weight']);
  });

  it('keeps the download size, all of it on the weight', () => {
    const before = memoryStateDict({ 'conv.weight': conv, ...bn }).totalBytes();
    const sd = fold();
    expect(sd.tensors.get('conv.bias')!.byteLength).toBe(0);
    // num_batches_tracked is dropped without being read, so it is not counted
    expect(sd.totalBytes()).toBe(before - 4);
  });

  it('reads nothing until the weight or bias is read, then folds once', async () => {
    const sd = memoryStateDict({ 'conv.weight': conv, ...bn });
    // bf16 tensors decode on every read and never cache, so counting gamma's
    // reads counts how many times the fold ran
    const gammaReads = vi.spyOn(sd.tensors.get('bn.weight')!, 'f32');
    foldBnIntoConv(sd, 'conv', 'bn', EPS);
    expect(gammaReads).not.toHaveBeenCalled();
    await Promise.all([sd.tensors.get('conv.weight')!.f32(), sd.tensors.get('conv.bias')!.f32()]);
    expect(gammaReads).toHaveBeenCalledOnce();
  });

  it('does nothing when the batch norm is already folded', () => {
    const sd = memoryStateDict({ 'conv.weight': conv });
    const weight = sd.tensors.get('conv.weight');
    foldBnIntoConv(sd, 'conv', 'bn', EPS);
    expect(sd.tensors.get('conv.weight')).toBe(weight);
    expect(sd.tensors.has('conv.bias')).toBe(false);
  });

  it.each(['bn.weight', 'bn.bias', 'bn.running_var', 'conv.weight'])(
    'rejects a checkpoint missing %s',
    (missing) => {
      const sd = memoryStateDict({ 'conv.weight': conv, ...bn });
      sd.tensors.delete(missing);
      expect(() => foldBnIntoConv(sd, 'conv', 'bn', EPS)).toThrow(
        expect.objectContaining({ name: 'RunntimeError', code: 'CHECKPOINT_MISMATCH' }),
      );
    },
  );
});
