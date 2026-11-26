/** Maps the HF Moonshine checkpoint to the model. Every tensor binds under
 *  its own name, since the modules are named like the checkpoint, except the
 *  tied `proj_out.weight`, which the model never reads and is dropped.
 *
 *  No rotary permutation is needed, since HF's convention already matches the
 *  kernel's. */

import type { LazyStateDict, LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';

export const qualify = (prefix: string, name: string): string =>
  prefix ? `${prefix}.${name}` : name;

export function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('moonshine state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

export function transformDecoder(sd: LazyStateDict, prefix: string): void {
  // The LM head is the embedding weight, so the tied copy is never read.
  sd.tensors.delete(qualify(prefix, 'proj_out.weight'));
}

export function transformMoonshineStateDict(sd: LazyStateDict, prefix: string): void {
  transformDecoder(sd, prefix);
}
