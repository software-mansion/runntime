/** State dict helpers shared by the zoo models' hooks. */

import type { LazyStateDict, LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';

/** Row order for permuteRows that takes each `block` rows from the
 *  checkpoint's half-split rope layout to the kernel's interleaved pairs:
 *  rows (p, p + block/2) become (2p, 2p + 1). */
export function ropeOrder(rows: number, block: number): number[] {
  return Array.from({ length: rows }, (_, r) => {
    const i = r % block;
    return r - i + (i % 2 ? (i - 1) / 2 + block / 2 : i / 2);
  });
}

/** Moves the tensors under `sources` out of `sd`. take() throws for a
 *  missing one, done() for any left unused. */
export function remap(sd: LazyStateDict, prefix: string, sources: readonly string[], what: string) {
  const src = new Map<string, LazyTensor>();
  for (const [key, t] of sd.tensors) {
    if (sources.some((s) => key.startsWith(s))) src.set(key, t);
  }
  sd.tensors.clear();
  return {
    take(name: string): LazyTensor {
      const t = src.get(name);
      if (!t) throw checkpointMismatch(what, `missing tensor '${name}'`);
      src.delete(name);
      return t;
    },
    put(name: string, t: LazyTensor): void {
      sd.tensors.set(prefix ? `${prefix}.${name}` : name, t);
    },
    done(): void {
      if (src.size)
        throw checkpointMismatch(what, `unused tensors: ${[...src.keys()].slice(0, 5)}`);
    },
  };
}
