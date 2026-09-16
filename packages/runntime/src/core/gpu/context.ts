/** Default-device state behind initRunntime(). Import-free, so buffers.ts can
 *  resolve the default root without a cycle through the executor. */
import type { TgpuRoot } from 'typegpu';

let currentRoot: TgpuRoot | undefined;

export const executorPerRoot = new WeakMap<TgpuRoot, unknown>();

export function setDefaultRoot(root: TgpuRoot): void {
  currentRoot = root;
}

export function supportsF16(root: TgpuRoot = defaultRoot()): boolean {
  return root.enabledFeatures.has('shader-f16');
}

export function defaultRoot(): TgpuRoot {
  if (!currentRoot) {
    throw new Error('no default device — call initRunntime(root) first, or pass root explicitly');
  }
  return currentRoot;
}

export function maybeDefaultRoot(): TgpuRoot | undefined {
  return currentRoot;
}

export function resetRunntime(): void {
  currentRoot = undefined;
}
