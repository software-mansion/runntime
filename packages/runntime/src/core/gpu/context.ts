/** The default device set by initRunntime(). No engine imports, so
 *  buffers.ts can use it without an import cycle. */
import type { TgpuRoot } from 'typegpu';
import { RunntimeError } from '../error.ts';

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
    throw new RunntimeError(
      'NOT_INITIALIZED',
      'no default device — call initRunntime(root) first, or pass root explicitly',
    );
  }
  return currentRoot;
}

export function maybeDefaultRoot(): TgpuRoot | undefined {
  return currentRoot;
}

export function resetRunntime(): void {
  currentRoot = undefined;
}
