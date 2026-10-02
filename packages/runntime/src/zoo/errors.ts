/** Every error from a create<Task>() call or a model method is a
 *  RunntimeError. An error that has a code keeps it. Any other error gets
 *  the code of the step that failed. */

import {
  isRunntimeError,
  RunntimeError,
  type LazyStateDict,
  type LazyTensor,
} from '../core/index.ts';

export const isAbort = (err: unknown): boolean =>
  (err as { name?: unknown } | null)?.name === 'AbortError';

/** The weights file does not fit the model. `what` names the file, like
 *  'yolo26 weights'. */
export const checkpointMismatch = (what: string, message: string) =>
  new RunntimeError('CHECKPOINT_MISMATCH', `${what}: ${message}`);

/** Returns the tensor at `key`, or throws CHECKPOINT_MISMATCH if it is
 *  missing. `where` names the layer that asked. The state dict is not
 *  changed. */
export function requireTensor(sd: LazyStateDict, key: string, where: string): LazyTensor {
  const t = sd.tensors.get(key);
  if (!t) throw checkpointMismatch(where, `missing '${key}' in the state dict`);
  return t;
}

export const messageOf = (err: unknown): string =>
  err instanceof Error ? err.message : String(err);

/** The error a create<Task>() call throws:
 *  - signal fired: LOAD_ABORTED, with the signal's reason as cause. We check
 *    the signal, not the error, because abort(reason) makes fetch reject
 *    with that reason, which is not always an AbortError.
 *  - a RunntimeError: kept as it is.
 *  - an AbortError, like one from a custom RangeSource: LOAD_ABORTED.
 *  - anything else: LOAD_FAILED, with the error as cause. */
export function asLoadError(err: unknown, signal?: AbortSignal): RunntimeError {
  if (signal?.aborted) {
    if (isRunntimeError(err, 'LOAD_ABORTED')) return err;
    return new RunntimeError('LOAD_ABORTED', 'model load aborted', { cause: signal.reason });
  }
  if (isRunntimeError(err)) return err;
  if (isAbort(err)) return new RunntimeError('LOAD_ABORTED', 'model load aborted', { cause: err });
  return new RunntimeError('LOAD_FAILED', messageOf(err), { cause: err });
}

/** Catch handler for a model's work. A RunntimeError is kept as it is.
 *  Anything else (a lost device, a failed readback) becomes
 *  EXECUTION_FAILED, with the error as cause. */
export function rethrowRunError(err: unknown): never {
  if (isRunntimeError(err)) throw err;
  throw new RunntimeError('EXECUTION_FAILED', messageOf(err), { cause: err });
}
