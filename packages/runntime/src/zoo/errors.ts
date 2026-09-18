/** Every error out of a factory or a runner method is a RunntimeError.
 *  Coded errors pass through, the rest get the code of the phase they
 *  failed in. */

import { isRunntimeError, RunntimeError } from '../core/index.ts';

export const isAbort = (err: unknown): boolean =>
  (err as { name?: unknown } | null)?.name === 'AbortError';

const messageOf = (err: unknown): string => (err instanceof Error ? err.message : String(err));

/** The error a create<Task>() factory throws. A RunntimeError passes
 *  through with its code, an aborted fetch becomes LOAD_ABORTED, anything
 *  else LOAD_FAILED with the original error as cause. */
export function asLoadError(err: unknown): RunntimeError {
  if (isRunntimeError(err)) return err;
  if (isAbort(err)) return new RunntimeError('LOAD_ABORTED', 'model load aborted', { cause: err });
  return new RunntimeError('LOAD_FAILED', messageOf(err), { cause: err });
}

/** For the catch of a runner's work promise. A RunntimeError passes
 *  through with its code, anything else (a lost device, a failed readback)
 *  becomes EXECUTION_FAILED with the original error as cause. */
export function rethrowRunError(err: unknown): never {
  if (isRunntimeError(err)) throw err;
  throw new RunntimeError('EXECUTION_FAILED', messageOf(err), { cause: err });
}
