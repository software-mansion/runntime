/** An error with a `code`. Apps check the code, not the message. The
 *  message is for people and can change between versions. */

export const RUNNTIME_ERROR_CODES = [
  /** Weights, a tokenizer or a config could not be fetched, read or uploaded. */
  'LOAD_FAILED',
  /** The load stopped because its AbortSignal fired. */
  'LOAD_ABORTED',
  /** The weights or config file does not match the model. */
  'CHECKPOINT_MISMATCH',
  /** The device lacks a feature the task needs. */
  'UNSUPPORTED_DEVICE',
  /** The GPU failed to run the work. */
  'EXECUTION_FAILED',
  /** A value passed by the caller is wrong. */
  'INVALID_ARGUMENT',
  /** The model was disposed and cannot run again. */
  'RESOURCE_DISPOSED',
  /** initRunntime(root) was not called, so there is no device. */
  'NOT_INITIALIZED',
] as const;

export type RunntimeErrorCode = (typeof RUNNTIME_ERROR_CODES)[number];

export class RunntimeError<C extends RunntimeErrorCode = RunntimeErrorCode> extends Error {
  readonly code: C;

  constructor(code: C, message: string, options?: ErrorOptions) {
    super(message, options);
    this.code = code;
  }
}
// Set on the prototype so stack traces say RunntimeError. A class field is
// set too late for that.
RunntimeError.prototype.name = 'RunntimeError';

/** True when `value` is a RunntimeError, with this `code` if one is given.
 *  Checks `name` and `code`, not the class, so it works even when a page
 *  loads two copies of the library. */
export function isRunntimeError<C extends RunntimeErrorCode>(
  value: unknown,
  code?: C,
): value is RunntimeError<C> {
  if (!(value instanceof Error) || value.name !== 'RunntimeError') return false;
  const c = (value as { code?: unknown }).code;
  if (typeof c !== 'string' || !(RUNNTIME_ERROR_CODES as readonly string[]).includes(c)) {
    return false;
  }
  return code === undefined || c === code;
}
