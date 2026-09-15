/** API-surface stub of runntime/zoo/transformers: the transformers.js
 *  backend adapter. Every function throws; the docs only quote these in
 *  code samples. */

const notImplemented = (name: string): never => {
  throw new Error(
    `runntime: ${name}() is a stub in this repo. The library is not published yet.`,
  );
};

/** Registers runntime as a transformers.js backend and initialises it. */
export function initRunntimeBackend(_options?: unknown): Promise<void> {
  return notImplemented('initRunntimeBackend');
}

/** Registers runntime as a transformers.js backend against an existing root. */
export function registerRunntimeBackend(_options?: unknown): void {
  return notImplemented('registerRunntimeBackend');
}

/** True when transformers.js would route this model id through runntime. */
export function isRunntimeModel(_modelId: string): boolean {
  return notImplemented('isRunntimeModel');
}

/** Loader for all-MiniLM-L6-v2 weights. */
export function minilmLoader(_options?: unknown): unknown {
  return notImplemented('minilmLoader');
}

/** Loader for YOLO26 weights. */
export function yolo26Loader(_options?: unknown): unknown {
  return notImplemented('yolo26Loader');
}
