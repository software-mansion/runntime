import { createOpfsCache, type LoadOptions } from 'runntime/zoo';
import { useCallback, useState } from 'react';
import { initEngine } from './engine.ts';

export type ModelState<T> =
  | { status: 'idle' }
  | { status: 'loading'; percent: number }
  | { status: 'ready'; model: T }
  | { status: 'error'; message: string };

/** Loads one model on demand: sets up the engine, hands the factory a cache
 *  and a progress callback, and reports how far the download got so the
 *  button can show a bar. The weights land in OPFS, so a reload skips the
 *  network. */
export function useModel<T>(load: (opts: LoadOptions) => Promise<T>): {
  state: ModelState<T>;
  start: () => void;
} {
  const [state, setState] = useState<ModelState<T>>({ status: 'idle' });

  const start = useCallback(() => {
    setState({ status: 'loading', percent: 0 });
    void (async () => {
      try {
        await initEngine();
        const model = await load({
          cache: await createOpfsCache('runntime-examples'),
          onProgress: (_name, doneBytes, totalBytes) =>
            setState({ status: 'loading', percent: totalBytes ? doneBytes / totalBytes : 0 }),
        });
        setState({ status: 'ready', model });
      } catch (err) {
        setState({ status: 'error', message: (err as Error).message });
      }
    })();
  }, [load]);

  return { state, start };
}
