/** Loads yolo26 weights from an unfused safetensors export onto the
 *  initRunntime() device; the blocks declare their own BN folding. */
import { fromSafetensors, type WeightCache } from '../../core/index.ts';
import type { Yolo26Model, Yolo26Task } from './model.ts';

export async function loadYolo26Weights(
  model: Yolo26Model<Yolo26Task>,
  baseUrl: string,
  opts: {
    onBytes?: (chunkBytes: number) => void;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
    cache?: WeightCache;
  } = {},
): Promise<void> {
  const url = `${baseUrl}/model.safetensors`;
  // cacheId defaults to the URL basename, which is 'model.safetensors' for
  // every variant here — key by the full URL so variants never collide.
  const sd = await fromSafetensors(url, { onBytes: opts.onBytes, cache: opts.cache, cacheId: url });
  await model.loadStateDict(sd, { onProgress: opts.onProgress });
}
