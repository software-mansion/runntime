/** Load MobileNetV4 weights from an unfused timm safetensors file (the Hub
 *  checkpoint or tools/export_weights_mobilenetv4.py) via the core streaming
 *  loader onto the initRunntime() device. BN folding is declared by the blocks themselves
 *  (BatchNorm2d.transformStateDict); this is just transport + upload. */
import { fromSafetensors, type WeightCache } from '../../core/index.ts';
import type { MobileNetV4Model } from './model.ts';

export async function loadMobileNetV4Weights(
  model: MobileNetV4Model,
  baseUrl: string,
  opts: {
    onBytes?: (chunkBytes: number) => void;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
    cache?: WeightCache;
  } = {},
): Promise<void> {
  const url = `${baseUrl}/model.safetensors`;
  const sd = await fromSafetensors(url, { onBytes: opts.onBytes, cache: opts.cache, cacheId: url });
  await model.loadStateDict(sd, { onProgress: opts.onProgress });
}
