/** API-surface stub of runntime/zoo. The types and the model registry are
 *  the real ones; every create<Task>() and helper throws. Here so apps/docs
 *  builds and typechecks before the library lands in this repo. */

export { bufferSource, createOpfsCache, initRunntime } from '../core/index.ts';
export type { RangeSource, WeightCache } from '../core/index.ts';

export { models } from './models.ts';
export { COCO_LANDMARKS } from './types.ts';
export type * from './types.ts';

import type {
  DepthEstimator,
  DepthEstimatorModel,
  ImageBuffer,
  ImageClassifier,
  ImageClassifierModel,
  ImageFormat,
  InstanceSegmenter,
  InstanceSegmenterModel,
  KeypointDetector,
  KeypointDetectorModel,
  LoadOptions,
  ObjectDetector,
  ObjectDetectorModel,
  PrivacyFilter,
  PrivacyFilterModel,
  SpeechToText,
  SpeechToTextModel,
  TextEmbedder,
  TextEmbedderModel,
} from './types.ts';

/** Sample rate every speech model expects, in Hz. */
export const SPEECH_SAMPLE_RATE = 16_000;

const notImplemented = (name: string): never => {
  throw new Error(
    `runntime: ${name}() is a stub in this repo. The library is not published yet.`,
  );
};

/* ------------------------------------------------------------- task runners */

export function createTextEmbedder(
  _config: TextEmbedderModel = {},
  _opts: LoadOptions = {},
): Promise<TextEmbedder> {
  return notImplemented('createTextEmbedder');
}

export function createPrivacyFilter(
  _config: PrivacyFilterModel = {},
  _opts: LoadOptions = {},
): Promise<PrivacyFilter> {
  return notImplemented('createPrivacyFilter');
}

export function createSpeechToText(
  _config: SpeechToTextModel = {},
  _opts: LoadOptions = {},
): Promise<SpeechToText> {
  return notImplemented('createSpeechToText');
}

export function createObjectDetector(
  _config: ObjectDetectorModel = {},
  _opts: LoadOptions = {},
): Promise<ObjectDetector> {
  return notImplemented('createObjectDetector');
}

export function createInstanceSegmenter(
  _config: InstanceSegmenterModel = {},
  _opts: LoadOptions = {},
): Promise<InstanceSegmenter> {
  return notImplemented('createInstanceSegmenter');
}

export function createKeypointDetector(
  _config: KeypointDetectorModel = {},
  _opts: LoadOptions = {},
): Promise<KeypointDetector> {
  return notImplemented('createKeypointDetector');
}

export function createDepthEstimator(
  _config: DepthEstimatorModel = {},
  _opts: LoadOptions = {},
): Promise<DepthEstimator> {
  return notImplemented('createDepthEstimator');
}

export function createImageClassifier(
  _config: ImageClassifierModel = {},
  _opts: LoadOptions = {},
): Promise<ImageClassifier> {
  return notImplemented('createImageClassifier');
}

/* ------------------------------------------------------------------ helpers */

/** Dot product of two unit-length embeddings, which is their cosine
 *  similarity: 1 for the same meaning, near 0 for unrelated text. */
export function similarity(_a: Float32Array, _b: Float32Array): number {
  return notImplemented('similarity');
}

/** Decodes an audio file to mono samples at 16 kHz. */
export function decodeAudio(_bytes: ArrayBuffer | Uint8Array): Promise<Float32Array> {
  return notImplemented('decodeAudio');
}

/** Resamples mono PCM from `fromRate` to 16 kHz. */
export function resampleAudio(_samples: Float32Array, _fromRate: number): Promise<Float32Array> {
  return notImplemented('resampleAudio');
}

/** Wraps raw pixel bytes as an ImageBuffer. */
export function imageBuffer(
  _data: Uint8Array | Uint8ClampedArray,
  _width: number,
  _height: number,
  _format?: ImageFormat,
): ImageBuffer {
  return notImplemented('imageBuffer');
}

/** Wraps a canvas ImageData as an rgba ImageBuffer. */
export function imageBufferFromImageData(_img: ImageData): ImageBuffer {
  return notImplemented('imageBufferFromImageData');
}
