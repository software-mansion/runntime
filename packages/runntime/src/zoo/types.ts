/** The public types of runntime/zoo. Copied from the library so apps/docs
 *  typechecks against the real shape while the implementation lives
 *  elsewhere. Types only — no inference code. */

import type { RangeSource, WeightCache } from '../core/index.ts';

/* ---------------------------------------------------------------- loading */

/** Where the weights come from: a URL, or any byte-range reader. */
export type ModelPath = string | RangeSource;

/** Options accepted by every create<Task>() factory. */
export interface LoadOptions {
  /** Saves fetched bytes. The next load reads them from here, no network. */
  cache?: WeightCache;
  /** Cache key. Defaults to the URL's file name; required for a RangeSource. */
  cacheId?: string;
  /** Called while tensors upload: tensor name, bytes done, bytes total. */
  onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  /** Called per fetched network chunk with its size in bytes. */
  onBytes?: (chunkBytes: number) => void;
  /** Stops the load between steps. A step already running finishes first. */
  signal?: AbortSignal;
}

/* ------------------------------------------------------------- geometry */

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface Size {
  readonly width: number;
  readonly height: number;
}

export type BoxMap = Readonly<{
  xyxy: Readonly<{ xmin: number; ymin: number; xmax: number; ymax: number }>;
  xywh: Readonly<{ xmin: number; ymin: number; w: number; h: number }>;
  cxcywh: Readonly<{ cx: number; cy: number; w: number; h: number }>;
}>;

export type BoxFormat = keyof BoxMap;

/** A box tagged with its format. */
export type BoundingBox<F extends BoxFormat = BoxFormat> = F extends BoxFormat
  ? { readonly format: F } & BoxMap[F]
  : never;

/* ---------------------------------------------------------------- images */

/** Pixel formats an ImageBuffer can carry: which channels, in what order. */
export type ImageFormat = 'rgb' | 'rgba' | 'bgr' | 'bgra' | 'gray';

/** A raw image in memory: one row after another, all channels of one pixel
 *  together (the way a canvas or a camera gives it). */
export interface ImageBuffer {
  readonly data: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  readonly format: ImageFormat;
  readonly layout: 'hwc';
}

/** How an image is fitted into the model's input size.
 *  - `stretch`: scale width and height separately; nothing is cut or padded.
 *  - `letterbox`: keep the aspect ratio, pad the rest with a flat color.
 *  - `crop`: keep the aspect ratio, cut the centered square, then scale. */
export type ResizeMode = 'stretch' | 'letterbox' | 'crop';

/* ----------------------------------------------------------- model sizes */

/** YOLO26 model size, nano to extra-large. */
export type Yolo26Variant = 'n' | 's' | 'm' | 'l' | 'x';

/** DepthART model size. */
export type DepthartVariant = 's' | 'b';

/* --------------------------------------------------------- text embedding */

export interface TextEmbedderModel {
  readonly modelPath?: ModelPath;
  readonly tokenizerPath?: string;
  readonly dtype?: 'f32' | 'f16';
  readonly maxTokens?: number;
}

export interface TextEmbedder {
  /** Length of every returned vector. */
  readonly dim: number;
  /** One text → unit-length vector. */
  embed(input: string): Promise<Float32Array>;
  /** Many texts in one GPU pass. Much faster than embed() in a loop. */
  embedBatch(inputs: readonly string[]): Promise<Float32Array[]>;
  dispose(): void;
}

/* ---------------------------------------------------------- privacy filter */

export interface PrivacyFilterModel {
  readonly modelPath?: ModelPath;
  readonly tokenizerPath?: string;
}

export interface PrivacySpan {
  readonly label: string;
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

export interface PrivacyFilter {
  findSpans(input: string): Promise<PrivacySpan[]>;
  redact(input: string): Promise<string>;
  dispose(): void;
}

/* --------------------------------------------------------- speech to text */

/** The two Moonshine families. Each has its own weights and decoder. */
export type SpeechToTextArch = 'moonshine' | 'moonshine-streaming';

export interface SpeechToTextModel {
  readonly arch?: SpeechToTextArch;
  readonly modelPath?: ModelPath;
  readonly tokenizerPath?: string;
  readonly dtype?: 'f32' | 'f16';
}

/** One update of a live stream: the whole transcript so far. */
export interface SpeechStreamUpdate {
  /** The sentences the speaker finished. Each ended in a pause. */
  readonly committed: string;
  /** The sentence being spoken now. It changes until it commits. */
  readonly nonCommitted: string;
}

export interface SpeechToText {
  /** Mono samples at 16 kHz in, the spoken text out. */
  transcribe(audio: Float32Array): Promise<string>;
  /** Starts live transcription. One stream at a time. */
  stream(): AsyncIterable<SpeechStreamUpdate>;
  /** Adds microphone samples, mono at 16 kHz, to the open stream. */
  streamInsert(samples: Float32Array): void;
  /** Commits what is left in the open stream and ends its iterator. */
  streamStop(): void;
  dispose(): void;
}

/* ------------------------------------------------------- object detection */

export interface ObjectDetectorModel {
  readonly modelPath?: ModelPath;
  readonly variant?: Yolo26Variant;
  readonly inputSize?: number;
  readonly resizeMode?: ResizeMode;
}

export interface DetectObjectsOptions {
  /** Drops objects scored below this, 0 to 1. Default 0.3. */
  readonly confidenceThreshold?: number;
  /** At most this many objects, best first. Default 300. */
  readonly maxDetections?: number;
}

/** One object found in the image. */
export interface ObjectDetection {
  readonly label: string;
  readonly classId: number;
  readonly confidence: number;
  readonly box: BoundingBox<'xyxy'>;
}

export interface ObjectDetector {
  readonly labels: readonly string[];
  detectObjects(image: ImageBuffer, options?: DetectObjectsOptions): Promise<ObjectDetection[]>;
  dispose(): void;
}

/* --------------------------------------------------- instance segmentation */

export interface InstanceSegmenterModel {
  readonly modelPath?: ModelPath;
  readonly variant?: Yolo26Variant;
  readonly inputSize?: number;
  readonly resizeMode?: ResizeMode;
}

export interface SegmentInstancesOptions {
  readonly confidenceThreshold?: number;
  readonly maxDetections?: number;
}

export interface InstanceSegmentation {
  readonly label: string;
  readonly classId: number;
  readonly confidence: number;
  readonly box: BoundingBox<'xyxy'>;
  /** Which pixels of `box` are the object: a `gray` image covering the box,
   *  at about a quarter of the model input's resolution. */
  readonly mask: ImageBuffer;
}

export interface InstanceSegmenter {
  readonly labels: readonly string[];
  segmentInstances(
    image: ImageBuffer,
    options?: SegmentInstancesOptions,
  ): Promise<InstanceSegmentation[]>;
  dispose(): void;
}

/* ------------------------------------------------------ keypoint detection */

export const COCO_LANDMARKS = [
  'nose',
  'leftEye',
  'rightEye',
  'leftEar',
  'rightEar',
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
] as const;

export type CocoLandmark = (typeof COCO_LANDMARKS)[number];

export interface KeypointDetectorModel {
  readonly modelPath?: ModelPath;
  readonly variant?: Yolo26Variant;
  readonly inputSize?: number;
  readonly resizeMode?: ResizeMode;
}

export interface DetectKeypointsOptions {
  readonly confidenceThreshold?: number;
  readonly maxDetections?: number;
}

/** One landmark of a person, in pixels of the input image. */
export interface Landmark extends Point {
  /** How sure the model is the landmark is visible, 0 to 1. */
  readonly confidence: number;
}

/** One person found in the image. */
export interface KeypointDetection {
  readonly box: BoundingBox<'xyxy'>;
  readonly confidence: number;
  /** The 17 COCO landmarks by name, `nose` to `rightAnkle`. */
  readonly landmarks: Readonly<Record<CocoLandmark, Landmark>>;
}

export interface KeypointDetector {
  readonly landmarkNames: readonly CocoLandmark[];
  /** Pairs of landmark names to draw as limbs. */
  readonly skeleton: readonly (readonly [CocoLandmark, CocoLandmark])[];
  detectKeypoints(
    image: ImageBuffer,
    options?: DetectKeypointsOptions,
  ): Promise<KeypointDetection[]>;
  dispose(): void;
}

/* -------------------------------------------------------- depth estimation */

export interface DepthEstimatorModel {
  readonly modelPath?: ModelPath;
  readonly variant?: DepthartVariant;
  readonly resizeMode?: ResizeMode;
}

/** Relative depth of the model input, one value per pixel, row by row.
 *  Smaller = nearer. Values have no unit: compare them within one map. */
export interface DepthMap {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

export interface DepthEstimator {
  /** Width and height of every returned map. */
  readonly inputSize: number;
  estimateDepth(image: ImageBuffer): Promise<DepthMap>;
  dispose(): void;
}

/* ---------------------------------------------------- image classification */

export interface ImageClassifierModel {
  readonly modelPath?: ModelPath;
  readonly labels?: readonly string[];
  readonly resizeMode?: ResizeMode;
}

export interface ClassifyOptions {
  /** How many classes to return, best first. Default: all of them. */
  readonly topk?: number;
}

export interface Classification {
  readonly classId: number;
  readonly confidence: number;
}

export interface ImageClassifier {
  readonly labels: readonly string[];
  /** Width and height of the model input in pixels. */
  readonly inputSize: number;
  classify(
    image: ImageBuffer,
    options?: ClassifyOptions,
  ): Promise<(Classification & { label: string })[]>;
  dispose(): void;
}
