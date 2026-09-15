/** Ready-made model configs: the Hugging Face URLs of the weights and the
 *  tokenizer, plus the options each variant needs. Pass one to a
 *  create<Task>() call, or spread it and change a field.
 *
 *  Every family has a DEFAULT, the smallest download that runs everywhere,
 *  and one entry per hosted precision or size. The files live in the
 *  software-mansion `runntime-*` repos, one per model, laid out as
 *  `<size>/<precision>/model.safetensors` with `tokenizer.json` at the
 *  root where the model has one. */

import type {
  DepthEstimatorModel,
  ImageClassifierModel,
  InstanceSegmenterModel,
  KeypointDetectorModel,
  ObjectDetectorModel,
  PrivacyFilterModel,
  SpeechToTextModel,
  TextEmbedderModel,
} from './types.ts';

const HUB = 'https://huggingface.co/software-mansion';

const weights = (repo: string, size: string, precision: string) =>
  `${HUB}/runntime-${repo}/resolve/main/${size}/${precision}/model.safetensors`;
const tokenizer = (repo: string) => `${HUB}/runntime-${repo}/resolve/main/tokenizer.json`;

/** DEFAULT, F16 and F32 entries of one size of a model that ships both
 *  precisions. DEFAULT is the f16 file with no dtype set: half the download,
 *  and the task runs it in f16 where the device has shader-f16 and in f32
 *  elsewhere. F16 forces f16 math, F32 is the f32 file in f32. */
const precisions = <T extends object>(repo: string, size: string, common: T) => {
  const f16 = { ...common, modelPath: weights(repo, size, 'f16') };
  return {
    DEFAULT: f16,
    F16: { ...f16, dtype: 'f16' as const },
    F32: { ...common, modelPath: weights(repo, size, 'f32'), dtype: 'f32' as const },
  };
};

const MINILM = precisions('all-MiniLM-L6-v2', 'base', {
  tokenizerPath: tokenizer('all-MiniLM-L6-v2'),
} satisfies TextEmbedderModel);

const MOONSHINE_TINY = precisions('moonshine', 'tiny', {
  arch: 'moonshine',
  tokenizerPath: tokenizer('moonshine'),
} satisfies SpeechToTextModel);
const MOONSHINE_BASE = precisions('moonshine', 'base', {
  arch: 'moonshine',
  tokenizerPath: tokenizer('moonshine'),
} satisfies SpeechToTextModel);
const MOONSHINE_STREAMING_TINY = precisions('moonshine-streaming', 'tiny', {
  arch: 'moonshine-streaming',
  tokenizerPath: tokenizer('moonshine-streaming'),
} satisfies SpeechToTextModel);

const PRIVACY_FILTER_INT8 = {
  modelPath: weights('privacy-filter', 'base', 'int8g640'),
} satisfies PrivacyFilterModel;

const YOLO26_N = { modelPath: weights('yolo26', 'n', 'f16') } satisfies ObjectDetectorModel;
const YOLO26_M = { modelPath: weights('yolo26', 'm', 'f16') } satisfies ObjectDetectorModel;

const MOBILENETV4_SMALL = {
  modelPath: weights('mobilenetv4', 'small', 'f16'),
} satisfies ImageClassifierModel;

const YOLO26_SEG_N = {
  modelPath: weights('yolo26-seg', 'n', 'f16'),
} satisfies InstanceSegmenterModel;
const YOLO26_SEG_M = {
  modelPath: weights('yolo26-seg', 'm', 'f16'),
} satisfies InstanceSegmenterModel;

const YOLO26_POSE_N = {
  modelPath: weights('yolo26-pose', 'n', 'f16'),
} satisfies KeypointDetectorModel;
const YOLO26_POSE_M = {
  modelPath: weights('yolo26-pose', 'm', 'f16'),
} satisfies KeypointDetectorModel;

const DEPTHART_S = { modelPath: weights('depthart', 's', 'f16') } satisfies DepthEstimatorModel;
const DEPTHART_B = { modelPath: weights('depthart', 'b', 'f16') } satisfies DepthEstimatorModel;

export const models = {
  textEmbedding: {
    /** 384-dim sentence embeddings, 22M parameters. */
    ALL_MINILM_L6_V2: MINILM,
  },
  privacyFilter: {
    /** OpenAI's privacy-filter: 1.5B parameters, 50M active per token,
     *  int8 weights in groups of 640. The one hosted format. */
    PRIVACY_FILTER: { DEFAULT: PRIVACY_FILTER_INT8 },
  },
  speechToText: {
    /** English. */
    MOONSHINE: {
      /** 27M parameters. */
      TINY: MOONSHINE_TINY,
      /** 61M parameters. More accurate, slower. */
      BASE: MOONSHINE_BASE,
    },
    /** English, the streaming family. */
    MOONSHINE_STREAMING: {
      /** 34M parameters. */
      TINY: MOONSHINE_STREAMING_TINY,
    },
  },
  objectDetection: {
    /** YOLO26 detect, COCO-80 classes, f16 weights. */
    YOLO26: {
      DEFAULT: YOLO26_N,
      /** The nano size, 5 MB. */
      N: YOLO26_N,
      /** The medium size, 41 MB. More accurate, slower. */
      M: YOLO26_M,
    },
  },
  instanceSegmentation: {
    /** YOLO26 segment, COCO-80 classes with a mask per object, f16 weights. */
    YOLO26_SEG: {
      DEFAULT: YOLO26_SEG_N,
      /** The nano size, 6 MB. */
      N: YOLO26_SEG_N,
      /** The medium size, 48 MB. More accurate, slower. */
      M: YOLO26_SEG_M,
    },
  },
  keypointDetection: {
    /** YOLO26 pose, people with 17 COCO landmarks, f16 weights. */
    YOLO26_POSE: {
      DEFAULT: YOLO26_POSE_N,
      /** The nano size, 6 MB. */
      N: YOLO26_POSE_N,
      /** The medium size, 44 MB. More accurate, slower. */
      M: YOLO26_POSE_M,
    },
  },
  depthEstimation: {
    /** DepthART relative depth at 448×448, f16 weights. */
    DEPTHART: {
      DEFAULT: DEPTHART_S,
      /** The small size, 13 MB. */
      S: DEPTHART_S,
      /** The base size, 24 MB. More detail, slower. */
      B: DEPTHART_B,
    },
  },
  imageClassification: {
    /** MobileNetV4 conv, ImageNet-1k, f16 weights. The model runs in f16
     *  only, so each size has one entry. */
    MOBILENETV4: {
      DEFAULT: MOBILENETV4_SMALL,
      /** The small size, 3.8M parameters, 10 MB. */
      SMALL: MOBILENETV4_SMALL,
    },
  },
} as const;
