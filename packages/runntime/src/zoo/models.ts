/** Ready-made model configs: the Hugging Face URLs of the weights and the
 *  tokenizer, plus the options each variant needs. Pass one to a
 *  create<Task>() call, or spread it and change a field.
 *
 *  Every family has a DEFAULT, the smallest download that runs everywhere,
 *  and one entry per hosted precision or size. The files live in the
 *  software-mansion `runntime-*` repos, one per model, laid out as
 *  `<size>/<precision>/model.safetensors` with `tokenizer.json` at the
 *  root where the model has one. */

import type { SpeechToTextModel } from './tasks/audio/speechToText.ts';
import type { ImageClassifierModel } from './tasks/cv/imageClassification.ts';
import type { DepthEstimatorModel } from './tasks/cv/depthEstimation.ts';
import type { InstanceSegmenterModel } from './tasks/cv/instanceSegmentation.ts';
import type { ObjectDetectorModel } from './tasks/cv/objectDetection.ts';
import type { KeypointDetectorModel } from './tasks/cv/keypointDetection.ts';
import type { EmbedderModel } from './tasks/multimodal/embedding.ts';
import type { PrivacyFilterModel } from './tasks/nlp/privacyFilter.ts';
import type { TextGeneratorModel } from './tasks/nlp/textGeneration.ts';
import type { DecoderConfig } from './decoder/config.ts';
import { GEMMA3_1B, GEMMA3_270M } from './gemma3/config.ts';
import { QWEN3_0_6B } from './qwen3/config.ts';

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
} satisfies EmbedderModel);

/** EmbeddingGemma 2 ships the text model and the vision tower as two files
 *  of each precision. */
const embeddingGemma2 = (precision: string) => ({
  modelPath: weights('embeddinggemma-2', 'text', precision),
  visionPath: weights('embeddinggemma-2', 'vision', precision),
  audioPath: weights('embeddinggemma-2', 'audio', precision),
  tokenizerPath: tokenizer('embeddinggemma-2'),
});
const EMBEDDINGGEMMA_2 = {
  DEFAULT: embeddingGemma2('f16'),
  F16: { ...embeddingGemma2('f16'), dtype: 'f16' as const },
  F32: { ...embeddingGemma2('f32'), dtype: 'f32' as const },
} satisfies Record<string, EmbedderModel>;

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
const YOLO26_S = { modelPath: weights('yolo26', 's', 'f16') } satisfies ObjectDetectorModel;
const YOLO26_M = { modelPath: weights('yolo26', 'm', 'f16') } satisfies ObjectDetectorModel;

const MOBILENETV4_SMALL = {
  modelPath: weights('mobilenetv4', 'small', 'f16'),
} satisfies ImageClassifierModel;

const YOLO26_SEG_N = {
  modelPath: weights('yolo26-seg', 'n', 'f16'),
} satisfies InstanceSegmenterModel;
const YOLO26_SEG_S = {
  modelPath: weights('yolo26-seg', 's', 'f16'),
} satisfies InstanceSegmenterModel;
const YOLO26_SEG_M = {
  modelPath: weights('yolo26-seg', 'm', 'f16'),
} satisfies InstanceSegmenterModel;

const YOLO26_POSE_N = {
  modelPath: weights('yolo26-pose', 'n', 'f16'),
} satisfies KeypointDetectorModel;
const YOLO26_POSE_S = {
  modelPath: weights('yolo26-pose', 's', 'f16'),
} satisfies KeypointDetectorModel;
const YOLO26_POSE_M = {
  modelPath: weights('yolo26-pose', 'm', 'f16'),
} satisfies KeypointDetectorModel;

/** Gemma 3 instruction-tuned checkpoints, the ungated unsloth mirrors of
 *  google/gemma-3-*-it, pinned to a revision so cached bytes stay valid. */
const gemma3 = (name: string, revision: string, config: DecoderConfig): TextGeneratorModel => {
  const root = `https://huggingface.co/unsloth/${name}/resolve/${revision}`;
  return {
    modelPath: `${root}/model.safetensors`,
    tokenizerPath: `${root}/tokenizer.json`,
    config,
    cacheId: `gemma3-v1/${name}@${revision.slice(0, 12)}`,
  };
};
const GEMMA3_1B_IT = gemma3('gemma-3-1b-it', '5b11413a10db4e486ef16a20101fd028f8f2499c', GEMMA3_1B);
const GEMMA3_270M_IT = gemma3(
  'gemma-3-270m-it',
  '23cf460f6bb16954176b3ddcc8d4f250501458a9',
  GEMMA3_270M,
);

/** Qwen 3 checkpoints from the (ungated) Qwen org, pinned to a revision. */
const qwen3 = (name: string, revision: string, config: DecoderConfig): TextGeneratorModel => {
  const root = `https://huggingface.co/Qwen/${name}/resolve/${revision}`;
  return {
    modelPath: `${root}/model.safetensors`,
    tokenizerPath: `${root}/tokenizer.json`,
    config,
    cacheId: `qwen3-v1/${name}@${revision.slice(0, 12)}`,
  };
};
const QWEN3_0_6B_MODEL = qwen3(
  'Qwen3-0.6B',
  'c1899de289a04d12100db370d81485cdf75e47ca',
  QWEN3_0_6B,
);

const DEPTHART_S = { modelPath: weights('depthart', 's', 'f16') } satisfies DepthEstimatorModel;
const DEPTHART_B = { modelPath: weights('depthart', 'b', 'f16') } satisfies DepthEstimatorModel;

export const models = {
  embedding: {
    /** Google's EmbeddingGemma 2: text in 100+ languages and code, images
     *  and video in one 768-dim space, inputs up to 2048 tokens. 270M
     *  parameters in the text model, 170M in the vision tower. */
    EMBEDDINGGEMMA_2: EMBEDDINGGEMMA_2,
    /** all-MiniLM-L6-v2: English text in 384-dim vectors, 22M parameters. */
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
      /** The small size, 18 MB. */
      S: YOLO26_S,
      /** The medium size, 39 MB. More accurate, slower. */
      M: YOLO26_M,
    },
  },
  instanceSegmentation: {
    /** YOLO26 segment, COCO-80 classes with a mask per object, f16 weights. */
    YOLO26_SEG: {
      DEFAULT: YOLO26_SEG_N,
      /** The nano size, 5 MB. */
      N: YOLO26_SEG_N,
      /** The small size, 20 MB. */
      S: YOLO26_SEG_S,
      /** The medium size, 45 MB. More accurate, slower. */
      M: YOLO26_SEG_M,
    },
  },
  keypointDetection: {
    /** YOLO26 pose, people with 17 COCO landmarks, f16 weights. */
    YOLO26_POSE: {
      DEFAULT: YOLO26_POSE_N,
      /** The nano size, 6 MB. */
      N: YOLO26_POSE_N,
      /** The small size, 20 MB. */
      S: YOLO26_POSE_S,
      /** The medium size, 42 MB. More accurate, slower. */
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
  textGeneration: {
    /** Gemma 3 1B instruction-tuned, bf16 weights (2.0 GB download). */
    GEMMA3_1B_IT,
    /** Gemma 3 270M instruction-tuned, bf16 weights (536 MB download). */
    GEMMA3_270M_IT,
    /** Qwen 3 0.6B (hybrid thinking, tool calling), bf16 weights (1.5 GB
     *  download). Thinking is off unless `chatTemplate.enableThinking`. */
    QWEN3_0_6B: QWEN3_0_6B_MODEL,
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
