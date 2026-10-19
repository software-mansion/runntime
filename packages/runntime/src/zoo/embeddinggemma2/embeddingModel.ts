/** EmbeddingGemma 2 for createEmbedder: text, images, video and audio,
 *  task instructions, 768 values or fewer with `dims`. */

import { createResourceScope, RunntimeError } from '../../core/index.ts';
import type {
  EmbedInput,
  EmbedPart,
  EmbeddingModel,
  EmbeddingModelLoad,
  EmbeddingTask,
} from '../tasks/multimodal/embedding.ts';
import { imageBuffer, type ImageBuffer } from '../tasks/cv/image.ts';
import { audioTokens, encodeAudio } from './audioEncoder.ts';
import { EmbeddingGemma2AudioModel } from './audioModel.ts';
import {
  EMBEDDING_DIM,
  EMBEDDING_DIMS,
  EMBEDDINGGEMMA2_MAX_TOKENS,
  EMBEDDINGGEMMA2_PROMPTS,
  EMBEDDINGGEMMA2_VISION,
  IMAGE_TOKEN_BUDGETS,
  IMAGE_TOKENS,
  VIDEO_FRAME_TOKENS,
} from './config.ts';
import { createEmbeddingGemma2EmbedCore, type EmbeddingGemma2Input } from './embedCore.ts';
import { EmbeddingGemma2Model } from './model.ts';
import { createEmbeddingGemma2Tokenizer } from './tokenizer.ts';
import { encodeImage, patchGrid } from './visionEncoder.ts';
import { EmbeddingGemma2VisionModel } from './visionModel.ts';

export async function createEmbeddingModel(load: EmbeddingModelLoad): Promise<EmbeddingModel> {
  const { maxTokens = EMBEDDINGGEMMA2_MAX_TOKENS, imageTokens = IMAGE_TOKENS } = load;
  if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > EMBEDDINGGEMMA2_MAX_TOKENS) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `createEmbedder: maxTokens must be an integer from 2 to ${EMBEDDINGGEMMA2_MAX_TOKENS}, got ${maxTokens}`,
    );
  }
  const budgets: readonly number[] = IMAGE_TOKEN_BUDGETS;
  if (!budgets.includes(imageTokens)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `createEmbedder: imageTokens must be one of ${budgets.join(', ')}, got ${imageTokens}`,
    );
  }
  const tokenizer = createEmbeddingGemma2Tokenizer(load.tokenizer);

  const scope = createResourceScope();
  try {
    const model = scope.track(new EmbeddingGemma2Model());
    const vision = load.vision && scope.track(new EmbeddingGemma2VisionModel());
    const audio = load.audio && scope.track(new EmbeddingGemma2AudioModel());
    if (load.dtype === 'f16') {
      model.half();
      vision?.half();
      audio?.half();
    }
    // The files load at once; progress counts their bytes together.
    const progress = new Map<string, [number, number]>();
    const track = (file: string) => (name: string, done: number, total: number) => {
      progress.set(file, [done, total]);
      let d = 0;
      let t = 0;
      for (const [fd, ft] of progress.values()) {
        d += fd;
        t += ft;
      }
      load.onProgress?.(name, d, t);
    };
    const loads = [model.loadStateDict(load.text, { onProgress: track('text') })];
    if (vision && load.vision) {
      loads.push(vision.loadStateDict(load.vision, { onProgress: track('vision') }));
    }
    if (audio && load.audio) {
      loads.push(audio.loadStateDict(load.audio, { onProgress: track('audio') }));
    }
    await Promise.all(loads);
    const core = scope.track(createEmbeddingGemma2EmbedCore(model));
    const { bosId, eosId, padId, boiId, eoiId, boaId, eoaId } = tokenizer;
    const { patch, pool } = EMBEDDINGGEMMA2_VISION;

    /** An input as token ids and media rows: <bos>, the instruction and the
     *  parts in order, <eos>. Text next to other text joins it, each image
     *  or video frame is <|image> rows <image|>, and each clip <|audio> rows
     *  <audio|>. The length is checked before any tower runs. */
    const toInput = async (
      where: string,
      input: EmbedInput,
      task: EmbeddingTask | undefined,
    ): Promise<EmbeddingGemma2Input> => {
      // Array.isArray does not narrow a readonly array.
      const parts = (Array.isArray(input) ? input : [input]) as readonly EmbedPart[];
      const ids = [bosId];
      // Each media part's first row and how to compute its rows.
      const pending: { at: number; encode: () => Promise<Float32Array> }[] = [];
      let text = task === undefined ? '' : EMBEDDINGGEMMA2_PROMPTS[task];
      const flush = () => {
        ids.push(...tokenizer.encode(text));
        text = '';
      };
      const addMedia = (
        begin: number,
        tokens: number,
        end: number,
        encode: () => Promise<Float32Array>,
      ) => {
        flush();
        ids.push(begin);
        pending.push({ at: ids.length, encode });
        for (let i = 0; i < tokens; i++) ids.push(padId);
        ids.push(end);
      };
      const addImage = (image: ImageBuffer, budget: number) => {
        const [cols, rows] = patchGrid(image.width, image.height, budget, patch, pool);
        addMedia(boiId, (cols * rows) / (pool * pool), eoiId, () => {
          if (!vision) {
            throw new RunntimeError(
              'INVALID_ARGUMENT',
              `${where}: images and video need the vision tower, which this embedder did not load`,
            );
          }
          return encodeImage(vision, image, budget);
        });
      };
      for (const part of parts) {
        if (typeof part === 'string') text += part;
        else if ('image' in part) addImage(part.image, imageTokens);
        else if ('video' in part) {
          for (const frame of part.video) addImage(frame, VIDEO_FRAME_TOKENS);
        } else {
          const samples = part.audio;
          addMedia(boaId, audioTokens(samples.length), eoaId, () => {
            if (!audio) {
              throw new RunntimeError(
                'INVALID_ARGUMENT',
                `${where}: audio needs the audio tower, which this embedder did not load`,
              );
            }
            return encodeAudio(audio, samples);
          });
        }
      }
      flush();
      ids.push(eosId);
      if (ids.length > maxTokens) {
        if (pending.length === 0) {
          return { ids: [...ids.slice(0, maxTokens - 1), eosId], media: [] };
        }
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: the input takes ${ids.length} tokens, more than maxTokens ${maxTokens}`,
        );
      }
      const media: { at: number; rows: Float32Array }[] = [];
      for (const { at, encode } of pending) media.push({ at, rows: await encode() });
      return { ids, media };
    };

    return {
      dim: EMBEDDING_DIM,
      dims: EMBEDDING_DIMS,
      media: true,
      async warm() {
        // The empty text is just <bos> and <eos>.
        const probe = await toInput('createEmbedder', '', undefined);
        await core.embedPooledBatch([probe, probe]);
        await core.warmBuckets(probe);
        // A 1x1 image grows to the budget. The image and the video frame
        // budgets each run once, as their sizes may take different attention
        // paths.
        if (vision) {
          const dot = imageBuffer(new Uint8Array(4), 1, 1);
          await encodeImage(vision, dot, imageTokens);
          if (imageTokens !== VIDEO_FRAME_TOKENS) {
            await encodeImage(vision, dot, VIDEO_FRAME_TOKENS);
          }
        }
        // A tenth of a second of silence runs every audio kernel.
        if (audio) await encodeAudio(audio, new Float32Array(1600));
        return core.embedPooled(probe);
      },
      async pool(where, inputs, task) {
        const prepared: EmbeddingGemma2Input[] = [];
        for (const input of inputs) prepared.push(await toInput(where, input, task));
        // A batch of one is a single input: take the replay path.
        return prepared.length === 1
          ? core.embedPooled(prepared[0]!)
          : core.embedPooledBatch(prepared);
      },
      dispose: () => scope.dispose(),
    };
  } catch (err) {
    scope.dispose();
    throw err;
  }
}
