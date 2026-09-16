/** Moonshine v2 (streaming) architecture config: frozen TS presets
 *  (authoritative — model code only ever sees these) + a checkpoint-config
 *  parser used to validate that the exported config.json still matches the
 *  preset at load time. The v2 config.json nests the encoder under
 *  `encoder_config` and keeps decoder + shared keys top-level, with rope
 *  hyperparameters inside `rope_parameters`. Encoder and decoder dims are
 *  independent (they agree only for tiny), and the attention inner dim
 *  heads·headDim matches hidden_size only for tiny — never assume square. */

import type { LazyStateDict } from '../../core/index.ts';

export interface StreamingEncoderConfig {
  readonly hidden: number;
  readonly ffn: number;
  readonly layers: number;
  readonly heads: number; // MHA — kv heads == attention heads, asserted at parse
  readonly headDim: number;
  readonly windows: readonly (readonly [number, number])[];
}

export interface StreamingDecoderConfig {
  readonly hidden: number;
  readonly ffn: number;
  readonly layers: number;
  readonly heads: number;
  readonly headDim: number;
  readonly rotaryDim: number; // int(headDim · partial_rotary_factor), even
  readonly ropeTheta: number;
}

export interface MoonshineStreamingConfig {
  readonly enc: StreamingEncoderConfig;
  readonly dec: StreamingDecoderConfig;
  readonly frameLen: number;
  readonly vocab: number;
  readonly maxPositions: number;
  readonly bos: number;
  readonly eos: number;
}

const windows = (layers: number): readonly (readonly [number, number])[] =>
  Object.freeze(
    Array.from({ length: layers }, (_, i) =>
      i < 2 || i >= layers - 2 ? ([16, 4] as const) : ([16, 0] as const),
    ),
  );

export const MOONSHINE_STREAMING_TINY: MoonshineStreamingConfig = Object.freeze({
  enc: Object.freeze({
    hidden: 320,
    ffn: 1280,
    layers: 6,
    heads: 8,
    headDim: 40,
    windows: windows(6),
  }),
  dec: Object.freeze({
    hidden: 320,
    ffn: 1280,
    layers: 6,
    heads: 8,
    headDim: 40,
    rotaryDim: 32, // int(40 · 0.8)
    ropeTheta: 10000,
  }),
  frameLen: 80,
  vocab: 32768,
  maxPositions: 4096,
  bos: 1,
  eos: 2,
});

export const MOONSHINE_STREAMING_SMALL: MoonshineStreamingConfig = Object.freeze({
  enc: Object.freeze({
    hidden: 620,
    ffn: 2480,
    layers: 10,
    heads: 8,
    headDim: 64,
    windows: windows(10),
  }),
  dec: Object.freeze({
    hidden: 512,
    ffn: 2048,
    layers: 10,
    heads: 8,
    headDim: 64,
    rotaryDim: 32, // int(64 · 0.5)
    ropeTheta: 10000,
  }),
  frameLen: 80,
  vocab: 32768,
  maxPositions: 4096,
  bos: 1,
  eos: 2,
});

export const MOONSHINE_STREAMING_MEDIUM: MoonshineStreamingConfig = Object.freeze({
  enc: Object.freeze({
    hidden: 768,
    ffn: 3072,
    layers: 14,
    heads: 10,
    headDim: 64,
    windows: windows(14),
  }),
  dec: Object.freeze({
    hidden: 640,
    ffn: 2560,
    layers: 14,
    heads: 10,
    headDim: 64,
    rotaryDim: 32, // int(64 · 0.5)
    ropeTheta: 10000,
  }),
  frameLen: 80,
  vocab: 32768,
  maxPositions: 4096,
  bos: 1,
  eos: 2,
});

export function presetFromStateDict(sd: LazyStateDict): MoonshineStreamingConfig {
  const embedding = sd.tensors.get('model.decoder.embed_tokens.weight');
  const width = embedding?.shape[1];
  const presets = [MOONSHINE_STREAMING_TINY, MOONSHINE_STREAMING_SMALL, MOONSHINE_STREAMING_MEDIUM];
  for (const cfg of presets) if (cfg.dec.hidden === width) return cfg;
  throw new Error(
    embedding === undefined
      ? 'moonshine-streaming weights: no token embedding, not a Moonshine checkpoint'
      : `moonshine-streaming weights: embedding width ${width} matches no known size`,
  );
}

type Json = Record<string, unknown>;

function num(json: Json, key: string): number {
  const v = json[key];
  if (typeof v !== 'number') {
    throw new Error(`moonshine-streaming config: '${key}' missing or not a number`);
  }
  return v;
}

export function configFromCheckpoint(json: Json): MoonshineStreamingConfig {
  const encJson = json.encoder_config as Json | undefined;
  if (!encJson) throw new Error("moonshine-streaming config: 'encoder_config' missing");
  const ropeParams = (json.rope_parameters ?? {}) as Json;
  const ropeTheta = ropeParams.rope_theta as number | undefined;
  const partialRotary = ropeParams.partial_rotary_factor as number | undefined;
  if (typeof ropeTheta !== 'number' || typeof partialRotary !== 'number') {
    throw new Error(
      'moonshine-streaming config: rope_parameters.{rope_theta,partial_rotary_factor} missing',
    );
  }

  for (const [scope, j] of [
    ['decoder', json],
    ['encoder', encJson],
  ] as const) {
    if (j.attention_bias !== false) {
      throw new Error(
        `moonshine-streaming config: ${scope} attention_bias must be false — loader folds no biases`,
      );
    }
    if (num(j, 'num_key_value_heads') !== num(j, 'num_attention_heads')) {
      throw new Error(
        `moonshine-streaming config: ${scope} kv heads != attention heads — model assumes MHA`,
      );
    }
  }

  const decHeadDim = num(json, 'head_dim');
  const rotaryDim = Math.floor(decHeadDim * partialRotary); // HF: int(head_dim · factor)
  if (rotaryDim <= 0 || rotaryDim % 2 !== 0 || rotaryDim > decHeadDim) {
    throw new Error(
      `moonshine-streaming config: derived rotaryDim ${rotaryDim} invalid for headDim ${decHeadDim}`,
    );
  }

  const bos = num(json, 'bos_token_id');
  if (json.decoder_start_token_id !== bos) {
    throw new Error('moonshine-streaming config: decoder_start_token_id != bos_token_id');
  }

  const frameLen = Math.round((num(encJson, 'sample_rate') * num(encJson, 'frame_ms')) / 1000);
  const rawWindows = encJson.sliding_windows;
  if (
    !Array.isArray(rawWindows) ||
    rawWindows.length !== num(encJson, 'num_hidden_layers') ||
    !rawWindows.every((w) => Array.isArray(w) && w.length === 2)
  ) {
    throw new Error(
      'moonshine-streaming config: sliding_windows must hold one [left, right] pair per layer',
    );
  }

  return Object.freeze({
    enc: Object.freeze({
      hidden: num(encJson, 'hidden_size'),
      ffn: num(encJson, 'intermediate_size'),
      layers: num(encJson, 'num_hidden_layers'),
      heads: num(encJson, 'num_attention_heads'),
      headDim: num(encJson, 'head_dim'),
      windows: Object.freeze(
        rawWindows.map((w) => Object.freeze([w[0], w[1]] as readonly [number, number])),
      ),
    }),
    dec: Object.freeze({
      hidden: num(json, 'hidden_size'),
      ffn: num(json, 'intermediate_size'),
      layers: num(json, 'num_hidden_layers'),
      heads: num(json, 'num_attention_heads'),
      headDim: decHeadDim,
      rotaryDim,
      ropeTheta,
    }),
    frameLen,
    vocab: num(json, 'vocab_size'),
    maxPositions: num(json, 'max_position_embeddings'),
    bos,
    eos: num(json, 'eos_token_id'),
  });
}

export function assertConfigMatches(
  parsed: MoonshineStreamingConfig,
  preset: MoonshineStreamingConfig,
): void {
  const walk = (a: unknown, b: unknown, path: string): void => {
    if (typeof a === 'object' && a !== null && typeof b === 'object' && b !== null) {
      for (const key of Object.keys(b)) {
        walk((a as Json)[key], (b as Json)[key], path ? `${path}.${key}` : key);
      }
      return;
    }
    if (a !== b) {
      throw new Error(`moonshine-streaming config: checkpoint ${path}=${a} != preset ${b}`);
    }
  };
  walk(parsed, preset, '');
}
