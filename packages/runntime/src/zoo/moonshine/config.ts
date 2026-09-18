/** Moonshine architecture config: frozen TS presets, which model code sees,
 *  plus a parser that validates an exported config.json still matches one.
 *
 *  The parser reads both the old top-level rope keys and v5's rope_parameters,
 *  checkpoint keys first. Never trust class defaults — the documented
 *  max_position_embeddings disagrees with the checkpoints. */

import { RunntimeError, type LazyStateDict } from '../../core/index.ts';

export interface MoonshineConfig {
  readonly dModel: number;
  readonly ffn: number;
  readonly encLayers: number;
  readonly decLayers: number;
  readonly heads: number; // MHA — kv heads == attention heads, asserted at parse
  readonly headDim: number; // dModel / heads
  readonly rotaryDim: number; // floor(headDim · partial_rotary_factor), even
  readonly ropeTheta: number;
  readonly vocab: number;
  readonly maxPositions: number;
  readonly bos: number;
  readonly eos: number;
}

export const MOONSHINE_TINY: MoonshineConfig = Object.freeze({
  dModel: 288,
  ffn: 1152,
  encLayers: 6,
  decLayers: 6,
  heads: 8,
  headDim: 36,
  rotaryDim: 32, // int(36 · 0.9)
  ropeTheta: 10000,
  vocab: 32768,
  maxPositions: 194,
  bos: 1,
  eos: 2,
});

export const MOONSHINE_BASE: MoonshineConfig = Object.freeze({
  dModel: 416,
  ffn: 1664,
  encLayers: 8,
  decLayers: 8,
  heads: 8,
  headDim: 52,
  rotaryDim: 32, // int(52 · 0.62)
  ropeTheta: 10000,
  vocab: 32768,
  maxPositions: 194,
  bos: 1,
  eos: 2,
});

export function presetFromStateDict(sd: LazyStateDict): MoonshineConfig {
  const embedding = sd.tensors.get('model.decoder.embed_tokens.weight');
  const width = embedding?.shape[1];
  for (const cfg of [MOONSHINE_TINY, MOONSHINE_BASE]) if (cfg.dModel === width) return cfg;
  throw new RunntimeError(
    'CHECKPOINT_MISMATCH',
    embedding === undefined
      ? 'moonshine weights: no token embedding, not a Moonshine checkpoint'
      : `moonshine weights: embedding width ${width} matches no known size`,
  );
}

type Json = Record<string, unknown>;

function num(json: Json, key: string): number {
  const v = json[key];
  if (typeof v !== 'number')
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine config: '${key}' missing or not a number`,
    );
  return v;
}

export function configFromCheckpoint(json: Json): MoonshineConfig {
  const ropeParams = (json.rope_parameters ?? {}) as Json;
  const ropeTheta =
    (json.rope_theta as number | undefined) ?? (ropeParams.rope_theta as number | undefined);
  const partialRotary =
    (json.partial_rotary_factor as number | undefined) ??
    (ropeParams.partial_rotary_factor as number | undefined);
  if (typeof ropeTheta !== 'number' || typeof partialRotary !== 'number') {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      'moonshine config: rope_theta/partial_rotary_factor missing (checked top-level and rope_parameters)',
    );
  }

  const dModel = num(json, 'hidden_size');
  const heads = num(json, 'encoder_num_attention_heads');
  if (num(json, 'decoder_num_attention_heads') !== heads) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      'moonshine config: encoder/decoder attention heads differ',
    );
  }
  for (const key of ['encoder_num_key_value_heads', 'decoder_num_key_value_heads']) {
    if (json[key] !== undefined && json[key] !== heads) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `moonshine config: '${key}' != attention heads — model assumes MHA`,
      );
    }
  }
  if (json.attention_bias !== false) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      'moonshine config: attention_bias must be false — loader folds no biases',
    );
  }
  if (dModel % heads !== 0) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine config: hidden_size ${dModel} not divisible by ${heads} heads`,
    );
  }
  const headDim = dModel / heads;
  const rotaryDim = Math.floor(headDim * partialRotary); // HF: int(head_dim · factor)
  if (rotaryDim <= 0 || rotaryDim % 2 !== 0 || rotaryDim > headDim) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine config: derived rotaryDim ${rotaryDim} invalid for headDim ${headDim}`,
    );
  }

  const bos = num(json, 'bos_token_id');
  if (json.decoder_start_token_id !== undefined && json.decoder_start_token_id !== bos) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      'moonshine config: decoder_start_token_id != bos_token_id',
    );
  }

  return Object.freeze({
    dModel,
    ffn: num(json, 'intermediate_size'),
    encLayers: num(json, 'encoder_num_hidden_layers'),
    decLayers: num(json, 'decoder_num_hidden_layers'),
    heads,
    headDim,
    rotaryDim,
    ropeTheta,
    vocab: num(json, 'vocab_size'),
    maxPositions: num(json, 'max_position_embeddings'),
    bos,
    eos: num(json, 'eos_token_id'),
  });
}

export function assertConfigMatches(parsed: MoonshineConfig, preset: MoonshineConfig): void {
  for (const key of Object.keys(preset) as (keyof MoonshineConfig)[]) {
    if (parsed[key] !== preset[key]) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `moonshine config: checkpoint ${key}=${parsed[key]} != preset ${preset[key]}`,
      );
    }
  }
}
