/** Maps the HF Moonshine checkpoint onto the model's parameter tree as lazy
 *  derived tensors; nothing is fetched here.
 *
 *  q/k/v row-stack into one fused weight, so the hot path is a single matmul.
 *  No rotary permutation is needed, since HF's convention already matches the
 *  kernel's. Cross-attention q absorbs the full score scale, having no rope.
 *  Everything else renames.
 *
 *  Weights stay in torch layout for the core Linear and Conv1d hooks to
 *  transpose. Derived byteLengths carry the source file bytes, so load progress
 *  still sums to the checkpoint size. */

import {
  derivedTensor,
  RunntimeError,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';

export const qualify = (prefix: string, name: string): string =>
  prefix ? `${prefix}.${name}` : name;

export function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t)
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine state dict: missing tensor '${name}'`,
    );
  sd.tensors.delete(name);
  return t;
}

export function move(sd: LazyStateDict, from: string, to: string): void {
  sd.tensors.set(to, take(sd, from));
}

export function stackRows(
  parts: readonly { t: LazyTensor; scale?: number }[],
  cols: number,
): LazyTensor {
  let rows = 0;
  for (const { t } of parts) {
    if (t.shape.length !== 2 || t.shape[1] !== cols) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `moonshine state dict: stackRows part shape [${t.shape}] != [_, ${cols}]`,
      );
    }
    rows += t.shape[0]!;
  }
  return derivedTensor(
    [rows, cols],
    async () => {
      const out = new Float32Array(rows * cols);
      let base = 0;
      for (const { t, scale } of parts) {
        const data = await t.f32();
        if (scale === undefined) out.set(data, base);
        else for (let i = 0; i < data.length; i++) out[base + i] = data[i]! * scale;
        base += data.length;
      }
      return out;
    },
    parts.reduce((sum, p) => sum + p.t.byteLength, 0),
  );
}

export function fuseQkv(
  sd: LazyStateDict,
  hfPrefix: string,
  ours: string,
  cols: number,
  qScale?: number,
): void {
  sd.tensors.set(
    ours,
    stackRows(
      ['q', 'k', 'v'].map((p) => ({
        t: take(sd, `${hfPrefix}.${p}_proj.weight`),
        scale: p === 'q' ? qScale : undefined,
      })),
      cols,
    ),
  );
}

export function moveTiedEmbedding(sd: LazyStateDict, prefix: string): void {
  sd.tensors.delete(qualify(prefix, 'proj_out.weight'));
  move(
    sd,
    qualify(prefix, 'model.decoder.embed_tokens.weight'),
    qualify(prefix, 'embedding.weight'),
  );
}

export function transformDecoder(
  sd: LazyStateDict,
  prefix: string,
  dec: { layers: number; hidden: number; headDim: number },
): void {
  const q = (name: string) => qualify(prefix, name);
  const crossQScale = 1 / Math.sqrt(dec.headDim);
  for (let i = 0; i < dec.layers; i++) {
    const hf = q(`model.decoder.layers.${i}`);
    const en = q(`decoder.layers.${i}`);
    move(sd, `${hf}.input_layernorm.weight`, `${en}.norm1.weight`);
    fuseQkv(sd, `${hf}.self_attn`, `${en}.selfAttn.qkv.weight`, dec.hidden);
    move(sd, `${hf}.self_attn.o_proj.weight`, `${en}.selfAttn.out.weight`);
    move(sd, `${hf}.post_attention_layernorm.weight`, `${en}.norm2.weight`);
    sd.tensors.set(
      `${en}.crossAttn.q.weight`,
      stackRows(
        [{ t: take(sd, `${hf}.encoder_attn.q_proj.weight`), scale: crossQScale }],
        dec.hidden,
      ),
    );
    sd.tensors.set(
      `${en}.crossAttn.kv.weight`,
      stackRows(
        ['k', 'v'].map((p) => ({ t: take(sd, `${hf}.encoder_attn.${p}_proj.weight`) })),
        dec.hidden,
      ),
    );
    move(sd, `${hf}.encoder_attn.o_proj.weight`, `${en}.crossAttn.out.weight`);
    move(sd, `${hf}.final_layernorm.weight`, `${en}.norm3.weight`);
    // fc1 is one fused [2·ffn, d] projection; the swigluChunk kernel
    // splits it (hidden = first half, gate = second half).
    move(sd, `${hf}.mlp.fc1.weight`, `${en}.mlp.fc1.weight`);
    move(sd, `${hf}.mlp.fc1.bias`, `${en}.mlp.fc1.bias`);
    move(sd, `${hf}.mlp.fc2.weight`, `${en}.mlp.fc2.weight`);
    move(sd, `${hf}.mlp.fc2.bias`, `${en}.mlp.fc2.bias`);
  }
  move(sd, q('model.decoder.norm.weight'), q('decoder.norm.weight'));
}

export function transformMoonshineStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MoonshineConfig,
): void {
  const q = (name: string) => qualify(prefix, name);
  const d = cfg.dModel;

  moveTiedEmbedding(sd, prefix);

  move(sd, q('model.encoder.conv1.weight'), q('stem.conv1.weight'));
  move(sd, q('model.encoder.groupnorm.weight'), q('stem.groupnorm.weight'));
  move(sd, q('model.encoder.groupnorm.bias'), q('stem.groupnorm.bias'));
  move(sd, q('model.encoder.conv2.weight'), q('stem.conv2.weight'));
  move(sd, q('model.encoder.conv2.bias'), q('stem.conv2.bias'));
  move(sd, q('model.encoder.conv3.weight'), q('stem.conv3.weight'));
  move(sd, q('model.encoder.conv3.bias'), q('stem.conv3.bias'));

  for (let i = 0; i < cfg.encLayers; i++) {
    const hf = q(`model.encoder.layers.${i}`);
    const en = q(`encoder.layers.${i}`);
    move(sd, `${hf}.input_layernorm.weight`, `${en}.norm1.weight`);
    fuseQkv(sd, `${hf}.self_attn`, `${en}.attn.qkv.weight`, d);
    move(sd, `${hf}.self_attn.o_proj.weight`, `${en}.attn.out.weight`);
    move(sd, `${hf}.post_attention_layernorm.weight`, `${en}.norm2.weight`);
    move(sd, `${hf}.mlp.fc1.weight`, `${en}.mlp.fc1.weight`);
    move(sd, `${hf}.mlp.fc1.bias`, `${en}.mlp.fc1.bias`);
    move(sd, `${hf}.mlp.fc2.weight`, `${en}.mlp.fc2.weight`);
    move(sd, `${hf}.mlp.fc2.bias`, `${en}.mlp.fc2.bias`);
  }
  move(sd, q('model.encoder.layer_norm.weight'), q('encoder.norm.weight'));

  transformDecoder(sd, prefix, { layers: cfg.decLayers, hidden: d, headDim: cfg.headDim });
}
