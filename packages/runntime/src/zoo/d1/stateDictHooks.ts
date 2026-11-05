/** Maps the d1-omni checkpoint onto D1Model's parameter tree as lazy derived
 *  tensors; nothing is fetched here. Audio tensors are dropped. Weights stay
 *  in torch layout for the core Linear hook to transpose. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';

const HIDDEN = 1024;
const HEAD_DIM = 64;
const VISION = 768;
const LAYER_TYPES = 'ccaccaccacacacac';
const HEAD_LAYERS = 2;
const VISION_LAYERS = 12;

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('d1 state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

const move = (sd: LazyStateDict, from: string, to: string) => sd.tensors.set(to, take(sd, from));

/** Rows (or elements, for a vector) scaled, and within each head of
 *  `HEAD_DIM` reordered from rotate-half to interleaved pairs:
 *  new 2i = old i, new 2i+1 = old i + HEAD_DIM/2. */
function ropeRows(t: LazyTensor, scale = 1): LazyTensor {
  const rows = t.shape[0]!;
  const cols = t.shape.length === 2 ? t.shape[1]! : 1;
  const half = HEAD_DIM / 2;
  return derivedTensor(
    t.shape,
    async () => {
      const src = await t.f32();
      const out = new Float32Array(src.length);
      for (let r = 0; r < rows; r++) {
        const head = r - (r % HEAD_DIM);
        const i = r % HEAD_DIM;
        const from = head + (i % 2 === 0 ? i / 2 : half + (i - 1) / 2);
        for (let c = 0; c < cols; c++) out[r * cols + c] = src[from * cols + c]! * scale;
      }
      return out;
    },
    t.byteLength,
  );
}

/** Row-stacks torch-layout tensors, scaling the rows [0, scaledRows) of the
 *  result by `scale`. */
function stackRows(parts: LazyTensor[], scale = 1, scaledRows = 0): LazyTensor {
  const cols = parts[0]!.shape.length === 2 ? parts[0]!.shape[1]! : 1;
  const rows = parts.reduce((s, p) => s + p.shape[0]!, 0);
  return derivedTensor(
    cols === 1 ? [rows] : [rows, cols],
    async () => {
      const out = new Float32Array(rows * cols);
      let base = 0;
      for (const p of parts) {
        const data = await p.f32();
        out.set(data, base);
        base += data.length;
      }
      for (let i = 0; i < scaledRows * cols; i++) out[i]! *= scale;
      return out;
    },
    parts.reduce((s, p) => s + p.byteLength, 0),
  );
}

/** Scales the first `rows` rows of a torch-layout tensor. */
const scaleRows = (t: LazyTensor, rows: number, scale: number) => stackRows([t], scale, rows);

/** The depthwise conv weight [hidden, 1, 3] split into its three taps. */
function convTaps(sd: LazyStateDict, from: string, to: string): void {
  const w = take(sd, from);
  const taps = async () => w.f32();
  for (let k = 0; k < 3; k++) {
    sd.tensors.set(
      `${to}.tap${k}`,
      derivedTensor(
        [HIDDEN],
        async () => {
          const src = await taps();
          return Float32Array.from({ length: HIDDEN }, (_, c) => src[c * 3 + k]!);
        },
        k === 0 ? w.byteLength : 0,
      ),
    );
  }
}

export function transformD1StateDict(sd: LazyStateDict): void {
  for (const name of [...sd.tensors.keys()]) if (name.startsWith('audio.')) sd.tensors.delete(name);
  const qScale = 1 / Math.sqrt(HEAD_DIM);

  move(sd, 'encoder.embedding_norm.weight', 'encoder.embedding_norm.scale');
  [...LAYER_TYPES].forEach((kind, i) => {
    const p = `encoder.layers.${i}`;
    move(sd, `${p}.operator_norm.weight`, `${p}.operator_norm.scale`);
    move(sd, `${p}.ffn_norm.weight`, `${p}.ffn_norm.scale`);
    const w1 = take(sd, `${p}.feed_forward.w1.weight`);
    const w3 = take(sd, `${p}.feed_forward.w3.weight`);
    sd.tensors.set(`${p}.feed_forward.w13.weight`, stackRows([w3, w1]));
    if (kind === 'c') {
      convTaps(sd, `${p}.conv.conv.weight`, `${p}.conv`);
      return;
    }
    const a = `${p}.self_attn`;
    sd.tensors.set(`${a}.q_proj.weight`, ropeRows(take(sd, `${a}.q_proj.weight`)));
    sd.tensors.set(`${a}.k_proj.weight`, ropeRows(take(sd, `${a}.k_proj.weight`)));
    sd.tensors.set(`${a}.q_layernorm.scale`, ropeRows(take(sd, `${a}.q_layernorm.weight`), qScale));
    sd.tensors.set(`${a}.k_layernorm.scale`, ropeRows(take(sd, `${a}.k_layernorm.weight`)));
  });

  for (let i = 0; i < HEAD_LAYERS; i++) {
    const from = `head.head.layers.${i}`;
    const to = `head.layers.${i}`;
    sd.tensors.set(
      `${to}.in_proj.weight`,
      scaleRows(take(sd, `${from}.self_attn.in_proj_weight`), HIDDEN, qScale),
    );
    sd.tensors.set(
      `${to}.in_proj.bias`,
      scaleRows(take(sd, `${from}.self_attn.in_proj_bias`), HIDDEN, qScale),
    );
    for (const n of ['out_proj.weight', 'out_proj.bias'])
      move(sd, `${from}.self_attn.${n}`, `${to}.${n}`);
    for (const m of ['linear1', 'linear2', 'norm1', 'norm2']) {
      for (const n of ['weight', 'bias']) move(sd, `${from}.${m}.${n}`, `${to}.${m}.${n}`);
    }
  }
  for (const [from, to] of [
    ['0', 'scorer_norm'],
    ['1', 'scorer_fc'],
    ['3', 'scorer_out'],
  ] as const) {
    for (const n of ['weight', 'bias']) move(sd, `head.scorer.${from}.${n}`, `head.${to}.${n}`);
  }

  const v = 'vision.tower.vision_model';
  for (const n of ['patch_embedding.weight', 'patch_embedding.bias']) {
    move(sd, `${v}.embeddings.${n}`, `vision.${n}`);
  }
  move(sd, `${v}.embeddings.position_embedding.weight`, 'vision.position_embedding');
  for (const n of ['weight', 'bias'])
    move(sd, `${v}.post_layernorm.${n}`, `vision.post_layernorm.${n}`);
  for (const n of ['linear_1', 'linear_2']) {
    for (const k of ['weight', 'bias']) move(sd, `vision.projector.${n}.${k}`, `vision.${n}.${k}`);
  }
  for (let i = 0; i < VISION_LAYERS; i++) {
    const from = `${v}.encoder.layers.${i}`;
    const to = `vision.layers.${i}`;
    for (const k of ['weight', 'bias']) {
      const qkv = ['q_proj', 'k_proj', 'v_proj'].map((n) =>
        take(sd, `${from}.self_attn.${n}.${k}`),
      );
      sd.tensors.set(`${to}.qkv.${k}`, stackRows(qkv, qScale, VISION));
      move(sd, `${from}.self_attn.out_proj.${k}`, `${to}.out_proj.${k}`);
      for (const m of ['layer_norm1', 'layer_norm2'])
        move(sd, `${from}.${m}.${k}`, `${to}.${m}.${k}`);
      for (const m of ['fc1', 'fc2']) move(sd, `${from}.mlp.${m}.${k}`, `${to}.${m}.${k}`);
    }
  }
}
