/** Maps the official openai/privacy-filter checkpoint (transformers layout)
 *  onto EagerModel's parameter names. Nothing is fetched here: every tensor
 *  stays lazy until loadStateDict reads it.
 *  - q|k|v projections stack into one qkv weight and bias;
 *  - each expert tensor splits by column into glu and lin halves, with the
 *    expert axis folded into rows: [E, K, 2H] → two [E·K, H];
 *  - the rest renames. score.bias is all zero and the head has no bias, so
 *    it is dropped.
 *  Files from tools/export_pf_safetensors.py use these names as is and
 *  bypass this mapping. */

import { RunntimeError, type LazyStateDict, type LazyTensor } from '../../core/index.ts';

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t)
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `privacy-filter state dict: missing tensor '${name}'`,
    );
  sd.tensors.delete(name);
  return t;
}

function move(sd: LazyStateDict, from: string, to: string): void {
  sd.tensors.set(to, take(sd, from));
}

function lazyF32(
  shape: number[],
  byteLength: number,
  compute: () => Promise<Float32Array>,
): LazyTensor {
  const packed = async (): Promise<never> => {
    throw new Error('privacy-filter state dict: derived tensor holds f32');
  };
  return {
    kind: 'derived',
    dtype: 'F32',
    shape,
    byteLength,
    data: compute,
    f32: compute,
    halfWords: packed,
    words: packed,
    u16: packed,
  };
}

const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);

function concatRows(parts: readonly LazyTensor[]): LazyTensor {
  const rest = parts[0]!.shape.slice(1);
  for (const p of parts) {
    if (p.shape.slice(1).join(',') !== rest.join(',')) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `privacy-filter state dict: cannot stack shapes [${parts.map((q) => q.shape.join('x')).join('], [')}]`,
      );
    }
  }
  const rows = parts.reduce((sum, p) => sum + p.shape[0]!, 0);
  const bytes = parts.reduce((sum, p) => sum + p.byteLength, 0);
  return lazyF32([rows, ...rest], bytes, async () => {
    const out = new Float32Array(rows * numel(rest));
    let base = 0;
    for (const p of parts) {
      const data = await p.f32();
      out.set(data, base);
      base += data.length;
    }
    return out;
  });
}

function splitHalves(t: LazyTensor): [LazyTensor, LazyTensor] {
  const width = t.shape[t.shape.length - 1]!;
  const half = width / 2;
  const rows = numel(t.shape.slice(0, -1));
  let shared: Promise<Float32Array> | undefined;
  let reads = 0;
  const source = () => (shared ??= t.f32());
  const columns = (start: number) =>
    lazyF32([rows, half], t.byteLength / 2, async () => {
      const data = await source();
      if (++reads === 2) shared = undefined;
      const out = new Float32Array(rows * half);
      for (let r = 0; r < rows; r++) {
        out.set(data.subarray(r * width + start, r * width + start + half), r * half);
      }
      return out;
    });
  return [columns(0), columns(half)];
}

function foldRows(t: LazyTensor): LazyTensor {
  const cols = t.shape[t.shape.length - 1]!;
  return lazyF32([numel(t.shape) / cols, cols], t.byteLength, () => t.f32());
}

export function transformHfPrivacyFilterStateDict(sd: LazyStateDict): void {
  let n = 0;
  while (sd.tensors.has(`model.layers.${n}.input_layernorm.weight`)) {
    const src = `model.layers.${n}`;
    const dst = `block.${n}`;
    move(sd, `${src}.input_layernorm.weight`, `${dst}.attn.norm.scale`);
    move(sd, `${src}.post_attention_layernorm.weight`, `${dst}.mlp.norm.scale`);
    for (const suffix of ['weight', 'bias']) {
      sd.tensors.set(
        `${dst}.attn.qkv.${suffix}`,
        concatRows(['q', 'k', 'v'].map((p) => take(sd, `${src}.self_attn.${p}_proj.${suffix}`))),
      );
      move(sd, `${src}.self_attn.o_proj.${suffix}`, `${dst}.attn.out.${suffix}`);
      move(sd, `${src}.mlp.router.${suffix}`, `${dst}.mlp.gate.${suffix}`);
    }
    move(sd, `${src}.self_attn.sinks`, `${dst}.attn.sinks`);

    const [gluWeight, linWeight] = splitHalves(take(sd, `${src}.mlp.experts.gate_up_proj`));
    sd.tensors.set(`${dst}.mlp.gluWeight`, gluWeight);
    sd.tensors.set(`${dst}.mlp.linWeight`, linWeight);
    const [gluBias, linBias] = splitHalves(take(sd, `${src}.mlp.experts.gate_up_proj_bias`));
    sd.tensors.set(`${dst}.mlp.gluBias`, gluBias);
    sd.tensors.set(`${dst}.mlp.linBias`, linBias);
    sd.tensors.set(`${dst}.mlp.outWeight`, foldRows(take(sd, `${src}.mlp.experts.down_proj`)));
    move(sd, `${src}.mlp.experts.down_proj_bias`, `${dst}.mlp.outBias`);
    n++;
  }
  move(sd, 'model.embed_tokens.weight', 'embedding.weight');
  move(sd, 'model.norm.weight', 'norm.scale');
  move(sd, 'score.weight', 'unembedding.weight');
  sd.tensors.delete('score.bias');
  sd.metadata['model'] = 'privacy-filter';
}
