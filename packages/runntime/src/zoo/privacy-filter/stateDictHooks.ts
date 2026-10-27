/** Maps the official openai/privacy-filter checkpoint (transformers layout)
 *  onto EagerModel's parameter names. Nothing is fetched here: every tensor
 *  stays lazy until loadStateDict reads it.
 *  - q|k|v projections stack into one qkv weight and bias;
 *  - each expert tensor splits by column into glu and lin halves, with the
 *    expert axis folded into rows: [E, K, 2H] → two [E·K, H];
 *  - the rest renames. score.bias is all zero and the head has no bias, so
 *    it is dropped.
 *  An F16 checkpoint gives f16 derived tensors.
 *  Files from tools/export_pf_safetensors.py use these names as is and
 *  bypass this mapping. */

import {
  concatRows,
  reshapeWeight,
  splitCols,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('privacy-filter state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

function move(sd: LazyStateDict, from: string, to: string): void {
  sd.tensors.set(to, take(sd, from));
}

/** Folds every leading dim into rows: [E, K, H] → [E·K, H]. */
function foldRows(t: LazyTensor): LazyTensor {
  const cols = t.shape[t.shape.length - 1]!;
  return reshapeWeight(t, [t.shape.reduce((a, b) => a * b, 1) / cols, cols]);
}

/** Splits the last dim into glu and lin halves, each folded to 2D:
 *  [E, K, 2H] → two [E·K, H], and a [E, 2H] bias → two [E, H]. */
function splitHalves(t: LazyTensor): [LazyTensor, LazyTensor] {
  const half = t.shape[t.shape.length - 1]! / 2;
  const [glu, lin] = splitCols(t, [half, half]);
  return [foldRows(glu!), foldRows(lin!)];
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
