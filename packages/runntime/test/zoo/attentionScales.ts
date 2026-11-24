/** Reads the score scale every attention op of a model's graph runs with,
 *  on stand-in weights and inputs, with no GPU. */
import { pending, type Value } from '../../src/core/graph/value.ts';
import { CpuParameter, type AnyModule } from '../../src/core/nn/module.ts';

/** Binds every weight and buffer of `model` to a pending Value of its shape,
 *  and every CPU parameter to ones, so its forward builds a graph. */
export function standInWeights<M extends AnyModule>(model: M): M {
  for (const [, p] of model.storageParameters()) {
    if (p instanceof CpuParameter) p.bindData(new Float32Array(p.shape.elems).fill(1));
    else p.bind(pending(p.shape, 'reshape', []));
  }
  for (const [, b] of model.namedBuffers()) b.bind(pending(b.shape, 'reshape', []));
  return model;
}

/** The scale (attr 10) of every sdpa and sdpaPacked node under `outputs`,
 *  each node once, in no set order. */
export function attentionScales(...outputs: Value[]): number[] {
  const seen = new Set<Value>();
  const scales: number[] = [];
  const stack = [...outputs];
  while (stack.length > 0) {
    const v = stack.pop()!;
    if (seen.has(v)) continue;
    seen.add(v);
    if (!v.pending) continue;
    if (v.pending.op === 'attn') scales.push(v.pending.attrs![10]!);
    stack.push(...v.pending.inputs);
  }
  return scales;
}
