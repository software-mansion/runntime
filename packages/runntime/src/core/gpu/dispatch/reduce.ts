/** Dispatch specs for the row reductions (kernels/reduce). */

import { elemFor } from '../../kernels/elem.ts';
import { createSoftmaxPipeline, softmaxHandle } from '../../kernels/reduce/softmax.ts';
import { createMeanPipeline, meanHandle } from '../../kernels/reduce/reduce.ts';
import { createLayerNormPipeline, layerNormHandle } from '../../kernels/reduce/layerNorm.ts';
import { createTopkSelectPipeline, topkSelectHandle } from '../../kernels/reduce/topkSelect.ts';
import { defineSpec, floatDtype, narrow, narrowFloat } from './spec.ts';

export const softmaxSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createSoftmaxPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    return [
      softmaxHandle(
        ctx.root,
        pipeline,
        rows,
        cols,
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const meanSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createMeanPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.pending!.inputs[0]!.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    return [
      meanHandle(
        ctx.root,
        pipeline,
        rows,
        cols,
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const meanSquareSpec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({ inF16: node.pending!.inputs[0]!.shape.dtype === 'f16' }),
  pipeline: (root, _dtype, cfg) =>
    createMeanPipeline(root, elemFor(cfg.inF16 ? 'f16' : 'f32'), true),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const input = node.pending!.inputs[0]!.shape;
    const [rows, cols] = input.dims as [number, number];
    return [
      meanHandle(
        ctx.root,
        pipeline,
        rows,
        cols,
        { x: narrowFloat(inputs[0]!, input.dtype), out: narrowFloat(out, node.shape.dtype) },
        elemFor(input.dtype),
        true,
      ),
    ];
  },
});

export const layerNormSpec = defineSpec({
  attrs: (p) => ({ eps: p.scalar!, hasBias: p.inputs.length > 2 }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createLayerNormPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    const bias = attrs.hasBias ? narrowFloat(inputs[2]!, dtype) : ctx.dummy(floatDtype(dtype));
    return [
      layerNormHandle(
        ctx.root,
        pipeline,
        { rows, cols, eps: attrs.eps, hasBias: attrs.hasBias ? 1 : 0 },
        {
          x: narrowFloat(inputs[0]!, dtype),
          weight: narrowFloat(inputs[1]!, dtype),
          bias,
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

export const topkSpec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({ inF16: node.pending!.inputs[0]!.shape.dtype === 'f16' }),
  pipeline: (root, _dtype, cfg) =>
    createTopkSelectPipeline(root, elemFor(cfg.inF16 ? 'f16' : 'f32')),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const input = node.pending!.inputs[0]!.shape;
    const [tokens, experts] = input.dims as [number, number];
    return [
      topkSelectHandle(
        ctx.root,
        pipeline,
        { tokens, experts },
        { logits: narrowFloat(inputs[0]!, input.dtype), out: narrow(out, 'f32') },
        elemFor(input.dtype),
      ),
    ];
  },
});
