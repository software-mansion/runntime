/** Dispatch specs for the row reductions (kernels/reduce). */

import { elemFor } from '../../kernels/elem.ts';
import { createSoftmaxPipeline, softmaxHandle } from '../../kernels/reduce/softmax.ts';
import { createMeanPipeline, meanHandle } from '../../kernels/reduce/reduce.ts';
import { createLayerNormPipeline, layerNormHandle } from '../../kernels/reduce/layerNorm.ts';
import { createRmsNormPipeline, rmsNormHandle } from '../../kernels/reduce/rmsNorm.ts';
import { createTopkSelectPipeline, topkSelectHandle } from '../../kernels/reduce/topkSelect.ts';
import { defineSpec, floatDtype, narrow, narrowFloat } from './spec.ts';

export const softmaxSpec = defineSpec({
  attrs: () => undefined,
  // vec4 depends only on cols % 4, so a new width never builds a pipeline.
  cfg: (node, _attrs, ctx) => ({
    vec4: node.shape.dims![1]! % 4 === 0,
    subgroups: ctx.subgroupsOk,
  }),
  pipeline: (root, dtype, cfg) =>
    createSoftmaxPipeline(root, elemFor(dtype), cfg.vec4, cfg.subgroups),
  encode: ({ node, cfg, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    return [
      softmaxHandle(
        ctx.root,
        pipeline,
        { rows, cols, vec4: cfg.vec4, subgroups: cfg.subgroups },
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

export const rmsNormSpec = defineSpec({
  attrs: (p) => ({ eps: p.scalar! }),
  cfg: (node, _attrs, ctx) => ({
    vec4: node.shape.dims![1]! % 4 === 0,
    subgroups: ctx.subgroupsOk,
  }),
  pipeline: (root, dtype, cfg) =>
    createRmsNormPipeline(root, elemFor(dtype), cfg.vec4, cfg.subgroups),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    return [
      rmsNormHandle(
        ctx.root,
        pipeline,
        { rows, cols, eps: attrs.eps, vec4: cfg.vec4, subgroups: cfg.subgroups },
        {
          x: narrowFloat(inputs[0]!, dtype),
          weight: narrowFloat(inputs[1]!, dtype),
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
