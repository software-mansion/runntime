/** Dispatch specs for the row reductions (kernels/reduce). */

import { elemFor } from '../../kernels/elem.ts';
import { createSoftmaxPipeline, softmaxHandle } from '../../kernels/reduce/softmax.ts';
import { createMeanPipeline, meanHandle } from '../../kernels/reduce/reduce.ts';
import { createLayerNormPipeline, layerNormHandle } from '../../kernels/reduce/layerNorm.ts';
import { createRmsNormPipeline, rmsNormHandle } from '../../kernels/reduce/rmsNorm.ts';
import { createTopkSelectPipeline, topkSelectHandle } from '../../kernels/reduce/topkSelect.ts';
import { defineSpec, floatDtype, narrow, narrowFloat } from './spec.ts';

export const softmaxSpec = defineSpec({
  attrs: (p) => {
    const [masked, windowLeft, windowRight] = p.attrs as readonly number[];
    if (windowLeft === undefined || windowRight === undefined) {
      throw new Error('dispatch: softmax attrs need [masked, windowLeft, windowRight]');
    }
    return { masked: masked === 1, windowLeft, windowRight };
  },
  // vec4 depends only on cols % 4, so a new width never builds a pipeline.
  cfg: (node, _attrs, ctx) => ({
    vec4: node.shape.dims![1]! % 4 === 0,
    subgroups: ctx.subgroupsOk,
  }),
  pipeline: (root, dtype, cfg) =>
    createSoftmaxPipeline(root, elemFor(dtype), cfg.vec4, cfg.subgroups),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    const { masked, windowLeft, windowRight } = attrs;
    return [
      softmaxHandle(
        ctx.root,
        pipeline,
        {
          rows,
          cols,
          vec4: cfg.vec4,
          subgroups: cfg.subgroups,
          mask: masked ? { windowLeft, windowRight } : undefined,
        },
        {
          x: narrowFloat(inputs[0]!, dtype),
          segs: masked ? narrow(inputs[1]!, 'f32') : ctx.dummy('f32'),
          out: narrowFloat(out, dtype),
        },
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
  attrs: (p) => {
    const [group, outScale] = p.attrs as readonly number[];
    return { eps: p.scalar!, group: group!, outScale: outScale! };
  },
  cfg: (_node, attrs, ctx) => ({
    vec4: attrs.group % 4 === 0,
    subgroups: ctx.subgroupsOk,
  }),
  pipeline: (root, dtype, cfg) =>
    createRmsNormPipeline(root, elemFor(dtype), cfg.vec4, cfg.subgroups),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const hasResidual = inputs.length > 2;
    return [
      rmsNormHandle(
        ctx.root,
        pipeline,
        {
          rows: node.shape.elems / attrs.group,
          cols: attrs.group,
          eps: attrs.eps,
          weightLen: node.pending!.inputs[1]!.shape.elems,
          hasResidual: hasResidual ? 1 : 0,
          outScale: attrs.outScale,
          vec4: cfg.vec4,
          subgroups: cfg.subgroups,
        },
        {
          x: narrowFloat(inputs[0]!, dtype),
          weight: narrowFloat(inputs[1]!, dtype),
          residual: hasResidual ? narrowFloat(inputs[2]!, dtype) : ctx.dummy(floatDtype(dtype)),
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
