/** Dispatch specs for the matmul family (kernels/matmul). */

import { elemFor } from '../../kernels/elem.ts';
import {
  ARGMAX_SPLIT,
  argmaxDotPartialHandle,
  argmaxDotReduceHandle,
  createArgmaxDotPipeline,
  createArgmaxDotReducePipeline,
} from '../../kernels/matmul/argmaxDot.ts';
import {
  createMatmulPipeline,
  createMatmulTiledPipeline,
  getMatmulAccum,
  MATMUL_TILED_MIN_M,
  matmulHandle,
  matmulTiledHandle,
} from '../../kernels/matmul/matmul.ts';
import {
  createMatmulGemvPipeline,
  createMatmulSmallMPipeline,
  MATMUL_SMALL_M_MAX_ROWS,
  matmulGemvHandle,
  matmulSmallMHandle,
} from '../../kernels/matmul/matmulGemv.ts';
import {
  createMatmulGatherPipeline,
  matmulGatherHandle,
} from '../../kernels/matmul/matmulGather.ts';
import {
  createMatmulGatherQuantWPipeline,
  matmulGatherQuantWHandle,
} from '../../kernels/matmul/matmulGatherQuantW.ts';
import {
  createMatmulQuantWPipeline,
  matmulQuantWHandle,
} from '../../kernels/matmul/matmulQuantW.ts';
import { defineSpec, narrow, narrowFloat } from './spec.ts';

type MatmulRoute = 'plain' | 'gemv' | 'smallm' | 'tiled';

function smallMEligible(m: number, n: number, subgroupsOk: boolean): boolean {
  return subgroupsOk && m >= 4 && m % 4 === 0 && m <= MATMUL_SMALL_M_MAX_ROWS && n % 4 === 0;
}

export const matmulSpec = defineSpec({
  attrs: (p) => {
    const [hasBias, hasAdd, act, baseRow, hasBaseRow] = p.attrs as [
      number,
      number,
      number,
      number,
      number,
    ];
    return { hasBias, hasAdd, act: act ?? 0, baseRow: baseRow ?? 0, isView: hasBaseRow === 1 };
  },
  cfg: (node, attrs, ctx) => {
    const [m, k] = node.pending!.inputs[0]!.shape.dims as [number, number];
    const n = node.shape.dims![1]!;
    const f16 = node.shape.dtype === 'f16';
    let route: MatmulRoute = 'plain';
    if (!attrs.isView) {
      if (smallMEligible(m, n, ctx.subgroupsOk)) route = 'smallm';
      else if (m === 1 && attrs.act === 0) route = 'gemv';
      else if (!f16 && m >= MATMUL_TILED_MIN_M) route = 'tiled';
    }
    return {
      k,
      n,
      hasBias: attrs.hasBias,
      hasAdd: attrs.hasAdd,
      act: attrs.act,
      // Switching accumulation compiles a new pipeline, so it is part of the key.
      accum: f16 ? getMatmulAccum() : 0,
      hasBaseRow: attrs.isView ? 1 : 0,
      route,
    };
  },
  pipeline: (root, dtype, cfg) => {
    const elem = elemFor(dtype);
    switch (cfg.route) {
      case 'smallm':
        return createMatmulSmallMPipeline(root, cfg, elem);
      case 'gemv':
        return createMatmulGemvPipeline(root, cfg, elem);
      case 'tiled':
        return createMatmulTiledPipeline(root, cfg);
      case 'plain':
        return createMatmulPipeline(root, cfg, elem);
    }
  },
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const m = node.pending!.inputs[0]!.shape.dims![0]!;
    const n = cfg.n;
    const biasAt = 2;
    const addendAt = attrs.hasBias === 1 ? 3 : 2;
    const buffers = {
      a: narrowFloat(inputs[0]!, dtype),
      b: narrowFloat(inputs[1]!, dtype),
      bias: attrs.hasBias === 1 ? narrowFloat(inputs[biasAt]!, dtype) : undefined,
      addend: attrs.hasAdd === 1 ? narrowFloat(inputs[addendAt]!, dtype) : undefined,
      out: narrowFloat(out, dtype),
    };
    const elem = elemFor(dtype);
    switch (cfg.route) {
      case 'smallm':
        return [matmulSmallMHandle(ctx.root, pipeline, m, n, buffers, elem)];
      case 'gemv':
        return [matmulGemvHandle(ctx.root, pipeline, n, buffers, elem)];
      case 'tiled':
        return [matmulTiledHandle(ctx.root, pipeline, m, n, buffers)];
      case 'plain':
        return [matmulHandle(ctx.root, pipeline, m, n, buffers, elem, attrs.baseRow)];
    }
  },
});

export const matmulQuantWSpec = defineSpec({
  attrs: (p) => {
    const [baseRow, scaleBase, bits, groupSize] = p.attrs as [number, number, number, number];
    return { baseRow, scaleBase, bits, groupSize };
  },
  cfg: (node, attrs) => ({
    k: node.pending!.inputs[0]!.shape.dims![1]!,
    n: node.shape.dims![1]!,
    bits: attrs.bits,
    groupSize: attrs.groupSize,
  }),
  pipeline: (root, _dtype, cfg) => createMatmulQuantWPipeline(root, cfg),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => [
    matmulQuantWHandle(
      ctx.root,
      pipeline,
      {
        m: node.pending!.inputs[0]!.shape.dims![0]!,
        n: cfg.n,
        baseRow: attrs.baseRow,
        scaleBase: attrs.scaleBase,
      },
      {
        a: narrow(inputs[0]!, 'f32'),
        w: narrow(inputs[1]!, 'u32'),
        scales: narrow(inputs[2]!, 'f32'),
        out: narrow(out, 'f32'),
      },
    ),
  ],
});

export const matmulGatherQuantWSpec = defineSpec({
  attrs: (p) => {
    const [bits, groupSize] = p.attrs as [number, number];
    return { bits, groupSize };
  },
  cfg: (node, attrs) => ({
    k: node.pending!.inputs[0]!.shape.dims![1]!,
    n: node.shape.dims![1]!,
    bits: attrs.bits,
    groupSize: attrs.groupSize,
  }),
  pipeline: (root, _dtype, cfg) => createMatmulGatherQuantWPipeline(root, cfg),
  encode: ({ node, cfg, pipeline, inputs, out, ctx }) => [
    matmulGatherQuantWHandle(
      ctx.root,
      pipeline,
      { m: node.pending!.inputs[0]!.shape.dims![0]!, n: cfg.n },
      {
        a: narrow(inputs[0]!, 'f32'),
        w: narrow(inputs[1]!, 'u32'),
        scales: narrow(inputs[2]!, 'f32'),
        bias: narrow(inputs[3]!, 'f32'),
        expertIdx: narrow(inputs[4]!, 'f32'),
        out: narrow(out, 'f32'),
      },
    ),
  ],
});

export const matmulGatherSpec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({
    k: node.pending!.inputs[0]!.shape.dims![1]!,
    n: node.shape.dims![1]!,
  }),
  pipeline: (root, dtype, cfg) => createMatmulGatherPipeline(root, cfg, elemFor(dtype)),
  encode: ({ node, cfg, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      matmulGatherHandle(
        ctx.root,
        pipeline,
        { m: node.pending!.inputs[0]!.shape.dims![0]!, n: cfg.n },
        {
          a: narrowFloat(inputs[0]!, dtype),
          w: narrowFloat(inputs[1]!, dtype),
          bias: narrowFloat(inputs[2]!, dtype),
          expertIdx: narrow(inputs[3]!, 'f32'),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

export const argmaxDotSpec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({
    cols: node.pending!.inputs[0]!.shape.dims![1]!,
    inF16: node.pending!.inputs[1]!.shape.dtype === 'f16',
  }),
  pipeline: (root, _dtype, cfg) => ({
    partial: createArgmaxDotPipeline(root, { cols: cfg.cols }, elemFor(cfg.inF16 ? 'f16' : 'f32')),
    reduce: createArgmaxDotReducePipeline(root),
  }),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const w = node.pending!.inputs[0]!.shape;
    const partials = ctx.scratch('argmaxDot', ARGMAX_SPLIT * 2);
    return [
      argmaxDotPartialHandle(
        ctx.root,
        pipeline.partial,
        w.dims![0]!,
        { w: narrowFloat(inputs[0]!, w.dtype), x: narrowFloat(inputs[1]!, w.dtype), partials },
        elemFor(w.dtype),
      ),
      argmaxDotReduceHandle(ctx.root, pipeline.reduce, { partials, out: narrow(out, 'f32') }),
    ];
  },
});
