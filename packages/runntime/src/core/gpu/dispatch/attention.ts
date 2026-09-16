/** Dispatch specs for attention and rope (kernels/attention). */

import { MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE } from '../../kernels/common.ts';
import { elemFor } from '../../kernels/elem.ts';
import {
  ATTN_ROWS_CHUNK,
  ATTN_ROWS_MAX_CHUNKS,
  attnHandle,
  attnRowsHandle,
  attnRowsSplitHandles,
  createAttnPipeline,
  createAttnRowsMergePipeline,
  createAttnRowsPipeline,
  createAttnRowsSplitPipeline,
  createAttnRowsSubgroupSplitPipeline,
} from '../../kernels/attention/attn.ts';
import {
  ATTN_DECODE_CHUNK,
  ATTN_DECODE_MAX_CHUNKS,
  ATTN_SPLIT_SCRATCH_ELEMS,
  attnDecodeHandle,
  attnDecodeSplitHandles,
  createAttnDecodeMergePipeline,
  createAttnDecodePipeline,
  createAttnDecodeSplitPipeline,
} from '../../kernels/attention/attnDecode.ts';
import { createRopePipeline, ropeHandle } from '../../kernels/attention/rope.ts';
import { defineSpec, narrow, narrowFloat } from './spec.ts';

type AttnRoute = 'decode' | 'decodeSplit' | 'batch' | 'rows' | 'rowsSplit' | 'rowsSubgroupSplit';

/** Chunks per row in the split routes. With segments the longest segment
 *  sets the count; the subgroup pair also serves rows below the chunk size
 *  (chunks = 1). */
const rowsChunks = (effKvLen: number): number =>
  Math.max(1, Math.min(ATTN_ROWS_MAX_CHUNKS, Math.ceil(effKvLen / ATTN_ROWS_CHUNK)));

export const attnSpec = defineSpec({
  attrs: (p) => {
    const a = p.attrs as readonly number[];
    return {
      qHeads: a[0]!,
      kvHeads: a[1]!,
      headDim: a[2]!,
      windowLeft: a[3]!,
      windowRight: a[4]!,
      qPosOffset: a[5]!,
      hasSinks: a[6]!,
      hasSegs: a[7] ?? 0,
      maxSegment: a[8] ?? 0,
      qkvPacked: a[9] ?? 0,
    };
  },
  cfg: (node, attrs, ctx) => {
    const qLen = node.pending!.inputs[0]!.shape.dims![0]!;
    const kvLen = attrs.qkvPacked === 1 ? qLen : node.pending!.inputs[1]!.shape.dims![0]!;
    const effKvLen =
      attrs.hasSegs === 1 && attrs.maxSegment > 0 ? Math.min(attrs.maxSegment, kvLen) : kvLen;
    const f16 = node.shape.dtype === 'f16';
    const rowHeads = qLen * attrs.qHeads;
    // The split routes need a scratch slot per (row, head, chunk). When that
    // buffer would pass the binding limit, the plain rows kernel runs instead.
    const splitBytes = rowHeads * rowsChunks(effKvLen) * (2 + attrs.headDim) * 4;
    const splitFits = splitBytes <= ctx.root.device.limits.maxStorageBufferBindingSize;
    let route: AttnRoute;
    if (qLen === 1) {
      route = kvLen > ATTN_DECODE_CHUNK ? 'decodeSplit' : 'decode';
    } else if (Math.ceil(rowHeads / WORKGROUP_SIZE) > MAX_WORKGROUPS_PER_DIM) {
      route = 'batch';
    } else if (!f16 && ctx.subgroupsOk && rowHeads <= MAX_WORKGROUPS_PER_DIM && splitFits) {
      route = 'rowsSubgroupSplit';
    } else if (!f16 && effKvLen > ATTN_ROWS_CHUNK && splitFits) {
      route = 'rowsSplit';
    } else {
      route = 'rows';
    }
    // qPosOffset and kvLen ride the uniform; only the route depends on them.
    return {
      qHeads: attrs.qHeads,
      kvHeads: attrs.kvHeads,
      headDim: attrs.headDim,
      windowLeft: attrs.windowLeft,
      windowRight: attrs.windowRight,
      hasSinks: attrs.hasSinks,
      hasSegs: attrs.hasSegs,
      qkvPacked: attrs.qkvPacked,
      route,
    };
  },
  pipeline: (root, dtype, cfg) => {
    const elem = elemFor(dtype);
    switch (cfg.route) {
      case 'decode':
        return { route: cfg.route, main: createAttnDecodePipeline(root, cfg, elem) };
      case 'decodeSplit':
        return {
          route: cfg.route,
          partial: createAttnDecodeSplitPipeline(root, cfg, elem),
          merge: createAttnDecodeMergePipeline(root, cfg, elem),
        };
      case 'batch':
        return { route: cfg.route, main: createAttnPipeline(root, cfg, elem) };
      case 'rows':
        return { route: cfg.route, main: createAttnRowsPipeline(root, cfg, elem) };
      case 'rowsSplit':
        return {
          route: cfg.route,
          partial: createAttnRowsSplitPipeline(root, cfg, elem),
          merge: createAttnRowsMergePipeline(root, cfg, elem),
        };
      case 'rowsSubgroupSplit':
        return {
          route: cfg.route,
          partial: createAttnRowsSubgroupSplitPipeline(root, cfg, elem),
          merge: createAttnRowsMergePipeline(root, cfg, elem),
        };
    }
  },
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const elem = elemFor(dtype);
    const packed = attrs.qkvPacked === 1;
    const qLen = node.pending!.inputs[0]!.shape.dims![0]!;
    const kvLen = packed ? qLen : node.pending!.inputs[1]!.shape.dims![0]!;
    const effKvLen =
      attrs.hasSegs === 1 && attrs.maxSegment > 0 ? Math.min(attrs.maxSegment, kvLen) : kvLen;
    const q = narrowFloat(inputs[0]!, dtype);
    const k = packed ? q : narrowFloat(inputs[1]!, dtype);
    const v = packed ? q : narrowFloat(inputs[2]!, dtype);
    const optBase = packed ? 1 : 3;
    const sinks = attrs.hasSinks === 1 ? narrow(inputs[optBase]!, 'f32') : undefined;
    const segs = attrs.hasSegs === 1 ? narrow(inputs[optBase + attrs.hasSinks]!, 'f32') : undefined;
    const args = { qLen, kvLen, qPosOffset: attrs.qPosOffset, qHeads: attrs.qHeads };
    const bufs = { q, k, v, out: narrowFloat(out, dtype), sinks, segs, dummyF32: ctx.dummy('f32') };
    switch (pipeline.route) {
      case 'decode':
        return [attnDecodeHandle(ctx.root, pipeline.main, args, bufs, elem)];
      case 'decodeSplit': {
        const chunks = Math.min(ATTN_DECODE_MAX_CHUNKS, Math.ceil(kvLen / ATTN_DECODE_CHUNK));
        const partials = ctx.scratch('attnDecodeSplit', ATTN_SPLIT_SCRATCH_ELEMS);
        return attnDecodeSplitHandles(
          ctx.root,
          pipeline.partial,
          pipeline.merge,
          { kvLen, qPosOffset: attrs.qPosOffset, qHeads: attrs.qHeads, chunks },
          { ...bufs, partials },
          elem,
        );
      }
      case 'batch':
        return [attnHandle(ctx.root, pipeline.main, args, bufs, elem)];
      case 'rows':
        return [attnRowsHandle(ctx.root, pipeline.main, args, bufs, elem)];
      case 'rowsSplit':
      case 'rowsSubgroupSplit': {
        const chunks = rowsChunks(effKvLen);
        const partials = ctx.scratch(
          'attnRowsSplit',
          qLen * attrs.qHeads * chunks * (2 + attrs.headDim),
        );
        return attnRowsSplitHandles(
          ctx.root,
          pipeline.partial,
          pipeline.merge,
          { ...args, chunks, subgroup: pipeline.route === 'rowsSubgroupSplit' },
          { ...bufs, partials },
          elem,
        );
      }
    }
  },
});

export const ropeSpec = defineSpec({
  attrs: (p) => {
    const [headDim, srcStart] = p.attrs as [number, number, number];
    return { headDim, srcStart };
  },
  cfg: (node, attrs) => ({
    cols: node.shape.dims![1]!,
    headDim: attrs.headDim,
    srcStart: attrs.srcStart,
    srcCols: node.pending!.inputs[0]!.shape.dims![1]!,
  }),
  pipeline: (root, dtype, cfg) => createRopePipeline(root, cfg, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const [rows, cols] = node.shape.dims as [number, number];
    return [
      ropeHandle(
        ctx.root,
        pipeline,
        { rows, cols },
        {
          x: narrowFloat(inputs[0]!, dtype),
          cos: narrow(inputs[1]!, 'f32'),
          sin: narrow(inputs[2]!, 'f32'),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});
