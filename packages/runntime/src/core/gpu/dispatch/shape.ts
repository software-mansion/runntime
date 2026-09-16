/** Dispatch specs for the layout ops (kernels/shape). The CHW channel ops
 *  are flat copies, so they reuse the row kernels. */

import { elemFor } from '../../kernels/elem.ts';
import { uploadU32 } from '../buffers.ts';
import { castHandle, createCastPipeline } from '../../kernels/shape/cast.ts';
import { createTransposePipeline, transposeHandle } from '../../kernels/shape/transpose.ts';
import {
  concatColsHandle,
  createConcatColsPipeline,
  createSliceColsPipeline,
  sliceColsHandle,
} from '../../kernels/shape/columns.ts';
import {
  concatRowsHandle,
  createConcatRowsPipeline,
  createGatherRowsFromPipeline,
  createGatherRowsPipeline,
  createSliceRowsPipeline,
  createWriteRowsPipeline,
  gatherRowsFromHandle,
  gatherRowsHandle,
  sliceRowsHandle,
  writeRowsHandle,
} from '../../kernels/shape/rows.ts';
import { defineSpec, narrow, narrowFloat } from './spec.ts';

export const astypeSpec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({ fromF16: node.pending!.inputs[0]!.shape.dtype === 'f16' }),
  pipeline: (root, dtype, cfg) =>
    createCastPipeline(root, elemFor(cfg.fromF16 ? 'f16' : 'f32'), elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const from = node.pending!.inputs[0]!.shape.dtype;
    const to = node.shape.dtype;
    return [
      castHandle(
        ctx.root,
        pipeline,
        node.shape.elems,
        { x: narrowFloat(inputs[0]!, from), out: narrowFloat(out, to) },
        elemFor(from),
        elemFor(to),
      ),
    ];
  },
});

export const transposeSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createTransposePipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const [rows, cols] = node.pending!.inputs[0]!.shape.dims as [number, number];
    const dtype = node.shape.dtype;
    return [
      transposeHandle(
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

export const sliceColsSpec = defineSpec({
  attrs: (p) => ({ start: p.attrs![0]! }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createSliceColsPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      sliceColsHandle(
        ctx.root,
        pipeline,
        {
          total: node.shape.elems,
          srcCols: node.pending!.inputs[0]!.shape.dims![1]!,
          start: attrs.start,
          outCols: node.shape.dims![1]!,
        },
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const concatColsSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createConcatColsPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const [a, b] = node.pending!.inputs;
    return [
      concatColsHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, aCols: a!.shape.dims![1]!, bCols: b!.shape.dims![1]! },
        {
          a: narrowFloat(inputs[0]!, dtype),
          b: narrowFloat(inputs[1]!, dtype),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

export const sliceRowsSpec = defineSpec({
  attrs: (p) => ({ start: p.attrs![0]! }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createSliceRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      sliceRowsHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, cols: node.shape.dims![1]!, start: attrs.start },
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

const flatConcatSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createConcatRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      concatRowsHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, aTotal: node.pending!.inputs[0]!.shape.elems },
        {
          a: narrowFloat(inputs[0]!, dtype),
          b: narrowFloat(inputs[1]!, dtype),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});
export const concatRowsSpec = flatConcatSpec;
export const concatChannelsSpec = flatConcatSpec;

export const reshapeSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createSliceRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, extras, ctx }) => {
    const dtype = node.shape.dtype;
    const total = node.shape.elems;
    return [
      sliceRowsHandle(
        ctx.root,
        pipeline,
        { total, cols: total, start: 0, outBase: extras?.outBase ?? 0 },
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const sliceChannelsSpec = defineSpec({
  attrs: (p) => ({ startChannel: p.attrs![0]! }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createSliceRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, extras, ctx }) => {
    const dtype = node.shape.dtype;
    const [, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      sliceRowsHandle(
        ctx.root,
        pipeline,
        {
          total: node.shape.elems,
          cols: h * w,
          start: attrs.startChannel,
          outBase: extras?.outBase ?? 0,
        },
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const writeRowsSpec = defineSpec({
  attrs: (p) => ({ startRow: p.attrs![0]! }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createWriteRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const src = node.pending!.inputs[1]!.shape;
    const cols = src.dims![1]!;
    return [
      writeRowsHandle(
        ctx.root,
        pipeline,
        { total: src.elems, offset: attrs.startRow * cols },
        { src: narrowFloat(inputs[1]!, dtype), dst: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});

export const gatherRowsFromSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createGatherRowsFromPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      gatherRowsFromHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, cols: node.shape.dims![1]! },
        {
          x: narrowFloat(inputs[0]!, dtype),
          idx: narrow(inputs[1]!, 'f32'),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

export const gatherRowsSpec = defineSpec({
  memo: false,
  attrs: (p) => ({ ids: p.attrs! }),
  cfg: () => undefined,
  pipeline: (root, dtype) => createGatherRowsPipeline(root, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const idx = uploadU32(ctx.root, Uint32Array.from(attrs.ids));
    return [
      gatherRowsHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, cols: node.shape.dims![1]! },
        idx,
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});
