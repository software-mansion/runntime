/** Dispatch specs for the convolution kernels (kernels/conv). */

import { elemFor } from '../../kernels/elem.ts';
import { conv1dHandle, createConv1dPipeline } from '../../kernels/conv/conv1d.ts';
import {
  convTranspose2dHandle,
  createConvTranspose2dPipeline,
} from '../../kernels/conv/convTranspose2d.ts';
import {
  channelAffineHwc4Handle,
  conv1x1Kind,
  conv2dHwc4Handle,
  copyChHwc4Handle,
  createAvgPool2dHwc4Pipeline,
  createChannelAffineHwc4Pipeline,
  createConv2dHwc4Pipeline,
  createCopyChHwc4Pipeline,
  createMaxPool2dHwc4Pipeline,
  createPad2dHwc4Pipeline,
  createResizeBilinearHwc4Pipeline,
  createToChwPipeline,
  createToHwc4Pipeline,
  createUpsample2dHwc4Pipeline,
  moveHwc4Handle,
  resizeBilinearHwc4Handle,
  small1x1Lanes,
  toChwHandle,
  toHwc4Handle,
} from '../../kernels/conv/hwc4.ts';
import { defineSpec, floatDtype, narrow, narrowFloat } from './spec.ts';

export const conv1dSpec = defineSpec({
  attrs: (p) => {
    const [kernel, stride, padLeft, , hasBias, act] = p.attrs as [
      number,
      number,
      number,
      number,
      number,
      number,
    ];
    return { kernel, stride, padLeft, hasBias, act };
  },
  cfg: (node, attrs) => ({
    kernel: attrs.kernel,
    stride: attrs.stride,
    cIn: node.pending!.inputs[0]!.shape.dims![1]!,
    cOut: node.shape.dims![1]!,
    padLeft: attrs.padLeft,
    hasBias: attrs.hasBias,
    act: attrs.act,
  }),
  pipeline: (root, dtype, cfg) => createConv1dPipeline(root, cfg, elemFor(dtype)),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const tIn = node.pending!.inputs[0]!.shape.dims![0]!;
    const [tOut, cOut] = node.shape.dims as [number, number];
    return [
      conv1dHandle(
        ctx.root,
        pipeline,
        { tIn, tOut, cOut },
        {
          x: narrowFloat(inputs[0]!, dtype),
          w: narrowFloat(inputs[1]!, dtype),
          bias: attrs.hasBias === 1 ? narrowFloat(inputs[2]!, dtype) : undefined,
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

export const convTranspose2dSpec = defineSpec({
  attrs: (p) => {
    const [k, hasBias] = p.attrs as [number, number];
    return { k, hasBias };
  },
  cfg: (node, attrs) => ({
    cIn: node.pending!.inputs[0]!.shape.dims![0]!,
    cOut: node.shape.dims![0]!,
    k: attrs.k,
    hasBias: attrs.hasBias,
  }),
  pipeline: (root, dtype, cfg) => createConvTranspose2dPipeline(root, cfg, elemFor(dtype)),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    const [, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      convTranspose2dHandle(
        ctx.root,
        pipeline,
        cfg,
        { h, w },
        {
          x: narrowFloat(inputs[0]!, dtype),
          w: narrow(inputs[1]!, 'f16'),
          bias: attrs.hasBias === 1 ? narrowFloat(inputs[2]!, dtype) : ctx.dummy(floatDtype(dtype)),
          out: narrowFloat(out, dtype),
        },
        elemFor(dtype),
      ),
    ];
  },
});

// hwc4 family: f16 only.

function holderBlocks(
  pOut: number,
  extras: { outElems?: number; outBase?: number } | undefined,
): { outC4?: number; outOffB?: number } {
  return {
    outC4: extras?.outElems !== undefined ? extras.outElems / (4 * pOut) : undefined,
    outOffB: extras?.outBase !== undefined ? extras.outBase / (4 * pOut) : undefined,
  };
}

export const conv2dHwc4Spec = defineSpec({
  attrs: (p) => {
    const [kH, kW, stride, padding, groups, hasBias, hasAct] = p.attrs as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ];
    return { kH, kW, stride, padding, groups, hasBias, hasAct: hasAct ?? 0 };
  },
  cfg: (node, attrs, _ctx, extras) => {
    const [cIn, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pixels = h * w;
    const cOut = node.shape.dims![0]!;
    const is1x1 =
      attrs.kH === 1 &&
      attrs.kW === 1 &&
      attrs.stride === 1 &&
      attrs.padding === 0 &&
      attrs.groups === 1;
    const kind1x1 = is1x1 ? conv1x1Kind(pixels, cOut) : 0;
    const oneByOne = kind1x1 === 1;
    return {
      cIn,
      cOut,
      kH: attrs.kH,
      kW: attrs.kW,
      stride: attrs.stride,
      padding: attrs.padding,
      hasBias: attrs.hasBias,
      hasAct: attrs.hasAct,
      hasAdd: extras?.addend !== undefined ? 1 : 0,
      kind: kind1x1 !== 0 ? kind1x1 : attrs.groups !== 1 ? 2 : 0,
      pxT: oneByOne && pixels % 8 === 0 ? 8 : 4,
      ks: kind1x1 === 3 ? small1x1Lanes(cIn) : 0,
    };
  },
  pipeline: (root, _dtype, cfg) => createConv2dHwc4Pipeline(root, cfg),
  encode: ({ node, attrs, cfg, pipeline, inputs, out, extras, ctx }) => {
    const [, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pOut = node.shape.dims![1]! * node.shape.dims![2]!;
    return [
      conv2dHwc4Handle(
        ctx.root,
        pipeline,
        cfg,
        { h, w, ...holderBlocks(pOut, extras) },
        {
          x: narrow(inputs[0]!, 'f16'),
          w: narrow(inputs[1]!, 'f16'),
          bias: attrs.hasBias === 1 ? narrow(inputs[2]!, 'f16') : ctx.dummy('f16'),
          res: extras?.addend !== undefined ? narrow(extras.addend, 'f16') : ctx.dummy('f16'),
          out: narrow(out, 'f16'),
        },
      ),
    ];
  },
});

export const maxPool2dHwc4Spec = defineSpec({
  attrs: (p) => {
    const [kernelSize, stride, padding] = p.attrs as [number, number, number];
    return { kernelSize, stride, padding };
  },
  cfg: (_node, attrs) => attrs,
  pipeline: (root, _dtype, cfg) => createMaxPool2dHwc4Pipeline(root, cfg),
  encode: ({ node, pipeline, inputs, out, extras, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pOut = node.shape.dims![1]! * node.shape.dims![2]!;
    return [
      moveHwc4Handle(
        ctx.root,
        pipeline,
        'maxPool2d_hwc4',
        { c, h, w, outElems: node.shape.elems, ...holderBlocks(pOut, extras) },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const upsample2dHwc4Spec = defineSpec({
  attrs: (p) => ({ scale: p.attrs![0]! }),
  cfg: (_node, attrs) => attrs,
  pipeline: (root, _dtype, cfg) => createUpsample2dHwc4Pipeline(root, cfg),
  encode: ({ node, pipeline, inputs, out, extras, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pOut = node.shape.dims![1]! * node.shape.dims![2]!;
    return [
      moveHwc4Handle(
        ctx.root,
        pipeline,
        'upsample2d_hwc4',
        { c, h, w, outElems: node.shape.elems, ...holderBlocks(pOut, extras) },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const avgPool2dHwc4Spec = defineSpec({
  attrs: (p) => {
    const [kernelSize, stride] = p.attrs as [number, number];
    return { kernelSize, stride };
  },
  cfg: (_node, attrs) => attrs,
  pipeline: (root, _dtype, cfg) => createAvgPool2dHwc4Pipeline(root, cfg),
  encode: ({ node, pipeline, inputs, out, extras, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pOut = node.shape.dims![1]! * node.shape.dims![2]!;
    return [
      moveHwc4Handle(
        ctx.root,
        pipeline,
        'avgPool2d_hwc4',
        { c, h, w, outElems: node.shape.elems, ...holderBlocks(pOut, extras) },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const pad2dHwc4Spec = defineSpec({
  attrs: (p) => {
    const [padH, padW] = p.attrs as [number, number];
    return { padH, padW };
  },
  cfg: (_node, attrs) => attrs,
  pipeline: (root, _dtype, cfg) => createPad2dHwc4Pipeline(root, cfg),
  encode: ({ node, pipeline, inputs, out, extras, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const pOut = node.shape.dims![1]! * node.shape.dims![2]!;
    return [
      moveHwc4Handle(
        ctx.root,
        pipeline,
        'pad2d_hwc4',
        { c, h, w, outElems: node.shape.elems, ...holderBlocks(pOut, extras) },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const resizeBilinearHwc4Spec = defineSpec({
  attrs: (p) => {
    const [outH, outW, alignCorners] = p.attrs as [number, number, number];
    return { outH, outW, alignCorners };
  },
  cfg: (_node, attrs) => ({ alignCorners: attrs.alignCorners }),
  pipeline: (root, _dtype, cfg) => createResizeBilinearHwc4Pipeline(root, cfg),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      resizeBilinearHwc4Handle(
        ctx.root,
        pipeline,
        { c, h, w, outH: attrs.outH, outW: attrs.outW, alignCorners: attrs.alignCorners === 1 },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const channelAffineHwc4Spec = defineSpec({
  attrs: (p) => ({ hasShift: p.attrs![0]! === 1 }),
  cfg: (_node, attrs) => ({ hasShift: attrs.hasShift ? 1 : 0 }),
  pipeline: (root, _dtype, cfg) => createChannelAffineHwc4Pipeline(root, cfg),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      channelAffineHwc4Handle(
        ctx.root,
        pipeline,
        { c, h, w },
        {
          x: narrow(inputs[0]!, 'f16'),
          scale: narrow(inputs[1]!, 'f32'),
          shift: attrs.hasShift ? narrow(inputs[2]!, 'f32') : ctx.dummy('f32'),
          out: narrow(out, 'f16'),
        },
      ),
    ];
  },
});

export const concatChannelsHwc4Spec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root) => createCopyChHwc4Pipeline(root),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const [ca, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const cb = node.pending!.inputs[1]!.shape.dims![0]!;
    const pElems = h * w;
    const dstC4 = Math.ceil(node.shape.dims![0]! / 4);
    const outBuf = narrow(out, 'f16');
    return [
      copyChHwc4Handle(
        ctx.root,
        pipeline,
        { pElems, srcC4: ca / 4, dstC4, srcOffB: 0, dstOffB: 0, nB: ca / 4 },
        { x: narrow(inputs[0]!, 'f16'), out: outBuf },
      ),
      copyChHwc4Handle(
        ctx.root,
        pipeline,
        { pElems, srcC4: cb / 4, dstC4, srcOffB: 0, dstOffB: ca / 4, nB: cb / 4 },
        { x: narrow(inputs[1]!, 'f16'), out: outBuf },
      ),
    ];
  },
});

export const sliceChannelsHwc4Spec = defineSpec({
  attrs: (p) => {
    const [start, end] = p.attrs as [number, number];
    return { start, end };
  },
  cfg: () => undefined,
  pipeline: (root) => createCopyChHwc4Pipeline(root),
  encode: ({ node, attrs, pipeline, inputs, out, extras, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    const nB = (attrs.end - attrs.start) / 4;
    const pElems = h * w;
    const holder = holderBlocks(pElems, extras);
    return [
      copyChHwc4Handle(
        ctx.root,
        pipeline,
        {
          pElems,
          srcC4: Math.ceil(c / 4),
          dstC4: holder.outC4 ?? nB,
          srcOffB: attrs.start / 4,
          dstOffB: holder.outOffB ?? 0,
          nB,
        },
        { x: narrow(inputs[0]!, 'f16'), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const toHwc4Spec = defineSpec({
  attrs: () => undefined,
  cfg: (node) => ({ from: floatDtype(node.pending!.inputs[0]!.shape.dtype) }),
  pipeline: (root, _dtype, cfg) => createToHwc4Pipeline(root, cfg.from),
  encode: ({ node, cfg, pipeline, inputs, out, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      toHwc4Handle(
        ctx.root,
        pipeline,
        cfg.from,
        { c, h, w },
        { x: narrowFloat(inputs[0]!, cfg.from), out: narrow(out, 'f16') },
      ),
    ];
  },
});

export const toChwSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createToChwPipeline(root, floatDtype(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = floatDtype(node.shape.dtype);
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return [
      toChwHandle(
        ctx.root,
        pipeline,
        dtype,
        { c, h, w },
        { x: narrow(inputs[0]!, 'f16'), out: narrowFloat(out, dtype) },
      ),
    ];
  },
});
