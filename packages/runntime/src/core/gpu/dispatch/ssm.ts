/** Dispatch specs for the selective-scan kernels (kernels/ssm). All f32. */

import {
  createSsmScanMergePipeline,
  createSsmScanProjectDtPipeline,
  createSsmScanProjectXPipeline,
  createSsmSelectiveScanPipeline,
  ssmScanMergeHandle,
  ssmScanProjectHandles,
  ssmSelectiveScanHandle,
} from '../../kernels/ssm.ts';
import { defineSpec, narrow } from './spec.ts';

export const ssmScanProjectSpec = defineSpec({
  attrs: (p) => ({ rank: p.attrs![0]! }),
  cfg: () => undefined,
  pipeline: (root) => ({
    x: createSsmScanProjectXPipeline(root),
    dt: createSsmScanProjectDtPipeline(root),
  }),
  encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
    const [c, h, w] = node.pending!.inputs[0]!.shape.dims as [number, number, number];
    return ssmScanProjectHandles(
      ctx.root,
      pipeline.x,
      pipeline.dt,
      { c, h, w, rank: attrs.rank },
      {
        src: narrow(inputs[0]!, 'f32'),
        xw: narrow(inputs[1]!, 'f32'),
        dw: narrow(inputs[2]!, 'f32'),
        out: narrow(out, 'f32'),
      },
    );
  },
});

export const ssmSelectiveScanSpec = defineSpec({
  attrs: (p) => {
    const [c, h, w] = p.attrs as [number, number, number];
    return { c, h, w };
  },
  cfg: () => undefined,
  pipeline: (root) => createSsmSelectiveScanPipeline(root),
  encode: ({ attrs, pipeline, inputs, out, ctx }) => [
    ssmSelectiveScanHandle(ctx.root, pipeline, attrs, {
      src: narrow(inputs[0]!, 'f32'),
      proj: narrow(inputs[1]!, 'f32'),
      a: narrow(inputs[2]!, 'f32'),
      dSkip: narrow(inputs[3]!, 'f32'),
      deltaBias: narrow(inputs[4]!, 'f32'),
      out: narrow(out, 'f32'),
    }),
  ],
});

export const ssmScanMergeSpec = defineSpec({
  attrs: (p) => {
    const [c, h, w] = p.attrs as [number, number, number];
    return { c, h, w };
  },
  cfg: () => undefined,
  pipeline: (root) => createSsmScanMergePipeline(root),
  encode: ({ attrs, pipeline, inputs, out, ctx }) => [
    ssmScanMergeHandle(ctx.root, pipeline, attrs, {
      directional: narrow(inputs[0]!, 'f32'),
      out: narrow(out, 'f32'),
    }),
  ],
});
