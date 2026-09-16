/** Dispatch specs for the elementwise kernels (kernels/elementwise): binary
 *  ops with broadcast, scalar unaries, and the swiglu chunk. */

import type { TgpuRoot } from 'typegpu';
import { type Elem, elemFor } from '../../kernels/elem.ts';
import {
  createAddF32Pipeline,
  createMulF32Pipeline,
  createSubF32Pipeline,
  createSwigluF32Pipeline,
  elementwiseF32Handle,
} from '../../kernels/elementwise/elementwiseF32.ts';
import {
  createAddScalarPipeline,
  createAsinhPipeline,
  createClampScalarPipeline,
  createGeluPipeline,
  createMulScalarPipeline,
  createRsqrtPipeline,
  createSigmoidPipeline,
  createSiluPipeline,
  createTanhPipeline,
  unaryHandle,
} from '../../kernels/elementwise/unary.ts';
import {
  createSwigluChunkPipeline,
  swigluChunkHandle,
} from '../../kernels/elementwise/swigluChunk.ts';
import { defineSpec, narrowFloat } from './spec.ts';

function binarySpec(
  name: string,
  factory: (root: TgpuRoot, elem: Elem) => ReturnType<typeof createAddF32Pipeline>,
) {
  return defineSpec({
    attrs: (p) => ({ mode: p.attrs![0]! }),
    cfg: () => undefined,
    pipeline: (root, dtype) => factory(root, elemFor(dtype)),
    encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
      const dtype = node.shape.dtype;
      const cols = node.shape.dims ? node.shape.dims[1]! : node.shape.elems;
      return [
        elementwiseF32Handle(
          ctx.root,
          pipeline,
          name,
          node.shape.elems,
          cols,
          attrs.mode,
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
}

export const addSpec = binarySpec('add', createAddF32Pipeline);
export const mulSpec = binarySpec('mul', createMulF32Pipeline);
export const subSpec = binarySpec('sub', createSubF32Pipeline);
export const swigluSpec = binarySpec('swiglu', createSwigluF32Pipeline);

const FLT_MAX = 3.4028235e38;
const finite = (v: number) => (v === Infinity ? FLT_MAX : v === -Infinity ? -FLT_MAX : v);

function unarySpec(
  name: string,
  factory: (root: TgpuRoot, elem: Elem) => ReturnType<typeof createRsqrtPipeline>,
  scalars: (p: { scalar?: number; attrs?: readonly number[] }) => { a: number; b: number },
) {
  return defineSpec({
    attrs: scalars,
    cfg: () => undefined,
    pipeline: (root, dtype) => factory(root, elemFor(dtype)),
    encode: ({ node, attrs, pipeline, inputs, out, ctx }) => {
      const dtype = node.shape.dtype;
      return [
        unaryHandle(
          ctx.root,
          pipeline,
          name,
          node.shape.elems,
          attrs.a,
          attrs.b,
          { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
          elemFor(dtype),
        ),
      ];
    },
  });
}

const scalarOnly = (p: { scalar?: number }) => ({ a: p.scalar ?? 0, b: 0 });

export const rsqrtSpec = unarySpec('rsqrt', createRsqrtPipeline, scalarOnly);
export const addScalarSpec = unarySpec('addScalar', createAddScalarPipeline, scalarOnly);
export const sigmoidSpec = unarySpec('sigmoid', createSigmoidPipeline, scalarOnly);
export const mulScalarSpec = unarySpec('mulScalar', createMulScalarPipeline, scalarOnly);
export const tanhSpec = unarySpec('tanh', createTanhPipeline, scalarOnly);
export const geluSpec = unarySpec('gelu', createGeluPipeline, scalarOnly);
export const siluSpec = unarySpec('silu', createSiluPipeline, scalarOnly);
export const asinhSpec = unarySpec('asinh', createAsinhPipeline, scalarOnly);
export const clampScalarSpec = unarySpec('clampScalar', createClampScalarPipeline, (p) => ({
  a: finite(p.attrs![0]!),
  b: finite(p.attrs![1]!),
}));

export const swigluChunkSpec = defineSpec({
  attrs: () => undefined,
  cfg: () => undefined,
  pipeline: (root, dtype) => createSwigluChunkPipeline(root, elemFor(dtype)),
  encode: ({ node, pipeline, inputs, out, ctx }) => {
    const dtype = node.shape.dtype;
    return [
      swigluChunkHandle(
        ctx.root,
        pipeline,
        { total: node.shape.elems, half: node.shape.dims![1]! },
        { x: narrowFloat(inputs[0]!, dtype), out: narrowFloat(out, dtype) },
        elemFor(dtype),
      ),
    ];
  },
});
