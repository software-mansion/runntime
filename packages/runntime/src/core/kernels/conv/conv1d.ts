import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Direct strided 1-D convolution, time-major.
 *
 *  One thread per (output frame, 4-channel unit) holds a vec4 accumulator, so
 *  each x value is loaded once and feeds four output channels. The last unit
 *  falls back to a scalar loop when C_out % 4 != 0, keeping the hot loop
 *  branch-free.
 *
 *  Direct rather than im2col, so there is no gather kernel and no scratch. Zero
 *  padding is implicit: out-of-range taps are skipped rather than read. Only
 *  padLeft reaches the shader; padRight just raises T_out, so some windows run
 *  past the last frame into the skip path.
 *
 *  Geometry bakes per pipeline while frame counts ride the uniform, so input
 *  length never compiles a shader. The epilogue runs act(conv + bias). */

const Dims = d.struct({ tIn: d.u32, tOut: d.u32 });

const Config = d.struct({
  kernel: d.u32,
  stride: d.u32,
  cIn: d.u32,
  cOut: d.u32,
  c4c: d.u32, // ceil(cOut/4) — channel units per frame (tail unit may be partial)
  padLeft: d.u32,
  hasBias: d.u32, // comptime 0|1 — epilogue adds bias[ch]
});
const config = tgpu.accessor(Config, {
  kernel: 1,
  stride: 1,
  cIn: 1,
  cOut: 1,
  c4c: 1,
  padLeft: 0,
  hasBias: 0,
});

function makeConv1dVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [T_in, C_in]
    w: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kernel·C_in, C_out]
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [C_out]; dead when hasBias=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [T_out, C_out]
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const f16 = elem.key === 'f16';
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const C = config.$;
    const tIn = layout.$.dims.tIn;
    const tOut = layout.$.dims.tOut;
    const idx = flatIndex(gid);
    if (idx >= tOut * C.c4c) {
      return;
    }
    const frame = d.u32(idx / C.c4c);
    const ch0 = (idx % C.c4c) * 4;
    const windowStart = frame * C.stride; // in padded coordinates

    // The left guard precedes the subtraction so u32 never underflows; the
    // right catches windows padRight pushed past the last frame.
    if (ch0 + 4 <= C.cOut) {
      let acc = d.vec4f();
      for (let tap = d.u32(0); tap < C.kernel; tap++) {
        if (windowStart + tap >= C.padLeft) {
          const xFrame = windowStart + tap - C.padLeft;
          if (xFrame < tIn) {
            const xBase = xFrame * C.cIn;
            const wBase = tap * C.cIn * C.cOut;
            for (let c = d.u32(0); c < C.cIn; c++) {
              const wb = wBase + c * C.cOut + ch0;
              const wv = f16
                ? d.vec4f(
                    d.vec4h(
                      layout.$.w[wb]!,
                      layout.$.w[wb + 1]!,
                      layout.$.w[wb + 2]!,
                      layout.$.w[wb + 3]!,
                    ),
                  )
                : d.vec4f(
                    layout.$.w[wb]!,
                    layout.$.w[wb + 1]!,
                    layout.$.w[wb + 2]!,
                    layout.$.w[wb + 3]!,
                  );
              acc += d.f32(layout.$.x[xBase + c]!) * wv;
            }
          }
        }
      }
      if (C.hasBias > 0) {
        acc += f16
          ? d.vec4f(
              d.vec4h(
                layout.$.bias[ch0]!,
                layout.$.bias[ch0 + 1]!,
                layout.$.bias[ch0 + 2]!,
                layout.$.bias[ch0 + 3]!,
              ),
            )
          : d.vec4f(
              layout.$.bias[ch0]!,
              layout.$.bias[ch0 + 1]!,
              layout.$.bias[ch0 + 2]!,
              layout.$.bias[ch0 + 3]!,
            );
      }
      acc = actSlot.$(acc);
      const oBase = frame * C.cOut + ch0;
      layout.$.out[oBase] = storeScalar(acc.x);
      layout.$.out[oBase + 1] = storeScalar(acc.y);
      layout.$.out[oBase + 2] = storeScalar(acc.z);
      layout.$.out[oBase + 3] = storeScalar(acc.w);
      return;
    }
    // Tail unit: one scalar per remaining channel.
    for (let ch = ch0; ch < C.cOut; ch++) {
      let acc = d.f32(0);
      for (let tap = d.u32(0); tap < C.kernel; tap++) {
        if (windowStart + tap >= C.padLeft) {
          const xFrame = windowStart + tap - C.padLeft;
          if (xFrame < tIn) {
            const xBase = xFrame * C.cIn;
            const wBase = tap * C.cIn * C.cOut;
            for (let c = d.u32(0); c < C.cIn; c++) {
              acc += d.f32(layout.$.x[xBase + c]!) * d.f32(layout.$.w[wBase + c * C.cOut + ch]!);
            }
          }
        }
      }
      if (C.hasBias > 0) {
        acc += d.f32(layout.$.bias[ch]!);
      }
      acc = actSlot.$(acc);
      layout.$.out[frame * C.cOut + ch] = storeScalar(acc);
    }
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeConv1dVariant>>();
export function conv1dVariant(elem: Elem): ReturnType<typeof makeConv1dVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeConv1dVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createConv1dPipeline(
  root: TgpuRoot,
  cfg: {
    kernel: number;
    stride: number;
    cIn: number;
    cOut: number;
    padLeft: number;
    hasBias?: number;
    act?: number; // index into ACTIVATIONS (activations.ts)
  },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(config, {
      kernel: cfg.kernel,
      stride: cfg.stride,
      cIn: cfg.cIn,
      cOut: cfg.cOut,
      c4c: Math.ceil(cfg.cOut / 4),
      padLeft: cfg.padLeft,
      hasBias: cfg.hasBias ?? 0,
    })
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: conv1dVariant(elem).kernel });
}

export function conv1dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConv1dPipeline>,
  args: { tIn: number; tOut: number; cOut: number },
  buffers: { x: FloatBuffer; w: FloatBuffer; out: FloatBuffer; bias?: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = root.createBuffer(Dims, { tIn: args.tIn, tOut: args.tOut }).$usage('uniform');
  // An absent bias aliases `x`; hasBias=0 pipelines never read it.
  const bindGroup = root.createBindGroup(conv1dVariant(elem).layout, {
    ...buffers,
    bias: buffers.bias ?? buffers.x,
    dims,
  });
  return makeHandle(
    pipeline,
    'conv1d',
    bindGroup,
    Math.ceil((args.tOut * Math.ceil(args.cOut / 4)) / WORKGROUP_SIZE),
  );
}
