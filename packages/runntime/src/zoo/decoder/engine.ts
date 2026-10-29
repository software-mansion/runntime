/** Decoder-only LLM (Gemma 3, Qwen 3) on raw WebGPU: weight upload, KV
 *  cache and one forward step (a chunk of up to `maxChunk` tokens at
 *  consecutive positions).
 *
 *  Per layer, Gemma 3 (HF Gemma3DecoderLayer):
 *      h  = x + post_attn_norm(o_proj(attn(rope(qk_norm(qkv(input_norm(x)))))))
 *      x' = h + post_ff_norm(down(gelu_tanh(gate(n)) · up(n))),  n = pre_ff_norm(h)
 *  with the embedding scaled by sqrt(hidden), RMSNorm scale (1 + w),
 *  sliding-window layers on rope_local_base_freq and global layers on
 *  rope_theta. Qwen 3 (HF Qwen3DecoderLayer) has pre-norms only:
 *      h  = x + o_proj(attn(rope(qk_norm(qkv(input_norm(x))))))
 *      x' = h + down(silu(gate(n)) · up(n)),  n = post_attention_norm(h)
 *  with scale w and no embedding scaling; each residual add is fused with
 *  the norm that follows it.
 *
 *  The eager runntime graph requires one dtype per matmul, but these models
 *  need f32 activations against bf16 weights, so this path dispatches its
 *  own kernels (kernels.ts) on the runntime device. Bind groups are built
 *  once; a step writes one 16-byte uniform and encodes one compute pass. */

import { RunntimeError, type RangeSource, type SafeTensorInfo } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { DecoderConfig } from './config.ts';
import {
  ARGMAX1_WGSL,
  ARGMAX2_WGSL,
  ATTN_COMBINE_WGSL,
  ATTN_BK,
  ATTN_SPLIT_MAX_M,
  ATTN_WGSL,
  ATTN_ZMAX,
  ADD_RMSNORM_WGSL,
  EMBED_WGSL,
  GATED_ACT_WGSL,
  GEMM_WGSL,
  GEMV_WGSL,
  QKV_WGSL,
  RMSNORM_WGSL,
} from './kernels.ts';

export interface DecoderWeights {
  tensors: Map<string, SafeTensorInfo>;
  source: RangeSource;
}

export interface DecoderEngineOptions {
  /** Longest sequence (prompt + generated) the cache holds. Default 4096. */
  maxContext?: number;
  /** Most tokens one prefill step runs. Default 128. */
  maxChunk?: number;
  onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  signal?: AbortSignal;
}

export interface DecoderEngine {
  readonly cfg: DecoderConfig;
  readonly maxContext: number;
  readonly maxChunk: number;
  /** Runs ids at positions pos0.. through the model, writing their K/V into
   *  the cache. With `logits`, resolves to the last token's logits [vocab]. */
  forward(ids: readonly number[], pos0: number, logits: true): Promise<Float32Array>;
  forward(ids: readonly number[], pos0: number, logits?: false): Promise<undefined>;
  /** Greedy decode of n (1..MAX_BURST) tokens in one submit and one
   *  readback: `first` runs at pos0 and each step's argmax feeds the next on
   *  the GPU. Resolves to the n argmax ids; positions pos0..pos0+n-1 are
   *  written to the cache (the last id is not fed). */
  decodeGreedy(first: number, pos0: number, n: number): Promise<number[]>;
  /** The largest number of trailing cache positions that can be dropped and
   *  recomputed without invalidating the sliding-window rings. */
  readonly maxRewind: number;
  /** GPU bytes held by weights + cache + scratch. */
  readonly gpuBytes: number;
  /** Debug: per-op wall time of one step, each op submitted alone. */
  profile(ids: readonly number[], pos0: number): Promise<Record<string, number>>;
  dispose(): void;
}

/** Most decode steps one greedy burst runs per readback. */
export const MAX_BURST = 32;

const S = 0x80; // GPUBufferUsage.STORAGE
const U = 0x40; // UNIFORM
const CD = 0x08; // COPY_DST
const CS = 0x04; // COPY_SRC
const MR = 0x01; // MAP_READ

type Kind = 'r' | 'rw' | 'u' | 'ud';

/** bf16 bits → f32, as a fresh array. */
function bf16ToF32Array(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  const u = new Uint32Array(out.buffer);
  for (let i = 0; i < bits.length; i++) u[i] = bits[i]! << 16;
  return out;
}

/** Tensor bytes as bf16 bits, whatever the file stored. */
function toBf16Bytes(info: SafeTensorInfo, bytes: Uint8Array): Uint8Array {
  if (info.dtype === 'BF16') return bytes;
  const f32 =
    info.dtype === 'F32'
      ? new Float32Array(bytes.slice().buffer)
      : info.dtype === 'F16'
        ? (() => {
            const h = new Uint16Array(bytes.slice().buffer);
            const o = new Float32Array(h.length);
            for (let i = 0; i < h.length; i++) {
              const v = h[i]!;
              const e = (v >> 10) & 0x1f;
              const f = v & 0x3ff;
              const sgn = v & 0x8000 ? -1 : 1;
              o[i] =
                e === 0
                  ? sgn * f * 2 ** -24
                  : e === 31
                    ? f
                      ? NaN
                      : sgn * Infinity
                    : sgn * (1 + f / 1024) * 2 ** (e - 15);
            }
            return o;
          })()
        : undefined;
  if (!f32) throw checkpointMismatch('decoder weights', `unsupported tensor dtype ${info.dtype}`);
  const u = new Uint32Array(f32.buffer);
  const out = new Uint16Array(f32.length);
  for (let i = 0; i < u.length; i++) {
    const x = u[i]!;
    out[i] = (x + 0x7fff + ((x >> 16) & 1)) >>> 16; // round to nearest even
  }
  return new Uint8Array(out.buffer);
}

function ropeTables(theta: number, scale: number, headDim: number, positions: number) {
  const half = headDim / 2;
  const inv = new Float32Array(half);
  for (let i = 0; i < half; i++) inv[i] = 1 / theta ** ((2 * i) / headDim) / scale;
  const cos = new Float32Array(positions * half);
  const sin = new Float32Array(positions * half);
  for (let p = 0; p < positions; p++) {
    for (let i = 0; i < half; i++) {
      const a = Math.fround(p * inv[i]!);
      cos[p * half + i] = Math.cos(a);
      sin[p * half + i] = Math.sin(a);
    }
  }
  return { cos, sin };
}

export async function loadDecoderEngine(
  device: GPUDevice,
  cfg: DecoderConfig,
  weights: DecoderWeights,
  opts: DecoderEngineOptions = {},
): Promise<DecoderEngine> {
  const { arch, hidden, ffn, layers, heads: H, kvHeads: KVH, headDim: D, vocab: V } = cfg;
  const gemma = arch === 'gemma3';
  const maxContext = Math.min(opts.maxContext ?? 4096, cfg.maxPositions);
  const maxChunk = opts.maxChunk ?? 128;
  if (D > 256 || D % 8 !== 0)
    throw checkpointMismatch(`${arch} config`, `headDim ${D} unsupported`);
  for (const k of [hidden, ffn, H * D]) {
    if (k % 32 !== 0) throw checkpointMismatch(`${arch} config`, `width ${k} not a multiple of 32`);
  }
  const QKVW = (H + 2 * KVH) * D;

  const owned: GPUBuffer[] = [];
  let gpuBytes = 0;
  const buf = (size: number, usage: number, label: string): GPUBuffer => {
    const b = device.createBuffer({
      size: Math.max(16, Math.ceil(size / 4) * 4),
      usage,
      label: `${arch}/${label}`,
    });
    owned.push(b);
    gpuBytes += b.size;
    return b;
  };
  const dispose = () => {
    for (const b of owned) b.destroy();
    owned.length = 0;
  };

  try {
    // ---- weights ------------------------------------------------------
    const name = (n: string): SafeTensorInfo => {
      for (const prefix of ['', 'language_model.', 'model.language_model.']) {
        const t = weights.tensors.get(prefix + n);
        if (t) return t;
      }
      // multimodal checkpoints name the decoder model.language_model.*
      const t = weights.tensors.get(n.replace(/^model\./, 'model.language_model.'));
      if (t) return t;
      throw checkpointMismatch(`${arch} weights`, `missing tensor '${n}'`);
    };
    const expectShape = (n: string, shape: number[]) => {
      const t = name(n);
      if (JSON.stringify(t.shape) !== JSON.stringify(shape)) {
        throw checkpointMismatch(`${arch} weights`, `'${n}' is [${t.shape}], expected [${shape}]`);
      }
      return t;
    };

    const used: SafeTensorInfo[] = [];
    const L = (i: number, n: string) => `model.layers.${i}.${n}`;
    const plan = (n: string, shape: number[]) => {
      const t = expectShape(n, shape);
      used.push(t);
      return t;
    };
    const embInfo = plan('model.embed_tokens.weight', [V, hidden]);
    const headInfo = cfg.tieEmbeddings ? embInfo : plan('lm_head.weight', [V, hidden]);
    const finalInfo = plan('model.norm.weight', [hidden]);
    // Gemma: pre/post norms around both blocks. Qwen: post_attention_layernorm
    // is the norm before the MLP.
    const layerInfo = Array.from({ length: layers }, (_, i) => ({
      inLn: plan(L(i, 'input_layernorm.weight'), [hidden]),
      postAttnLn: plan(L(i, 'post_attention_layernorm.weight'), [hidden]),
      preFfLn: gemma ? plan(L(i, 'pre_feedforward_layernorm.weight'), [hidden]) : undefined,
      postFfLn: gemma ? plan(L(i, 'post_feedforward_layernorm.weight'), [hidden]) : undefined,
      qn: plan(L(i, 'self_attn.q_norm.weight'), [D]),
      kn: plan(L(i, 'self_attn.k_norm.weight'), [D]),
      q: plan(L(i, 'self_attn.q_proj.weight'), [H * D, hidden]),
      k: plan(L(i, 'self_attn.k_proj.weight'), [KVH * D, hidden]),
      v: plan(L(i, 'self_attn.v_proj.weight'), [KVH * D, hidden]),
      o: plan(L(i, 'self_attn.o_proj.weight'), [hidden, H * D]),
      gate: plan(L(i, 'mlp.gate_proj.weight'), [ffn, hidden]),
      up: plan(L(i, 'mlp.up_proj.weight'), [ffn, hidden]),
      down: plan(L(i, 'mlp.down_proj.weight'), [hidden, ffn]),
    }));
    const elemBytes = (t: SafeTensorInfo) => (t.end - t.begin) / t.shape.reduce((a, b) => a * b, 1);
    const totalBytes = used.reduce((s, t) => s + (t.end - t.begin), 0);
    let doneBytes = 0;
    const progress = (n: string, bytes: number) => {
      doneBytes += bytes;
      opts.onProgress?.(n, doneBytes, totalBytes);
    };
    const checkAbort = () => {
      if (opts.signal?.aborted)
        throw new RunntimeError('LOAD_ABORTED', 'model load aborted', {
          cause: opts.signal.reason,
        });
    };

    /** Reads [rowStart, rowEnd) of a 2D tensor as bf16 bytes. */
    const readRows = async (t: SafeTensorInfo, rowStart: number, rowEnd: number) => {
      const rowBytes = (t.end - t.begin) / t.shape[0]!;
      const bytes = await weights.source.read(
        t.begin + rowStart * rowBytes,
        t.begin + rowEnd * rowBytes,
      );
      return toBf16Bytes(t, bytes);
    };
    const loadMatrix = async (label: string, parts: SafeTensorInfo[]): Promise<GPUBuffer> => {
      const rows = parts.reduce((s, t) => s + t.shape[0]!, 0);
      const cols = parts[0]!.shape[1]!;
      const b = buf(rows * cols * 2, S | CD, label);
      let off = 0;
      for (const t of parts) {
        checkAbort();
        const bytes = await readRows(t, 0, t.shape[0]!);
        device.queue.writeBuffer(b, off, bytes as Uint8Array<ArrayBuffer>);
        off += bytes.byteLength;
        progress(label, t.end - t.begin);
      }
      return b;
    };
    /** The RMSNorm scale in f32: (1 + w) for Gemma, w for Qwen. */
    const loadNorm = async (label: string, t: SafeTensorInfo): Promise<GPUBuffer> => {
      const bytes = toBf16Bytes(t, await weights.source.read(t.begin, t.end));
      const w = bf16ToF32Array(new Uint16Array(bytes.slice().buffer));
      if (gemma) for (let i = 0; i < w.length; i++) w[i]! += 1;
      const b = buf(w.byteLength, S | CD, label);
      device.queue.writeBuffer(b, 0, w as Float32Array<ArrayBuffer>);
      progress(label, t.end - t.begin);
      return b;
    };

    // The embedding (and the LM head) is split by rows so every binding
    // stays under the device limits (128 MiB by default).
    const limit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize,
      64 * 1024 * 1024,
    );
    const rowBytes = hidden * 2;
    const nChunks = Math.ceil((V * rowBytes) / limit);
    const chunkRows = Math.ceil(V / nChunks);
    const loadVocabTable = async (info: SafeTensorInfo, label: string) => {
      const chunks: { buf: GPUBuffer; rowBase: number; rows: number }[] = [];
      for (let c = 0; c < nChunks; c++) {
        checkAbort();
        const rowBase = c * chunkRows;
        const rows = Math.min(chunkRows, V - rowBase);
        const b = buf(rows * rowBytes, S | CD, `${label}.${c}`);
        device.queue.writeBuffer(
          b,
          0,
          (await readRows(info, rowBase, rowBase + rows)) as Uint8Array<ArrayBuffer>,
        );
        progress(label, rows * hidden * elemBytes(info));
        chunks.push({ buf: b, rowBase, rows });
      }
      return chunks;
    };
    const embChunks = await loadVocabTable(embInfo, 'embed');
    const headChunks = cfg.tieEmbeddings ? embChunks : await loadVocabTable(headInfo, 'lmHead');

    const layerW = [];
    for (let i = 0; i < layers; i++) {
      const li = layerInfo[i]!;
      layerW.push({
        inLn: await loadNorm(`l${i}.inLn`, li.inLn),
        postAttnLn: await loadNorm(`l${i}.postAttnLn`, li.postAttnLn),
        preFfLn: li.preFfLn && (await loadNorm(`l${i}.preFfLn`, li.preFfLn)),
        postFfLn: li.postFfLn && (await loadNorm(`l${i}.postFfLn`, li.postFfLn)),
        qn: await loadNorm(`l${i}.qn`, li.qn),
        kn: await loadNorm(`l${i}.kn`, li.kn),
        qkv: await loadMatrix(`l${i}.qkv`, [li.q, li.k, li.v]),
        o: await loadMatrix(`l${i}.o`, [li.o]),
        gu: await loadMatrix(`l${i}.gu`, [li.gate, li.up]),
        down: await loadMatrix(`l${i}.down`, [li.down]),
      });
    }
    const finalLn = await loadNorm('final', finalInfo);

    // ---- activations, cache, tables ----------------------------------
    const M = maxChunk;
    const act = (n: number, label: string) => buf(n * 4, S | CD | CS, label);
    const x = act(M * hidden, 'x');
    const xn = act(M * hidden, 'xn');
    const qkv = act(M * QKVW, 'qkv');
    const qb = act(M * H * D, 'q');
    const attn = act(M * H * D, 'attn');
    const y = act(M * hidden, 'y');
    const gu = act(M * 2 * ffn, 'gu');
    const hb = act(M * ffn, 'h');
    const xf = act(hidden, 'xf');
    const logits = act(V, 'logits');
    const ids = buf(M * 4, S | CD, 'ids');
    const readback = buf(V * 4, MR | CD, 'readback');
    // One 16-byte Step record per burst position, at the uniform offset
    // alignment, selected by dynamic offset.
    const stepStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
    const step = buf(stepStride * MAX_BURST, U | CD, 'step');
    const part = buf(256 * 8, S, 'argmax.part');
    const outIds = buf(MAX_BURST * 4, S | CS, 'argmax.out');
    const idsReadback = buf(MAX_BURST * 4, MR | CD, 'argmax.readback');
    // Split-K partials: (token, head, split) — splits only for small steps.
    const parts = H * Math.max(M, ATTN_SPLIT_MAX_M * ATTN_ZMAX);
    const pml = buf(parts * 8, S, 'attn.ml');
    const pacc = buf(parts * D * 4, S, 'attn.acc');

    const W = cfg.slidingWindow;
    const hasSliding = cfg.sliding.some(Boolean);
    const localCap = Math.min(maxContext, W + maxChunk);
    const tables = (theta: number, scale: number, label: string) => {
      const t = ropeTables(theta, scale, D, maxContext);
      const c = buf(t.cos.byteLength, S | CD, `${label}.cos`);
      const s = buf(t.sin.byteLength, S | CD, `${label}.sin`);
      device.queue.writeBuffer(c, 0, t.cos);
      device.queue.writeBuffer(s, 0, t.sin);
      return { cos: c, sin: s };
    };
    const ropeGlobal = tables(cfg.ropeTheta, cfg.ropeScale, 'ropeGlobal');
    const ropeLocal = hasSliding ? tables(cfg.ropeLocalTheta, 1, 'ropeLocal') : ropeGlobal;
    const caches = cfg.sliding.map((sliding, i) => {
      const cap = sliding ? localCap : maxContext;
      return {
        cap,
        sliding,
        k: buf(cap * KVH * D * 4, S, `l${i}.k`),
        v: buf(cap * KVH * D * 4, S, `l${i}.v`),
      };
    });

    // ---- pipelines ------------------------------------------------------
    const layouts = new Map<string, GPUBindGroupLayout>();
    const layoutFor = (kinds: Kind[]): GPUBindGroupLayout => {
      const key = kinds.join(',');
      let l = layouts.get(key);
      if (!l) {
        l = device.createBindGroupLayout({
          entries: kinds.map((k, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer:
              k === 'ud'
                ? { type: 'uniform', hasDynamicOffset: true }
                : { type: k === 'u' ? 'uniform' : k === 'r' ? 'read-only-storage' : 'storage' },
          })),
        });
        layouts.set(key, l);
      }
      return l;
    };
    const kernel = (code: string, kinds: Kind[], label: string) => {
      const layout = layoutFor(kinds);
      const pipeline = device.createComputePipeline({
        label: `${arch}/${label}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: {
          module: device.createShaderModule({ code, label: `${arch}/${label}` }),
          entryPoint: 'main',
        },
      });
      return { pipeline, layout };
    };
    const LIN: Kind[] = ['r', 'r', 'rw', 'u', 'ud'];
    const kEmbed = kernel(EMBED_WGSL, ['r', 'r', 'rw', 'u', 'ud'], 'embed');
    const kNorm = kernel(RMSNORM_WGSL, ['r', 'r', 'rw', 'u', 'ud'], 'rmsnorm');
    const kAddNorm = kernel(ADD_RMSNORM_WGSL, ['r', 'r', 'rw', 'rw', 'u', 'ud'], 'addRmsnorm');
    const kGemv = kernel(GEMV_WGSL, LIN, 'gemv');
    const kGemm = kernel(GEMM_WGSL, LIN, 'gemm');
    const kQkv = kernel(QKV_WGSL(D), ['r', 'r', 'r', 'r', 'r', 'rw', 'rw', 'rw', 'u', 'ud'], 'qkv');
    const kAttn = kernel(ATTN_WGSL(D), ['r', 'r', 'r', 'rw', 'rw', 'u', 'ud'], 'attn');
    const kComb = kernel(ATTN_COMBINE_WGSL(D), ['r', 'r', 'rw', 'u', 'ud'], 'attnCombine');
    const kGated = kernel(GATED_ACT_WGSL, ['r', 'rw', 'u', 'ud'], 'gatedAct');
    const kArg1 = kernel(ARGMAX1_WGSL, ['r', 'rw', 'u', 'ud'], 'argmax1');
    const kArg2 = kernel(ARGMAX2_WGSL, ['r', 'rw', 'rw', 'ud'], 'argmax2');

    // ---- uniforms and bind groups --------------------------------------
    const uniforms = new Map<string, GPUBuffer>();
    const uni = (vals: (number | { f: number })[]): GPUBuffer => {
      const key = vals.map((v) => (typeof v === 'number' ? `u${v}` : `f${v.f}`)).join(',');
      let b = uniforms.get(key);
      if (!b) {
        const data = new ArrayBuffer(Math.max(16, Math.ceil((vals.length * 4) / 16) * 16));
        const u32 = new Uint32Array(data);
        const f32 = new Float32Array(data);
        vals.forEach((v, i) => {
          if (typeof v === 'number') u32[i] = v;
          else f32[i] = v.f;
        });
        b = buf(data.byteLength, U | CD, 'uniform');
        device.queue.writeBuffer(b, 0, data);
        uniforms.set(key, b);
      }
      return b;
    };
    const group = (k: { layout: GPUBindGroupLayout }, bufs: GPUBuffer[]) =>
      device.createBindGroup({
        layout: k.layout,
        entries: bufs.map((buffer, binding) => ({
          binding,
          resource: buffer === step ? { buffer, size: 16 } : { buffer },
        })),
      });

    /** so: byte offset of this step's Step record in the step buffer. */
    type Op = ((pass: GPUComputePassEncoder, m: number, pos0: number, so: number) => void) & {
      label?: string;
    };
    const tag = (label: string, op: Op): Op => Object.assign(op, { label });
    const ceil = (a: number, b: number) => Math.ceil(a / b);
    const linear = (
      xIn: GPUBuffer,
      w: GPUBuffer,
      out: GPUBuffer,
      N: number,
      K: number,
      ldy = N,
      yOff = 0,
      oneRow = false,
    ): Op => {
      const g = group(kGemv, [xIn, w, out, uni([N, K, ldy, yOff]), step]);
      return (pass, m, _pos0, so) => {
        if (m === 1 || oneRow) {
          pass.setPipeline(kGemv.pipeline);
          pass.setBindGroup(0, g, [so]);
          pass.dispatchWorkgroups(ceil(N, 8));
        } else {
          pass.setPipeline(kGemm.pipeline);
          pass.setBindGroup(0, g, [so]);
          pass.dispatchWorkgroups(ceil(N, 64), ceil(m, 64));
        }
      };
    };
    const norm = (
      inp: GPUBuffer,
      w: GPUBuffer,
      out: GPUBuffer,
      N: number,
      mode: 0 | 1,
      lastOnly = false,
    ): Op => {
      const g = group(kNorm, [
        inp,
        w,
        out,
        uni([N, mode, { f: cfg.rmsEps }, lastOnly ? 1 : 0]),
        step,
      ]);
      return (pass, m, _pos0, so) => {
        pass.setPipeline(kNorm.pipeline);
        pass.setBindGroup(0, g, [so]);
        pass.dispatchWorkgroups(lastOnly ? 1 : m);
      };
    };
    /** x += y, then out = norm(x) · w (Qwen's residual add + next norm). */
    const addNorm = (w: GPUBuffer, out: GPUBuffer): Op => {
      const g = group(kAddNorm, [y, w, x, out, uni([hidden, 0, { f: cfg.rmsEps }, 0]), step]);
      return (pass, m, _pos0, so) => {
        pass.setPipeline(kAddNorm.pipeline);
        pass.setBindGroup(0, g, [so]);
        pass.dispatchWorkgroups(m);
      };
    };

    const embedOps: Op[] = embChunks.map((c) => {
      const g = group(kEmbed, [
        ids,
        c.buf,
        x,
        uni([hidden, c.rowBase, c.rows, { f: gemma ? Math.sqrt(hidden) : 1 }]),
        step,
      ]);
      return (pass, m, _pos0, so) => {
        pass.setPipeline(kEmbed.pipeline);
        pass.setBindGroup(0, g, [so]);
        pass.dispatchWorkgroups(ceil(hidden / 2, 256), m);
      };
    });

    const qScale = cfg.queryPreAttnScalar ** -0.5;
    const inLns: GPUBuffer[] = layerW.map((lw) => lw.inLn);
    const layerOps: Op[] = [];
    layerW.forEach((lw, i) => {
      const c = caches[i]!;
      const rope = c.sliding ? ropeLocal : ropeGlobal;
      const gQkv = group(kQkv, [
        qkv,
        lw.qn,
        lw.kn,
        rope.cos,
        rope.sin,
        qb,
        c.k,
        c.v,
        uni([H, KVH, c.cap, { f: cfg.rmsEps }, { f: qScale }, 0, 0, 0]),
        step,
      ]);
      const pAttn = uni([H, KVH, c.cap, c.sliding ? W : 0, { f: cfg.attnLogitSoftcap }, 0, 0, 0]);
      const gAttn = group(kAttn, [qb, c.k, c.v, pml, pacc, pAttn, step]);
      const gComb = group(kComb, [pml, pacc, attn, pAttn, step]);
      const gAct = group(kGated, [gu, hb, uni([ffn, gemma ? 0 : 1, 0, 0]), step]);
      // Qwen: layer i > 0 gets its input norm from the previous layer's
      // fused residual add.
      if (gemma || i === 0) layerOps.push(tag('norm', norm(x, lw.inLn, xn, hidden, 0)));
      layerOps.push(
        tag('qkv', linear(xn, lw.qkv, qkv, QKVW, hidden)),
        tag('qkvPost', (pass, m, _pos0, so) => {
          pass.setPipeline(kQkv.pipeline);
          pass.setBindGroup(0, gQkv, [so]);
          pass.dispatchWorkgroups(H + 2 * KVH, m);
        }),
        tag('attn', (pass, m, pos0, so) => {
          let span = pos0 + m;
          if (c.sliding) span = Math.min(span, W);
          const z = m > ATTN_SPLIT_MAX_M ? 1 : Math.min(Math.ceil(span / ATTN_BK), ATTN_ZMAX);
          pass.setPipeline(kAttn.pipeline);
          pass.setBindGroup(0, gAttn, [so]);
          pass.dispatchWorkgroups(H, m, z);
          pass.setPipeline(kComb.pipeline);
          pass.setBindGroup(0, gComb, [so]);
          pass.dispatchWorkgroups(H, m);
        }),
        tag('o', linear(attn, lw.o, y, hidden, H * D)),
      );
      if (lw.preFfLn && lw.postFfLn) {
        layerOps.push(
          tag('norm', norm(y, lw.postAttnLn, x, hidden, 1)),
          tag('norm', norm(x, lw.preFfLn, xn, hidden, 0)),
        );
      } else {
        layerOps.push(tag('addNorm', addNorm(lw.postAttnLn, xn)));
      }
      layerOps.push(
        tag('gateUp', linear(xn, lw.gu, gu, 2 * ffn, hidden)),
        tag('act', (pass, m, _pos0, so) => {
          pass.setPipeline(kGated.pipeline);
          pass.setBindGroup(0, gAct, [so]);
          pass.dispatchWorkgroups(ceil(ffn, 256), m);
        }),
        tag('down', linear(hb, lw.down, y, hidden, ffn)),
      );
      if (lw.postFfLn) layerOps.push(tag('norm', norm(y, lw.postFfLn, x, hidden, 1)));
      else {
        // The last layer's add also feeds the final norm (its xn is unused:
        // the head norms the last row alone).
        const next = inLns[i + 1] ?? finalLn;
        layerOps.push(tag('addNorm', addNorm(next, xn)));
      }
    });
    const headOps: Op[] = [
      norm(x, finalLn, xf, hidden, 0, true),
      ...headChunks.map((c) => linear(xf, c.buf, logits, c.rows, hidden, V, c.rowBase, true)),
    ];

    const gArg1 = group(kArg1, [logits, part, uni([V, 0, 0, 0]), step]);
    const gArg2 = group(kArg2, [part, ids, outIds, step]);
    const argmaxOp: Op = (pass, _m, _pos0, so) => {
      pass.setPipeline(kArg1.pipeline);
      pass.setBindGroup(0, gArg1, [so]);
      pass.dispatchWorkgroups(256);
      pass.setPipeline(kArg2.pipeline);
      pass.setBindGroup(0, gArg2, [so]);
      pass.dispatchWorkgroups(1);
    };

    const stepData = new Uint32Array(4);
    let busy = false;
    const forward = async (tokenIds: readonly number[], pos0: number, wantLogits = false) => {
      const m = tokenIds.length;
      if (m < 1 || m > maxChunk)
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${arch}: forward takes 1..${maxChunk} tokens, got ${m}`,
        );
      if (pos0 + m > maxContext) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${arch}: position ${pos0 + m} exceeds maxContext ${maxContext}`,
        );
      }
      if (busy)
        throw new RunntimeError('INVALID_ARGUMENT', `${arch}: forward() calls must not overlap`);
      busy = true;
      try {
        stepData[0] = m;
        stepData[1] = pos0;
        device.queue.writeBuffer(step, 0, stepData);
        device.queue.writeBuffer(ids, 0, Uint32Array.from(tokenIds));
        const enc = device.createCommandEncoder({ label: `${arch}/step` });
        const pass = enc.beginComputePass();
        for (const op of embedOps) op(pass, m, pos0, 0);
        for (const op of layerOps) op(pass, m, pos0, 0);
        if (wantLogits) for (const op of headOps) op(pass, m, pos0, 0);
        pass.end();
        if (!wantLogits) {
          device.queue.submit([enc.finish()]);
          return undefined;
        }
        enc.copyBufferToBuffer(logits, 0, readback, 0, V * 4);
        device.queue.submit([enc.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return out;
      } finally {
        busy = false;
      }
    };

    /** n greedy steps in one submit: token `first` at pos0, then each step's
     *  argmax feeds the next on the GPU. Resolves to the n argmax ids (the
     *  last is not fed). */
    const decodeGreedy = async (first: number, pos0: number, n: number): Promise<number[]> => {
      if (n < 1 || n > MAX_BURST)
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${arch}: burst of ${n} outside 1..${MAX_BURST}`,
        );
      if (pos0 + n > maxContext) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${arch}: position ${pos0 + n} exceeds maxContext ${maxContext}`,
        );
      }
      if (busy)
        throw new RunntimeError('INVALID_ARGUMENT', `${arch}: forward() calls must not overlap`);
      busy = true;
      try {
        const steps = new Uint32Array((stepStride / 4) * n);
        for (let k = 0; k < n; k++) {
          const o = (stepStride / 4) * k;
          steps[o] = 1;
          steps[o + 1] = pos0 + k;
          steps[o + 2] = k;
        }
        device.queue.writeBuffer(step, 0, steps);
        device.queue.writeBuffer(ids, 0, Uint32Array.of(first));
        const enc = device.createCommandEncoder({ label: `${arch}/burst` });
        const pass = enc.beginComputePass();
        for (let k = 0; k < n; k++) {
          const so = k * stepStride;
          for (const op of embedOps) op(pass, 1, pos0 + k, so);
          for (const op of layerOps) op(pass, 1, pos0 + k, so);
          for (const op of headOps) op(pass, 1, pos0 + k, so);
          argmaxOp(pass, 1, pos0 + k, so);
        }
        pass.end();
        enc.copyBufferToBuffer(outIds, 0, idsReadback, 0, n * 4);
        device.queue.submit([enc.finish()]);
        await idsReadback.mapAsync(GPUMapMode.READ, 0, n * 4);
        const out = Array.from(new Uint32Array(idsReadback.getMappedRange(0, n * 4)));
        idsReadback.unmap();
        return out;
      } finally {
        busy = false;
      }
    };

    /** Debug: runs one step op by op, each in its own submit, and returns
     *  wall milliseconds per op label (includes per-submit overhead). */
    const profile = async (tokenIds: readonly number[], pos0: number) => {
      const m = tokenIds.length;
      stepData[0] = m;
      stepData[1] = pos0;
      device.queue.writeBuffer(step, 0, stepData);
      device.queue.writeBuffer(ids, 0, Uint32Array.from(tokenIds));
      const times: Record<string, number> = {};
      const all = [
        ...embedOps.map((o) => tag('embed', o)),
        ...layerOps,
        ...headOps.map((o) => tag('head', o)),
      ];
      await device.queue.onSubmittedWorkDone();
      for (const op of all) {
        const t0 = performance.now();
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        op(pass, m, pos0, 0);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        times[op.label ?? '?'] = (times[op.label ?? '?'] ?? 0) + performance.now() - t0;
      }
      return times;
    };

    return {
      profile,
      cfg,
      maxContext,
      maxChunk,
      maxRewind: !hasSliding || localCap >= maxContext ? Infinity : Math.max(0, localCap - W),
      gpuBytes,
      forward: forward as DecoderEngine['forward'],
      decodeGreedy,
      dispose,
    };
  } catch (err) {
    dispose();
    throw err;
  }
}
