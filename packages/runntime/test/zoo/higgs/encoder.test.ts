import { describe, expect, it } from 'vitest';
import { matrix, pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import { memoryStateDict } from '../../../src/core/weights/safetensors.ts';
import { HIGGS_ENCODER } from '../../../src/zoo/higgs/config.ts';
import { HiggsEncoder } from '../../../src/zoo/higgs/encoder.ts';
import { resample } from '../../../src/zoo/higgs/resample.ts';
import { expectClose } from '../../gpu.ts';
import { CFG, checkpoint, type Sd } from './encoderCheckpoint.ts';

const POS = 'semantic_model.encoder.pos_conv_embed';
const ATTN = 'semantic_model.encoder.layers.0.attention';
const BOOK = 'quantizer.quantizers.1.codebook';

const read = (sd: ReturnType<typeof memoryStateDict>, name: string) => sd.tensors.get(name)!.f32();

describe('HiggsEncoder', () => {
  it('loads every tensor it reads from the codec checkpoint', async () => {
    const enc = new HiggsEncoder(CFG);
    await enc.loadStateDict(checkpoint(), {
      upload: (_data, shape: ValueMeta) => pending(shape, 'reshape', []),
    });
  });

  it('drops the decoder and keeps the tensors outside its own prefix', () => {
    const raw = checkpoint();
    const sd = memoryStateDict({
      ...Object.fromEntries(Object.entries(raw).map(([name, t]) => [`codec.${name}`, t])),
      'lm.weight': { data: new Float32Array(4), shape: [4] },
    });
    new HiggsEncoder(CFG).transformStateDict(sd, 'codec');
    const names = [...sd.tensors.keys()];
    expect(names).toContain('lm.weight');
    expect(names).toContain('codec.fc.weight');
    expect(
      names.filter((n) => /^codec\.(fc1|fc2|acoustic_decoder|decoder_semantic)\./.test(n)),
    ).toEqual([]);
  });

  it('folds the position conv weight norm and splits it into groups', async () => {
    const raw = checkpoint();
    const sd = memoryStateDict(raw);
    new HiggsEncoder(CFG).semantic_model.encoder.pos_conv_embed.transformStateDict(sd, POS);
    const { hidden, posGroups, posKernel: k } = CFG.hubert;
    const width = hidden / posGroups;
    const g = raw[`${POS}.conv.parametrizations.weight.original0`]!.data;
    const v = raw[`${POS}.conv.parametrizations.weight.original1`]!.data;
    const norm = Array.from({ length: k }, (_, j) =>
      Math.sqrt(v.reduce((s, x, i) => (i % k === j ? s + x * x : s), 0)),
    );
    // Group 1 holds output channels [width, 2·width).
    const want = Array.from(v.subarray(width * width * k, 2 * width * width * k), (x, i) => {
      return (x * g[i % k]!) / norm[i % k]!;
    });
    expectClose(await read(sd, `${POS}.convs.1.weight`), want, { atol: 1e-6, rtol: 1e-6 });
    expect(sd.tensors.get(`${POS}.convs.1.weight`)!.shape).toEqual([width, width, k]);
    expect(await read(sd, `${POS}.convs.1.bias`)).toEqual(
      raw[`${POS}.conv.bias`]!.data.subarray(width, 2 * width),
    );
    expect([...sd.tensors.keys()].filter((n) => n.startsWith(`${POS}.conv.`))).toEqual([]);
  });

  it('stacks q, k and v into one projection, q scaled by 1/√headDim', async () => {
    const raw = checkpoint();
    const sd = memoryStateDict(raw);
    new HiggsEncoder(CFG).semantic_model.encoder.layers.items[0]!.attention.transformStateDict(
      sd,
      ATTN,
    );
    const scale = 1 / Math.sqrt(CFG.hubert.hidden / CFG.hubert.heads);
    for (const part of ['weight', 'bias']) {
      const [q, k, v] = ['q', 'k', 'v'].map((p) => raw[`${ATTN}.${p}_proj.${part}`]!.data);
      expectClose(
        await read(sd, `${ATTN}.qkv.${part}`),
        [...q!.map((x) => x * scale), ...k!, ...v!],
        { atol: 1e-7, rtol: 1e-6 },
      );
    }
  });

  it('scores each codebook row as 2·row·x − |row|² and drops the training state', async () => {
    const raw = checkpoint();
    const sd = memoryStateDict(raw);
    new HiggsEncoder(CFG).quantizer.quantizers.items[1]!.codebook.transformStateDict(sd, BOOK);
    const e = raw[`${BOOK}.embed`]!.data;
    const dim = CFG.codebookDim;
    expectClose(
      await read(sd, `${BOOK}.score.weight`),
      Array.from(e, (x) => 2 * x),
      {
        atol: 0,
        rtol: 0,
      },
    );
    const sq = Array.from(
      { length: CFG.codebookSize },
      (_, i) => -e.subarray(i * dim, (i + 1) * dim).reduce((s, x) => s + x * x, 0),
    );
    expectClose(await read(sd, `${BOOK}.score.bias`), sq, { atol: 1e-6, rtol: 1e-6 });
    expect([...sd.tensors.keys()].filter((n) => n.startsWith(BOOK)).sort()).toEqual(
      [`${BOOK}.embed`, `${BOOK}.score.bias`, `${BOOK}.score.weight`].sort(),
    );
  });

  it.each([
    {
      name: 'a checkpoint without the position conv weight norm',
      edit: (sd: Sd) => delete sd[`${POS}.conv.parametrizations.weight.original0`],
      error:
        /missing 'semantic_model\.encoder\.pos_conv_embed\.conv\.parametrizations\.weight\.original0'/,
    },
    {
      name: 'a position conv of another kernel',
      edit: (sd: Sd) => {
        sd[`${POS}.conv.parametrizations.weight.original1`] = {
          data: new Float32Array(8 * 4 * 5),
          shape: [8, 4, 5],
        };
      },
      error: /pos_conv_embed weight norm has shapes \[1,1,4\] and \[8,4,5\]/,
    },
    {
      name: 'an attention without a v projection',
      edit: (sd: Sd) => delete sd[`${ATTN}.v_proj.bias`],
      error: /missing 'semantic_model\.encoder\.layers\.0\.attention\.v_proj\.bias'/,
    },
  ])('refuses $name before reading any tensor', ({ edit, error }) => {
    const raw = checkpoint();
    edit(raw);
    const sd = memoryStateDict(raw);
    const enc = new HiggsEncoder(CFG);
    const transform = () => {
      enc.semantic_model.encoder.pos_conv_embed.transformStateDict(sd, POS);
      enc.semantic_model.encoder.layers.items[0]!.attention.transformStateDict(sd, ATTN);
    };
    expect(transform).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(error),
      }),
    );
  });

  it('pads the samples to whole frames, and gives HuBERT them at 16 kHz with zeros on both sides', () => {
    const enc = new HiggsEncoder(CFG);
    const samples = Float32Array.from({ length: 7 }, (_, i) => i + 1);
    const { acoustic, semantic, frames } = enc.inputs(samples);
    expect(frames).toBe(3);
    expect([...acoustic]).toEqual([1, 2, 3, 4, 5, 6, 7, 0, 0]);
    const pad = CFG.hubert.pad;
    expect([...semantic]).toEqual([0, 0, ...resample(acoustic, 24_000, 16_000), 0, 0]);
    expect(semantic.length).toBe(6 + 2 * pad);
  });

  it('rejects empty samples with an INVALID_ARGUMENT', () => {
    expect(() => new HiggsEncoder(CFG).inputs(new Float32Array(0))).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('fits the frames of one encode to the largest buffer a device binds', () => {
    // HuBERT's first conv makes the largest buffer: 128·frames + 63 rows of
    // 512 channels, f32.
    const enc = new HiggsEncoder(HIGGS_ENCODER);
    expect(enc.maxFrames(2 ** 27)).toBe(511);
    expect(enc.maxFrames(2048 * (128 + 63))).toBe(1);
    expect(enc.maxFrames(1000)).toBe(0);
  });

  it('turns forward rows [frames, codebooks] into codebook-major codes', () => {
    const rows = Float32Array.from([0, 1, 2, 10, 11, 12]);
    const enc = new HiggsEncoder(CFG);
    expect([...enc.codes(rows, 2)]).toEqual([0, 10, 1, 11, 2, 12]);
    expect(() => enc.codes(rows, 3)).toThrow(/6 rows for 3 codebooks × 3 frames/);
  });

  it.each([
    // 3 frames of samples, and HuBERT inputs of 10 and 8 samples: 2 HuBERT
    // frames, one stride; and 1 HuBERT frame, off the stride.
    { semantic: 10, error: /3 acoustic frames but 1 semantic frames/ },
    { semantic: 8, error: /1 HuBERT frames are not whole strides of 2/ },
  ])('refuses a HuBERT input of $semantic samples for 3 frames', async ({ semantic, error }) => {
    const enc = new HiggsEncoder(CFG);
    await enc.loadStateDict(checkpoint(), {
      upload: (_data, shape: ValueMeta) => pending(shape, 'reshape', []),
    });
    const input = (n: number) => pending(matrix(n, 1), 'reshape', []);
    expect(() => enc.forward(input(9), input(semantic))).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT', message: expect.stringMatching(error) }),
    );
  });

  it('refuses an unbounded buffer size', () => {
    expect(() => new HiggsEncoder(CFG).maxFrames(Infinity)).toThrow(/must be finite/);
  });
});
