import { describe, expect, it } from 'vitest';
import { bufferSource } from '../../../../src/core/index.ts';
import {
  createTextGenerator,
  type GenerateOptions,
  type TextGenerator,
} from '../../../../src/zoo/tasks/nlp/textGeneration.ts';
import { buffersSince, expectClose, gpu, liveBuffers } from '../../../gpu.ts';
import { referenceLogits, tinyConfig, tinyDecoder } from '../../decoder/tinyDecoder.ts';

await gpu();

const SPECIALS = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<unk>'];
const EOS = 0;
const END_OF_TURN = 2;

/** A tokenizer.json with Qwen's turn markers over the tiny vocab: single
 *  characters and a few merges, padded with plain tokens. */
function tinyTokenizerJson(vocabSize: number) {
  const chars = [...'▁abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?<>/\n'];
  const merges = [
    ['▁', 't'],
    ['h', 'e'],
    ['▁t', 'he'],
    ['i', 'n'],
  ];
  const tokens = [...SPECIALS, ...chars, ...merges.map(([a, b]) => a! + b!)];
  for (let i = tokens.length; i < vocabSize; i++) tokens.push(`tok${i}`);
  return {
    added_tokens: SPECIALS.map((content, id) => ({ id, content, special: true })),
    normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
    pre_tokenizer: null,
    post_processor: null,
    decoder: { type: 'Replace', pattern: { String: '▁' }, content: ' ' },
    model: {
      type: 'BPE',
      vocab: Object.fromEntries(tokens.map((t, i) => [t, i])),
      merges: merges.map(([a, b]) => `${a} ${b}`),
      unk_token: '<unk>',
    },
  };
}

// Qwen's untied LM head keeps random weights from repeating one token.
const cfg = tinyConfig('qwen3');
// Zero LM-head rows tie the stop tokens' logits to exactly 0, so greedy
// decoding on random weights never ends early.
const model = tinyDecoder(cfg, [EOS, END_OF_TURN]);
const tokenizerUrl = URL.createObjectURL(
  new Blob([JSON.stringify(tinyTokenizerJson(cfg.vocab))], { type: 'application/json' }),
);
const load = () =>
  createTextGenerator(
    { modelPath: bufferSource(model.file), tokenizerPath: tokenizerUrl, config: cfg },
    { maxContext: 128, prefillChunk: 16 },
  );

const chat = [{ role: 'user', content: 'Tell me the thing in one line.' }];

/** Text and ids of one call, from onToken. */
async function run(gen: TextGenerator, opts: GenerateOptions = {}, input = chat) {
  const ids: number[] = [];
  let streamed = '';
  const text = await gen.generate(input, {
    maxNewTokens: 20,
    reuseCache: false,
    ...opts,
    onToken: (piece, id) => {
      ids.push(id);
      streamed += piece;
      return opts.onToken?.(piece, id);
    },
  });
  return { text, ids, streamed, stats: gen.lastStats! };
}

describe('createTextGenerator', () => {
  it('gives the reference logits for a chat prompt', async () => {
    const gen = await load();
    try {
      const ids = gen.tokenizer.encode(gen.tokenizer.applyChatTemplate(chat));
      expect(ids[0]).toBe(gen.tokenizer.tokenToId('<|im_start|>'));
      const ref = referenceLogits(model, ids);
      expectClose(await gen.logits(chat), ref[ids.length - 1]!, { atol: 2e-3, rtol: 2e-3 });
    } finally {
      gen.dispose();
    }
  });

  it('decodes the same tokens on the GPU in bursts and on the CPU', async () => {
    const gen = await load();
    try {
      const single = await run(gen, { decodeBurst: 1 });
      const burst = await run(gen, { decodeBurst: 8 });
      // A logits processor moves argmax to the CPU, one readback per step.
      const cpu = await run(gen, { logitsProcessor: () => {} });
      expect(single.ids).toHaveLength(20);
      expect(single.stats.finishReason).toBe('length');
      expect(single.stats.generatedTokens).toBe(20);
      expect(burst.ids).toEqual(single.ids);
      expect(cpu.ids).toEqual(single.ids);
      expect(burst.text).toBe(single.text);
      expect(single.streamed).toBe(single.text);
    } finally {
      gen.dispose();
    }
  });

  it('reuses the KV cache of a shared prefix without changing the output', async () => {
    const gen = await load();
    try {
      const first = await gen.generate(chat, { maxNewTokens: 8 });
      const next = [...chat, { role: 'model', content: first }, { role: 'user', content: 'More.' }];
      const reused = await run(gen, { reuseCache: true }, next);
      expect(reused.stats.reusedTokens).toBeGreaterThan(10);
      const fresh = await run(gen, { reuseCache: false }, next);
      expect(fresh.stats.reusedTokens).toBe(0);
      expect(reused.ids).toEqual(fresh.ids);
    } finally {
      gen.dispose();
    }
  });

  it('keeps every token inside allowedTokenIds', async () => {
    const gen = await load();
    try {
      const allowed = [10, 20, 30, 40];
      const { ids } = await run(gen, { allowedTokenIds: allowed, maxNewTokens: 12 });
      expect(ids).toHaveLength(12);
      for (const id of ids) expect(allowed).toContain(id);
      const perStep = await run(gen, {
        allowedTokenIds: ({ step }) => [50 + step],
        maxNewTokens: 5,
      });
      expect(perStep.ids).toEqual([50, 51, 52, 53, 54]);
    } finally {
      gen.dispose();
    }
  });

  it('cuts the text at a stop string', async () => {
    const gen = await load();
    try {
      const full = await run(gen, { maxNewTokens: 20 });
      // Spans two tokens, so the first one's text must wait for the second.
      const [a, b] = full.ids.slice(3, 5).map((id) => gen.tokenizer.decode([id]));
      const stop = a!.slice(-1) + b!.slice(0, 1);
      const cut = await run(gen, { maxNewTokens: 20, stop: [stop] });
      expect(cut.stats.finishReason).toBe('stop');
      expect(cut.text).toBe(full.text.slice(0, full.text.indexOf(stop)));
      expect(cut.streamed).toBe(cut.text);
    } finally {
      gen.dispose();
    }
  });

  it('stops when onToken returns false, and on abort', async () => {
    const gen = await load();
    try {
      const stopped = await run(gen, { onToken: () => false });
      expect(stopped.ids).toHaveLength(1);
      expect(stopped.stats.finishReason).toBe('callback');
      const ctrl = new AbortController();
      const reason = new Error('user stopped');
      const aborted = gen.generate(chat, {
        maxNewTokens: 20,
        signal: ctrl.signal,
        onToken: () => ctrl.abort(reason),
      });
      await expect(aborted).rejects.toBe(reason);
      expect(gen.lastStats!.finishReason).toBe('aborted');
    } finally {
      gen.dispose();
    }
  });

  it('samples reproducibly from a seed', async () => {
    const gen = await load();
    try {
      // Near-uniform over the 50 kept tokens: other seeds draw other ids.
      const opts = { temperature: 100, topK: 50, topP: 1 };
      const a = await run(gen, { ...opts, seed: 3 });
      expect((await run(gen, { ...opts, seed: 3 })).ids).toEqual(a.ids);
      expect((await run(gen, { ...opts, seed: 4 })).ids).not.toEqual(a.ids);
    } finally {
      gen.dispose();
    }
  });

  it('frees every GPU buffer on dispose and refuses calls after it', async () => {
    const before = liveBuffers();
    const gen = await load();
    await run(gen, { decodeBurst: 4 });
    gen.dispose();
    await expect(gen.generate(chat)).rejects.toThrow(
      expect.objectContaining({ code: 'RESOURCE_DISPOSED' }),
    );
    // dispose waits for the queue of calls, which is empty here.
    await Promise.resolve();
    expect(buffersSince(before)).toEqual([]);
  });

  it('refuses wrong generate arguments, and runs the next call', async () => {
    const gen = await load();
    try {
      const wrong: [string, unknown, GenerateOptions][] = [
        ['maxNewTokens 0', chat, { maxNewTokens: 0 }],
        ['topK 0', chat, { temperature: 1, topK: 0 }],
        ['topP 0', chat, { temperature: 1, topP: 0 }],
        ['topP above 1', chat, { temperature: 1, topP: 1.5 }],
        ['a negative temperature', chat, { temperature: -1 }],
        ['a burst above 32', chat, { decodeBurst: 33 }],
        ['a number as input', 42, {}],
        ['a message without content', [{ role: 'user' }], {}],
        ['every token masked', chat, { allowedTokenIds: [] }],
        ['a prompt past maxContext', 'the '.repeat(200), {}],
      ];
      for (const [name, input, opts] of wrong) {
        await expect(
          gen.generate(input as string, { maxNewTokens: 4, ...opts }),
          name,
        ).rejects.toThrow(expect.objectContaining({ code: 'INVALID_ARGUMENT' }));
      }
      expect((await run(gen)).ids.length).toBeGreaterThan(0);
    } finally {
      gen.dispose();
    }
  });
});

describe('createTextGenerator load errors', () => {
  const jsonUrl = (json: unknown) =>
    URL.createObjectURL(new Blob([JSON.stringify(json)], { type: 'application/json' }));
  const weights = { modelPath: bufferSource(model.file), tokenizerPath: tokenizerUrl };

  it.each([
    {
      name: 'a maxContext of 0',
      model: { ...weights, config: cfg },
      opts: { maxContext: 0 },
      code: 'INVALID_ARGUMENT',
    },
    {
      name: 'an unsupported model_type',
      model: { ...weights, config: jsonUrl({ model_type: 'llama' }) },
      code: 'CHECKPOINT_MISMATCH',
    },
    {
      name: 'weights of another shape',
      model: { ...weights, config: { ...cfg, ffn: 64 } },
      code: 'CHECKPOINT_MISMATCH',
    },
    {
      name: 'a tokenizer that is not BPE',
      model: { ...weights, tokenizerPath: jsonUrl({ model: { type: 'Unigram' } }), config: cfg },
      code: 'CHECKPOINT_MISMATCH',
    },
    {
      name: 'a tokenizer that cannot be fetched',
      model: { ...weights, tokenizerPath: 'blob:null/missing', config: cfg },
      code: 'LOAD_FAILED',
    },
    {
      name: 'an aborted signal',
      model: { ...weights, config: cfg },
      opts: { signal: AbortSignal.abort() },
      code: 'LOAD_ABORTED',
    },
  ])('throws $code for $name, leaving no buffers', async ({ model: m, opts, code }) => {
    const before = liveBuffers();
    await expect(createTextGenerator(m, { maxContext: 32, ...opts })).rejects.toThrow(
      expect.objectContaining({ name: 'RunntimeError', code }),
    );
    expect(buffersSince(before)).toEqual([]);
  });
});
