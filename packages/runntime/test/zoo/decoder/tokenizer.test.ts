import { Tokenizer } from '@huggingface/tokenizers';
import { describe, expect, it } from 'vitest';
import { isRunntimeError } from '../../../src/core/index.ts';
import {
  createTextTokenizer,
  formatGemmaChat,
  formatQwen3Chat,
} from '../../../src/zoo/decoder/tokenizer.ts';

/** A SentencePiece-style BPE (Gemma's kind): '▁' for spaces, byte fallback
 *  for code points outside the vocab, merges with competing ranks. */
function sentencePieceJson() {
  const specials = ['<pad>', '<eos>', '<bos>', '<start_of_turn>', '<end_of_turn>'];
  const bytes = Array.from(
    { length: 256 },
    (_, b) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`,
  );
  const merges = [
    ['▁', 't'],
    ['h', 'e'],
    ['▁t', 'he'],
    ['e', 'r'],
    ['▁', 'w'],
    ['o', 'r'],
    ['▁w', 'or'],
    ['l', 'd'],
    ['▁wor', 'ld'],
    ['a', 'a'],
    ['aa', 'a'],
  ];
  const chars = [...'▁abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n'];
  const tokens = [...specials, ...bytes, ...chars, ...merges.map(([a, b]) => a! + b!)];
  return {
    added_tokens: specials.map((content, id) => ({ id, content, special: true })),
    normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
    pre_tokenizer: null,
    post_processor: null,
    decoder: {
      type: 'Sequence',
      decoders: [
        { type: 'Replace', pattern: { String: '▁' }, content: ' ' },
        { type: 'ByteFallback' },
        { type: 'Fuse' },
      ],
    },
    model: {
      type: 'BPE',
      vocab: Object.fromEntries(tokens.map((t, i) => [t, i])),
      merges: merges.map(([a, b]) => `${a} ${b}`),
      byte_fallback: true,
      unk_token: null,
    },
  };
}

/** GPT-2 bytes_to_unicode, the alphabet of a byte-level BPE. */
function byteAlphabet(): string[] {
  const out: string[] = [];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    out.push(String.fromCodePoint(printable ? b : 256 + n++));
  }
  return out;
}

/** A byte-level BPE (Qwen's kind) with Qwen 3's split regex and NFC. */
function byteLevelJson() {
  const specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>'];
  const merges = [
    ['Ġ', 't'],
    ['h', 'e'],
    ['Ġt', 'he'],
    ['Ġ', 'w'],
    ['o', 'r'],
    ['Ġw', 'or'],
    ['l', 'd'],
    ['Ġwor', 'ld'],
    ['Ċ', 'Ċ'],
    ['Ã', '©'],
  ];
  const tokens = [...specials, ...byteAlphabet(), ...merges.map(([a, b]) => a! + b!)];
  return {
    added_tokens: specials.map((content, id) => ({ id, content, special: true })),
    normalizer: { type: 'NFC' },
    pre_tokenizer: {
      type: 'Sequence',
      pretokenizers: [
        {
          type: 'Split',
          pattern: {
            Regex:
              "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
          },
          behavior: 'Isolated',
          invert: false,
        },
        { type: 'ByteLevel', add_prefix_space: false, trim_offsets: false, use_regex: false },
      ],
    },
    post_processor: null,
    decoder: { type: 'ByteLevel', add_prefix_space: true, trim_offsets: true, use_regex: true },
    model: {
      type: 'BPE',
      vocab: Object.fromEntries(tokens.map((t, i) => [t, i])),
      merges: merges.map(([a, b]) => [a, b]),
      byte_fallback: false,
      ignore_merges: false,
      unk_token: null,
    },
  };
}

const TEXTS = [
  'the world',
  'aaaaa aaaa',
  'Hello, World!  two  spaces ',
  '\n\nnew lines\n',
  'Zażółć gęślą jaźń, café (café), 日本語, 👩‍👩‍👧',
  "it's THEY'RE 12345 3.14",
  '<start_of_turn>user\nthe<end_of_turn>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
];

describe.each([
  ['sentencepiece', sentencePieceJson()],
  ['byte-level', byteLevelJson()],
] as const)('%s tokenizer', (_, json) => {
  const ours = createTextTokenizer(json);
  const ref = new Tokenizer(json, {});

  it.each(TEXTS)('decodes %j back, and the same in a stream', (text) => {
    const ids = ours.encode(text);
    const want = ref.decode(ids, {
      skip_special_tokens: false,
      clean_up_tokenization_spaces: false,
    });
    expect(ours.decode(ids, { skipSpecial: false })).toBe(want);
    // One id at a time: a character split over byte tokens comes out whole.
    const stream = ours.createStreamDecoder({ skipSpecial: false });
    const pieces = ids.map((id) => stream.push(id));
    expect(pieces.join('') + stream.flush()).toBe(want);
    expect(pieces.some((p) => p.includes('\uFFFD'))).toBe(false);
  });
});

describe('tokenizer stream decoder', () => {
  const tok = createTextTokenizer(sentencePieceJson());

  it('holds back a character until its last byte token', () => {
    // 'é' is outside the vocab: two byte tokens, <0xC3> <0xA9>.
    const ids = tok.encode('aé');
    expect(ids).toHaveLength(3);
    const stream = tok.createStreamDecoder();
    expect(ids.map((id) => stream.push(id))).toEqual(['a', '', 'é']);
    expect(stream.flush()).toBe('');
  });

  it('returns an unfinished character on flush, and starts over', () => {
    const [a, c3] = tok.encode('aé');
    const stream = tok.createStreamDecoder();
    expect(stream.push(a!)).toBe('a');
    expect(stream.push(c3!)).toBe('');
    expect(stream.flush()).toBe('\uFFFD');
    expect(stream.push(a!)).toBe('a');
  });

  it('skips special tokens unless asked', () => {
    const ids = tok.encode('<start_of_turn>the<end_of_turn>');
    const skip = tok.createStreamDecoder();
    expect(ids.map((id) => skip.push(id)).join('')).toBe('the');
    const keep = tok.createStreamDecoder({ skipSpecial: false });
    expect(ids.map((id) => keep.push(id)).join('')).toBe('<start_of_turn>the<end_of_turn>');
  });
});

describe('tokenizer special tokens', () => {
  const tok = createTextTokenizer(sentencePieceJson());

  it('reads the chat format and stop ids from the vocab', () => {
    expect(tok.chatFormat).toBe('gemma');
    expect(tok.bosId).toBe(2);
    expect(tok.stopIds).toEqual([4, 1]);
    expect(createTextTokenizer(byteLevelJson()).chatFormat).toBe('qwen3');
    expect(createTextTokenizer(byteLevelJson()).bosId).toBeUndefined();
  });

  it('parses added tokens, and adds bos only when asked', () => {
    expect(tok.encode('<bos>the')).toEqual([2, ...tok.encode('the')]);
    expect(tok.encode('the')).not.toContain(2);
    expect(tok.encode('the', { addBos: true })).toEqual([2, ...tok.encode('the')]);
    expect(tok.isSpecial(2)).toBe(true);
    expect(tok.isSpecial(tok.encode('the')[0]!)).toBe(false);
  });

  it('drops special tokens on decode unless asked', () => {
    const ids = tok.encode('<start_of_turn>the<end_of_turn>');
    expect(tok.decode(ids)).toBe('the');
    expect(tok.decode(ids, { skipSpecial: false })).toBe('<start_of_turn>the<end_of_turn>');
  });

  it('refuses a tokenizer it cannot run', () => {
    const mismatch = (json: unknown, message: RegExp) => {
      let err: unknown;
      try {
        createTextTokenizer(json);
      } catch (e) {
        err = e;
      }
      expect(isRunntimeError(err, 'CHECKPOINT_MISMATCH')).toBe(true);
      expect((err as Error).message).toMatch(message);
    };
    mismatch({ model: { type: 'Unigram' } }, /BPE/);
    const noTurns = sentencePieceJson();
    noTurns.added_tokens = noTurns.added_tokens.filter((t) => t.content !== '<start_of_turn>');
    delete (noTurns.model.vocab as Record<string, number>)['<start_of_turn>'];
    mismatch(noTurns, /unknown chat format/);
  });
});

/** Expected strings rendered by the checkpoints' own Jinja chat templates
 *  (tokenizer_config.json of google/gemma-3-1b-it and Qwen/Qwen3-0.6B). */
describe('chat templates', () => {
  const tools = [
    {
      type: 'function',
      function: {
        name: 'mute_user',
        description: 'Mute a user',
        parameters: {
          type: 'object',
          properties: { user: { type: 'string' } },
          required: ['user'],
        },
      },
    },
  ];
  const call = '<tool_call>\n{"name": "mute_user", "arguments": {"user": "bob"}}\n</tool_call>';

  it.each([
    {
      name: 'gemma, system folded into the first user turn',
      got: formatGemmaChat([
        { role: 'system', content: 'Be brief.' },
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello!' },
        { role: 'user', content: 'Again' },
      ]),
      want:
        '<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n' +
        '<start_of_turn>user\nAgain<end_of_turn>\n<start_of_turn>model\n',
    },
    {
      name: 'qwen3, thinking off',
      got: formatQwen3Chat([
        { role: 'system', content: 'Be brief.' },
        { role: 'user', content: 'Hi' },
      ]),
      want:
        '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n' +
        '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    },
    {
      name: 'qwen3, thinking on, earlier reasoning dropped',
      got: formatQwen3Chat(
        [
          { role: 'user', content: 'Q' },
          { role: 'assistant', content: '<think>\nhmm\n</think>\n\nA' },
          { role: 'user', content: 'Q2' },
        ],
        { enableThinking: true },
      ),
      want:
        '<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n' +
        '<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n',
    },
    {
      name: 'qwen3, tools and grouped tool responses',
      got: formatQwen3Chat(
        [
          { role: 'user', content: 'Mute bob' },
          { role: 'assistant', content: call },
          { role: 'tool', content: 'ok' },
          { role: 'tool', content: 'done' },
        ],
        { tools },
      ),
      want:
        '<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' +
        'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' +
        '{"type": "function", "function": {"name": "mute_user", "description": "Mute a user", ' +
        '"parameters": {"type": "object", "properties": {"user": {"type": "string"}}, "required": ["user"]}}}\n' +
        '</tools>\n\nFor each function call, return a json object with function name and arguments ' +
        'within <tool_call></tool_call> XML tags:\n<tool_call>\n' +
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' +
        `<|im_start|>user\nMute bob<|im_end|>\n<|im_start|>assistant\n${call}<|im_end|>\n` +
        '<|im_start|>user\n<tool_response>\nok\n</tool_response>\n<tool_response>\ndone\n</tool_response><|im_end|>\n' +
        '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    },
  ])('$name', ({ got, want }) => {
    expect(got).toBe(want);
  });
});
