/** Text tokenizer of the decoder models: tokenizers.js runs the Hub's
 *  tokenizer.json (Gemma 3's SentencePiece-style BPE, Qwen 3's byte-level
 *  BPE), and this file adds the stream decoder and each model's chat
 *  template. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch, messageOf } from '../errors.ts';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'model' | 'tool' | (string & {});
  content: string;
}

export interface ChatTemplateOptions {
  /** End with an open assistant turn. Default true. */
  addGenerationPrompt?: boolean;
  /** Qwen 3: let the model think first. Default false, which pre-fills an
   *  empty `<think>\n\n</think>\n\n` block after the generation prompt (the
   *  HF template's enable_thinking=False). Ignored by Gemma. */
  enableThinking?: boolean;
  /** Qwen 3: function schemas listed in the system turn, rendered as the HF
   *  template's `tools=` does. Ignored by Gemma. */
  tools?: readonly unknown[];
}

/** Which chat template a tokenizer applies. */
export type ChatFormat = 'gemma' | 'qwen3';

export interface TextTokenizer {
  readonly chatFormat: ChatFormat;
  readonly vocabSize: number;
  /** Prepended by `encode(..., { addBos: true })`; undefined where the model
   *  has none (Qwen). */
  readonly bosId: number | undefined;
  readonly eosId: number;
  /** Closes a chat turn (`<end_of_turn>`, `<|im_end|>`). */
  readonly endOfTurnId: number;
  /** Ids that end generation: end of turn, end of text. */
  readonly stopIds: readonly number[];
  /** Text → ids. `addBos` prepends the bos token where the model has one
   *  (default false). Added tokens (`<start_of_turn>`, `<|im_start|>`, ...)
   *  in the text become their single ids. */
  encode(text: string, opts?: { addBos?: boolean }): number[];
  /** Ids → text. Special tokens (`<bos>`, `<end_of_turn>`, `<|im_end|>`, ...)
   *  are dropped unless `skipSpecial: false`. */
  decode(ids: readonly number[], opts?: { skipSpecial?: boolean }): string;
  /** Incremental decoder: push ids one at a time, get back the newly complete
   *  text (multi-byte characters split over byte tokens are held back).
   *  `flush()` returns the rest and resets it. */
  createStreamDecoder(opts?: { skipSpecial?: boolean }): {
    push(id: number): string;
    flush(): string;
  };
  tokenToId(token: string): number | undefined;
  idToToken(id: number): string | undefined;
  /** True for added tokens flagged special (`<bos>`, `<eos>`, turn markers). */
  isSpecial(id: number): boolean;
  /** The model's chat template, ending in an open assistant turn.
   *  Gemma: `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
   *  (no system role: a system message is prepended to the first user turn).
   *  Qwen 3: ChatML, `<|im_start|>role\n...<|im_end|>\n`. */
  applyChatTemplate(messages: readonly ChatMessage[], opts?: ChatTemplateOptions): string;
}

const isBpeTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Builds a tokenizer from a parsed Hugging Face tokenizer.json. The chat
 *  template follows from the vocab (`<start_of_turn>` → gemma,
 *  `<|im_start|>` → qwen3) unless `chatFormat` names it. Throws
 *  CHECKPOINT_MISMATCH for a file it cannot run. */
export function createTextTokenizer(
  json: unknown,
  opts: { chatFormat?: ChatFormat } = {},
): TextTokenizer {
  const bad = (why: string) => checkpointMismatch('text tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');
  let tokenizer: Tokenizer;
  try {
    tokenizer = new Tokenizer(json, {});
  } catch (err) {
    throw bad(messageOf(err));
  }
  const vocab = tokenizer.get_vocab(true);
  let vocabSize = 0;
  for (const id of vocab.values()) vocabSize = Math.max(vocabSize, id + 1);
  const special = new Set<number>();
  for (const [id, t] of tokenizer.get_added_tokens_decoder()) if (t.special) special.add(id);

  // tokenizers.js refuses an empty list.
  const decode = (ids: readonly number[], o: { skipSpecial?: boolean } = {}) =>
    ids.length === 0
      ? ''
      : tokenizer.decode(ids as number[], {
          skip_special_tokens: o.skipSpecial ?? true,
          clean_up_tokenization_spaces: false,
        });

  // Decodes a window of the last ids and emits what it adds to the text
  // before the newest one. A trailing U+FFFD is a character still split
  // over byte tokens, so it waits for the next id.
  const createStreamDecoder = (o: { skipSpecial?: boolean } = {}) => {
    let ids: number[] = [];
    // Window start, and the end of what was already emitted.
    let prefix = 0;
    let read = 0;
    const tail = (): string => {
      const done = decode(ids.slice(prefix, read), o);
      const text = decode(ids.slice(prefix), o);
      return text.length > done.length ? text.slice(done.length) : '';
    };
    return {
      push(id: number): string {
        ids.push(id);
        const out = tail();
        if (out === '' || out.endsWith('\uFFFD')) return '';
        prefix = read;
        read = ids.length;
        return out;
      },
      flush(): string {
        const out = tail();
        ids = [];
        prefix = read = 0;
        return out;
      },
    };
  };

  const need = (tok: string): number => {
    const id = vocab.get(tok);
    if (id === undefined) throw bad(`vocab has no ${tok}`);
    return id;
  };
  let chatFormat = opts.chatFormat;
  if (!chatFormat) {
    if (vocab.has('<start_of_turn>')) chatFormat = 'gemma';
    else if (vocab.has('<|im_start|>')) chatFormat = 'qwen3';
    else throw bad('unknown chat format (no <start_of_turn> or <|im_start|>)');
  }
  const gemma = chatFormat === 'gemma';
  const bosId = gemma ? need('<bos>') : undefined;
  const eosId = need(gemma ? '<eos>' : '<|endoftext|>');
  const endOfTurnId = need(gemma ? '<end_of_turn>' : '<|im_end|>');
  const format = gemma ? formatGemmaChat : formatQwen3Chat;

  return {
    chatFormat,
    vocabSize,
    bosId,
    eosId,
    endOfTurnId,
    stopIds: Object.freeze([endOfTurnId, eosId]),
    encode(text, o = {}) {
      const ids = tokenizer.encode(text, { add_special_tokens: false }).ids;
      return o.addBos && bosId !== undefined ? [bosId, ...ids] : ids;
    },
    decode,
    createStreamDecoder,
    tokenToId: (t) => vocab.get(t),
    idToToken: (id) => tokenizer.id_to_token(id),
    isSpecial: (id) => special.has(id),
    applyChatTemplate: (messages, o) => format(messages, o),
  };
}

/** The Gemma 3 chat template (HF tokenizer_config.json), with lenient role
 *  handling: a system message becomes the first user turn's prefix,
 *  consecutive same-role turns are merged, and any role other than
 *  assistant/model is treated as user. */
export function formatGemmaChat(
  messages: readonly ChatMessage[],
  opts: ChatTemplateOptions = {},
): string {
  const system: string[] = [];
  const turns: { role: 'user' | 'model'; content: string }[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      if (turns.length === 0) system.push(m.content.trim());
      else turns.push({ role: 'user', content: m.content.trim() });
      continue;
    }
    const role = m.role === 'assistant' || m.role === 'model' ? 'model' : 'user';
    const last = turns[turns.length - 1];
    if (last && last.role === role) last.content += `\n\n${m.content.trim()}`;
    else turns.push({ role, content: m.content.trim() });
  }
  if (turns.length === 0 || turns[0]!.role !== 'user') turns.unshift({ role: 'user', content: '' });
  if (system.length > 0) {
    const first = turns[0]!;
    first.content = `${system.join('\n\n')}\n\n${first.content}`;
  }
  let out = '<bos>';
  for (const t of turns) out += `<start_of_turn>${t.role}\n${t.content}<end_of_turn>\n`;
  if (opts.addGenerationPrompt ?? true) out += '<start_of_turn>model\n';
  return out;
}

/** Python's json.dumps defaults (', ' and ': ' separators, non-ASCII kept),
 *  which the HF template's `tojson` produces. */
function pyJson(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (Array.isArray(v)) return `[${v.map(pyJson).join(', ')}]`;
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined);
    return `{${entries.map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(', ')}}`;
  }
  return JSON.stringify(v);
}

const isToolResponse = (s: string) =>
  s.startsWith('<tool_response>') && s.endsWith('</tool_response>');

/** The Qwen 3 chat template (HF tokenizer_config.json) for string contents:
 *  ChatML turns; a leading system message, with the tool list when `tools`
 *  is given; assistant turns before the last user query stripped of their
 *  `<think>` part; consecutive `tool` messages grouped into one user turn of
 *  `<tool_response>` blocks; and, unless `enableThinking`, an empty think
 *  block after the generation prompt. Role `model` is read as assistant. */
export function formatQwen3Chat(
  messages: readonly ChatMessage[],
  opts: ChatTemplateOptions = {},
): string {
  const msgs = messages.map((m) => (m.role === 'model' ? { ...m, role: 'assistant' } : m));
  const first = msgs[0];
  const leadingSystem = first?.role === 'system';
  let out = '';
  if (opts.tools && opts.tools.length > 0) {
    out += '<|im_start|>system\n';
    if (leadingSystem) out += `${first.content}\n\n`;
    out +=
      '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' +
      'You are provided with function signatures within <tools></tools> XML tags:\n<tools>';
    for (const t of opts.tools) out += `\n${pyJson(t)}`;
    out +=
      '\n</tools>\n\nFor each function call, return a json object with function name and ' +
      'arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n' +
      '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n';
  } else if (leadingSystem) {
    out += `<|im_start|>system\n${first.content}<|im_end|>\n`;
  }
  let lastQuery = msgs.length - 1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]!;
    if (m.role === 'user' && !isToolResponse(m.content)) {
      lastQuery = i;
      break;
    }
  }
  msgs.forEach((m, i) => {
    let content = m.content;
    if (m.role === 'user' || (m.role === 'system' && i > 0)) {
      out += `<|im_start|>${m.role}\n${content}<|im_end|>\n`;
    } else if (m.role === 'assistant') {
      let reasoning = '';
      if (content.includes('</think>')) {
        const parts = content.split('</think>');
        reasoning = parts[0]!.replace(/\n+$/, '').split('<think>').pop()!.replace(/^\n+/, '');
        content = parts[parts.length - 1]!.replace(/^\n+/, '');
      }
      if (i > lastQuery && (i === msgs.length - 1 || reasoning)) {
        const r = reasoning.replace(/^\n+|\n+$/g, '');
        out += `<|im_start|>assistant\n<think>\n${r}\n</think>\n\n${content.replace(/^\n+/, '')}`;
      } else {
        out += `<|im_start|>assistant\n${content}`;
      }
      out += '<|im_end|>\n';
    } else if (m.role === 'tool') {
      if (i === 0 || msgs[i - 1]!.role !== 'tool') out += '<|im_start|>user';
      out += `\n<tool_response>\n${content}\n</tool_response>`;
      if (i === msgs.length - 1 || msgs[i + 1]!.role !== 'tool') out += '<|im_end|>\n';
    }
  });
  if (opts.addGenerationPrompt ?? true) {
    out += '<|im_start|>assistant\n';
    if (!opts.enableThinking) out += '<think>\n\n</think>\n\n';
  }
  return out;
}
