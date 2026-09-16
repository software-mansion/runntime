/** BERT WordPiece tokenizer, encode-only — text → [CLS] pieces [SEP] ids,
 *  truncated to the asset's maxWordpieces. Ports HF BertTokenizer's two-stage
 *  algorithm: BasicTokenizer (clean, CJK-ideograph spacing, whitespace split,
 *  lowercase, NFD accent strip, punctuation split) then greedy longest-match
 *  WordPiece with `##` continuations. The asset comes from
 *  minilmTokenizerAsset, read from a Hub tokenizer.json or the compact file
 *  tools/export_minilm_tokenizer.py writes. Parity cases in
 *  tokenizer_cases.json pin the exact ids per input. */

export interface MinilmTokenizerAsset {
  version: number;
  doLowerCase: boolean;
  maxWordpieces: number;
  clsId: number;
  sepId: number;
  padId: number;
  unkId: number;
  vocab: string[];
}

export interface MinilmTokenizer {
  encode(text: string): number[];
  readonly clsId: number;
  readonly sepId: number;
}

export const MINILM_MAX_WORDPIECES = 256;

interface HfTokenizerJson {
  normalizer?: { lowercase?: boolean } | null;
  model: { type: string; vocab: Record<string, number> };
}

export function minilmTokenizerAsset(json: unknown, maxTokens?: number): MinilmTokenizerAsset {
  const bad = (why: string) => new Error(`minilm tokenizer: ${why}`);
  if (typeof json !== 'object' || json === null) throw bad('not a JSON object');

  if ('vocab' in json && Array.isArray(json.vocab)) {
    const asset = json as MinilmTokenizerAsset;
    if (asset.version !== 1) throw bad(`unsupported asset version ${asset.version}`);
    return maxTokens === undefined ? asset : { ...asset, maxWordpieces: maxTokens };
  }

  const hf = json as Partial<HfTokenizerJson>;
  if (hf.model?.type !== 'WordPiece' || typeof hf.model.vocab !== 'object') {
    throw bad('expected a WordPiece tokenizer.json or a version 1 asset');
  }
  const entries = Object.entries(hf.model.vocab);
  const vocab = new Array<string>(entries.length);
  for (const [token, id] of entries) vocab[id] = token;
  const idOf = (token: string): number => {
    const id = hf.model!.vocab[token];
    if (id === undefined) throw bad(`vocab has no ${token}`);
    return id;
  };
  return {
    version: 1,
    doLowerCase: hf.normalizer?.lowercase ?? true,
    maxWordpieces: maxTokens ?? MINILM_MAX_WORDPIECES,
    clsId: idOf('[CLS]'),
    sepId: idOf('[SEP]'),
    padId: idOf('[PAD]'),
    unkId: idOf('[UNK]'),
    vocab,
  };
}

const MAX_WORD_CHARS = 100;

const isWhitespace = (ch: string): boolean =>
  ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r' || /\p{Zs}/u.test(ch);

const isControl = (ch: string): boolean =>
  ch !== '\t' && ch !== '\n' && ch !== '\r' && /\p{C}/u.test(ch);

const isPunctuation = (ch: string): boolean => {
  const c = ch.codePointAt(0)!;
  if (
    (c >= 33 && c <= 47) ||
    (c >= 58 && c <= 64) ||
    (c >= 91 && c <= 96) ||
    (c >= 123 && c <= 126)
  )
    return true;
  return /\p{P}/u.test(ch);
};

const isCjkIdeograph = (c: number): boolean =>
  (c >= 0x4e00 && c <= 0x9fff) ||
  (c >= 0x3400 && c <= 0x4dbf) ||
  (c >= 0x20000 && c <= 0x2a6df) ||
  (c >= 0x2a700 && c <= 0x2b73f) ||
  (c >= 0x2b740 && c <= 0x2b81f) ||
  (c >= 0x2b820 && c <= 0x2ceaf) ||
  (c >= 0xf900 && c <= 0xfaff) ||
  (c >= 0x2f800 && c <= 0x2fa1f);

function basicTokenize(text: string, doLowerCase: boolean): string[] {
  let cleaned = '';
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (c === 0 || c === 0xfffd || isControl(ch)) continue;
    if (isCjkIdeograph(c)) cleaned += ` ${ch} `;
    else cleaned += isWhitespace(ch) ? ' ' : ch;
  }

  const words: string[] = [];
  for (const token of cleaned.split(' ')) {
    if (!token) continue;
    let t = token;
    if (doLowerCase) {
      // Lowercase first, then NFD + drop combining marks — HF's order.
      t = t
        .toLowerCase()
        .normalize('NFD')
        .replace(/\p{Mn}/gu, '');
    }
    let current = '';
    for (const ch of t) {
      if (isPunctuation(ch)) {
        if (current) words.push(current);
        words.push(ch);
        current = '';
      } else {
        current += ch;
      }
    }
    if (current) words.push(current);
  }
  return words;
}

export function createMinilmTokenizer(asset: MinilmTokenizerAsset): MinilmTokenizer {
  if (asset.version !== 1) {
    throw new Error(`minilm tokenizer: unsupported asset version ${asset.version}`);
  }
  const tokenToId = new Map<string, number>();
  asset.vocab.forEach((token, id) => tokenToId.set(token, id));

  const wordpiece = (word: string, out: number[]): void => {
    const chars = [...word];
    if (chars.length > MAX_WORD_CHARS) {
      out.push(asset.unkId);
      return;
    }
    const pieces: number[] = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let id: number | undefined;
      while (end > start) {
        const piece = (start > 0 ? '##' : '') + chars.slice(start, end).join('');
        id = tokenToId.get(piece);
        if (id !== undefined) break;
        end--;
      }
      if (id === undefined) {
        out.push(asset.unkId);
        return;
      }
      pieces.push(id);
      start = end;
    }
    out.push(...pieces);
  };

  return {
    clsId: asset.clsId,
    sepId: asset.sepId,
    encode(text: string): number[] {
      const pieces: number[] = [];
      for (const word of basicTokenize(text, asset.doLowerCase)) wordpiece(word, pieces);
      const body = pieces.slice(0, asset.maxWordpieces - 2);
      return [asset.clsId, ...body, asset.sepId];
    },
  };
}
