/** Decode-only detokenizer for Moonshine's Llama-style BPE vocabulary. STT
 *  never encodes text — the decoder starts from BOS and emits ids, so id →
 *  string is the whole job: special ids skipped, `▁` → space, byte-fallback
 *  tokens (`<0xNN>`) accumulated per run and decoded as UTF-8 (multibyte
 *  characters span several byte tokens), one leading space stripped (the
 *  sentencepiece prefix on the first word). The asset comes from
 *  moonshineTokenizerAsset, read from a Hub tokenizer.json or the compact
 *  file tools/export_moonshine_tokenizer.py writes. */

export interface MoonshineTokenizer {
  version: number;
  bosId: number;
  eosId: number;
  specialIds: number[];
  vocab: string[];
}

const BYTE_TOKEN = /^<0x([0-9A-Fa-f]{2})>$/;

export function decodeTokens(tok: MoonshineTokenizer, ids: readonly number[]): string {
  const specials = new Set(tok.specialIds);
  const utf8 = new TextDecoder('utf-8');
  let out = '';
  let bytes: number[] = [];
  const flushBytes = () => {
    if (bytes.length === 0) return;
    out += utf8.decode(Uint8Array.from(bytes));
    bytes = [];
  };
  for (const id of ids) {
    if (specials.has(id)) continue;
    const token = tok.vocab[id];
    if (token === undefined) throw new Error(`decodeTokens: id ${id} outside vocab`);
    const byte = BYTE_TOKEN.exec(token);
    if (byte) {
      bytes.push(parseInt(byte[1]!, 16));
      continue;
    }
    flushBytes();
    out += token.replaceAll('▁', ' ');
  }
  flushBytes();
  return out.startsWith(' ') ? out.slice(1) : out;
}

interface HfTokenizerJson {
  model: { type: string; vocab: Record<string, number> };
  added_tokens?: { id: number; content: string; special: boolean }[];
}

export function moonshineTokenizerAsset(json: unknown): MoonshineTokenizer {
  const bad = (why: string) => new Error(`moonshine tokenizer: ${why}`);
  if (typeof json !== 'object' || json === null) throw bad('not a JSON object');

  if ('vocab' in json && Array.isArray(json.vocab)) {
    const asset = json as MoonshineTokenizer;
    if (asset.version !== 1) throw bad(`unsupported asset version ${asset.version}`);
    return asset;
  }

  const hf = json as Partial<HfTokenizerJson>;
  if (hf.model?.type !== 'BPE' || typeof hf.model.vocab !== 'object') {
    throw bad('expected a BPE tokenizer.json or a version 1 asset');
  }
  const added = hf.added_tokens ?? [];
  const entries = [
    ...Object.entries(hf.model.vocab),
    ...added.map((t) => [t.content, t.id] as const),
  ];
  let size = 0;
  for (const [, id] of entries) size = Math.max(size, id + 1);
  const vocab = new Array<string>(size).fill('');
  for (const [token, id] of entries) vocab[id] = token;
  const idOf = (content: string): number => {
    const id = added.find((t) => t.content === content)?.id ?? hf.model!.vocab[content];
    if (id === undefined) throw bad(`vocab has no ${content}`);
    return id;
  };
  return {
    version: 1,
    bosId: idOf('<s>'),
    eosId: idOf('</s>'),
    specialIds: added.filter((t) => t.special).map((t) => t.id),
    vocab,
  };
}
