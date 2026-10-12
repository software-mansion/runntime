/** Gemma BPE tokenizer, encode-only: text → <bos> pieces <eos> ids,
 *  truncated to the asset's maxTokens. Ports HF tokenizers for Gemma's
 *  tokenizer.json: special tokens match in the raw text first; every other
 *  run has its spaces replaced by `▁` and is one BPE word; characters
 *  outside the vocab fall back to their UTF-8 bytes (`<0xNN>`). Merges apply
 *  lowest rank first, ties to the leftmost pair, as HF's merge loop does. */

import { checkpointMismatch } from '../errors.ts';
import { EMBEDDINGGEMMA2_MAX_TOKENS } from './config.ts';

export interface EmbeddingGemma2TokenizerAsset {
  /** Token string to id. */
  vocab: ReadonlyMap<string, number>;
  /** Merge pairs, rank order. */
  merges: readonly (readonly [string, string])[];
  /** Special tokens matched verbatim in the input. */
  special: readonly { content: string; id: number }[];
  bosId: number;
  eosId: number;
  maxTokens: number;
}

export interface EmbeddingGemma2Tokenizer {
  encode(text: string): number[];
  readonly bosId: number;
  readonly eosId: number;
}

interface HfTokenizerJson {
  added_tokens: { id: number; content: string; special: boolean }[];
  model: {
    type: string;
    byte_fallback: boolean;
    vocab: Record<string, number>;
    merges: [string, string][];
  };
}

const isHfTokenizerJson = (json: unknown): json is HfTokenizerJson =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  'added_tokens' in json &&
  Array.isArray(json.added_tokens);

/** Reads a Hugging Face tokenizer.json of the Gemma BPE kind. `maxTokens`
 *  defaults to EMBEDDINGGEMMA2_MAX_TOKENS. */
export function embeddingGemma2TokenizerAsset(
  json: unknown,
  maxTokens = EMBEDDINGGEMMA2_MAX_TOKENS,
): EmbeddingGemma2TokenizerAsset {
  const bad = (why: string) => checkpointMismatch('embeddinggemma2 tokenizer', why);
  if (!isHfTokenizerJson(json)) throw bad('not a tokenizer.json');
  const { model } = json;
  if (model.type !== 'BPE' || !model.byte_fallback || !Array.isArray(model.merges)) {
    throw bad('expected a BPE model with byte fallback and [a, b] merges');
  }
  const vocab = new Map(Object.entries(model.vocab));
  for (let b = 0; b < 256; b++) {
    if (!vocab.has(byteToken(b))) throw bad(`vocab has no ${byteToken(b)}`);
  }
  const idOf = (token: string): number => {
    const id = vocab.get(token);
    if (id === undefined) throw bad(`vocab has no ${token}`);
    return id;
  };
  return {
    vocab,
    merges: model.merges,
    special: json.added_tokens.filter((t) => t.special).map(({ content, id }) => ({ content, id })),
    bosId: idOf('<bos>'),
    eosId: idOf('<eos>'),
    maxTokens,
  };
}

const byteToken = (b: number) => `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`;

/** Min-heap of candidate merges by (rank, position). */
class MergeHeap {
  private readonly rank: number[] = [];
  private readonly pos: number[] = [];
  private readonly id: number[] = [];
  get size(): number {
    return this.rank.length;
  }
  private less(i: number, j: number): boolean {
    return (
      this.rank[i]! < this.rank[j]! ||
      (this.rank[i] === this.rank[j] && this.pos[i]! < this.pos[j]!)
    );
  }
  private swap(i: number, j: number): void {
    for (const a of [this.rank, this.pos, this.id]) [a[i], a[j]] = [a[j]!, a[i]!];
  }
  push(rank: number, pos: number, id: number): void {
    this.rank.push(rank);
    this.pos.push(pos);
    this.id.push(id);
    for (let i = this.size - 1; i > 0;) {
      const parent = (i - 1) >> 1;
      if (!this.less(i, parent)) break;
      this.swap(i, parent);
      i = parent;
    }
  }
  /** Removes the smallest entry and returns its [pos, id]. */
  pop(): [number, number] {
    const top: [number, number] = [this.pos[0]!, this.id[0]!];
    const last = this.size - 1;
    this.swap(0, last);
    this.rank.pop();
    this.pos.pop();
    this.id.pop();
    for (let i = 0; ;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < this.size && this.less(l, m)) m = l;
      if (r < this.size && this.less(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m);
      i = m;
    }
    return top;
  }
}

export function createEmbeddingGemma2Tokenizer(
  asset: EmbeddingGemma2TokenizerAsset,
): EmbeddingGemma2Tokenizer {
  const { vocab, bosId, eosId, maxTokens } = asset;
  // (left id, right id) to merge rank and the merged token's id.
  const pairKey = (a: number, b: number) => a * vocab.size + b;
  const mergeRank = new Map<number, number>();
  const mergeId: number[] = [];
  asset.merges.forEach(([a, b], rank) => {
    const left = vocab.get(a);
    const right = vocab.get(b);
    const merged = vocab.get(a + b);
    if (left === undefined || right === undefined || merged === undefined) {
      throw checkpointMismatch('embeddinggemma2 tokenizer', `merge ${a} ${b} is not in the vocab`);
    }
    mergeRank.set(pairKey(left, right), rank);
    mergeId.push(merged);
  });
  const byteIds = Array.from({ length: 256 }, (_, b) => vocab.get(byteToken(b))!);
  const specialIds = new Map(asset.special.map((t) => [t.content, t.id]));
  // Longest first, so a token that starts another one wins.
  const specialRe = new RegExp(
    `(${[...specialIds.keys()]
      .sort((x, y) => y.length - x.length)
      .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('|')})`,
  );
  const utf8 = new TextEncoder();

  /** One BPE word to ids, appended to `out`. */
  const encodeWord = (word: string, out: number[]): void => {
    const ids: number[] = [];
    for (const ch of word) {
      const id = vocab.get(ch);
      if (id !== undefined) ids.push(id);
      else for (const b of utf8.encode(ch)) ids.push(byteIds[b]!);
    }
    // Symbols as a linked list; a merged-away symbol gets id -1.
    const next = ids.map((_, i) => (i + 1 < ids.length ? i + 1 : -1));
    const prev = ids.map((_, i) => i - 1);
    const heap = new MergeHeap();
    const offer = (pos: number) => {
      const right = next[pos]!;
      if (right < 0) return;
      const rank = mergeRank.get(pairKey(ids[pos]!, ids[right]!));
      if (rank !== undefined) heap.push(rank, pos, mergeId[rank]!);
    };
    for (let i = 0; i + 1 < ids.length; i++) offer(i);
    while (heap.size > 0) {
      const [pos, merged] = heap.pop();
      // Skip entries a merge since made stale.
      const right = next[pos]!;
      if (ids[pos]! < 0 || right < 0) continue;
      const rank = mergeRank.get(pairKey(ids[pos]!, ids[right]!));
      if (rank === undefined || mergeId[rank] !== merged) continue;
      ids[pos] = merged;
      ids[right] = -1;
      next[pos] = next[right]!;
      if (next[pos]! >= 0) prev[next[pos]!] = pos;
      if (prev[pos]! >= 0) offer(prev[pos]!);
      offer(pos);
    }
    for (const id of ids) if (id >= 0) out.push(id);
  };

  return {
    bosId,
    eosId,
    encode(text) {
      const pieces: number[] = [];
      for (const part of text.split(specialRe)) {
        if (part === '') continue;
        const special = specialIds.get(part);
        if (special !== undefined) pieces.push(special);
        else encodeWord(part.replaceAll(' ', '▁'), pieces);
      }
      return [bosId, ...pieces.slice(0, maxTokens - 2), eosId];
    },
  };
}
