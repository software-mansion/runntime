import { Tiktoken } from 'js-tiktoken/lite';
import o200k_base from 'js-tiktoken/ranks/o200k_base';

export interface Tokenizer {
  encode(text: string): number[];
  tokenBytes(id: number): Uint8Array;
  tokenCharOffsets(ids: readonly number[]): {
    text: string;
    charStarts: number[];
    charEnds: number[];
  };
  readonly eotToken: number;
}

export function createTokenizer(): Tokenizer {
  const enc = new Tiktoken(o200k_base);
  const idToBytes = new Map<number, Uint8Array>();
  for (const line of o200k_base.bpe_ranks.split('\n')) {
    if (!line) continue;
    const [, offsetStr, ...tokens] = line.split(' ');
    const offset = Number.parseInt(offsetStr!, 10);
    tokens.forEach((tok, i) => {
      idToBytes.set(
        offset + i,
        Uint8Array.from(atob(tok), (c) => c.charCodeAt(0)),
      );
    });
  }
  const textEncoder = new TextEncoder();
  for (const [text, rank] of Object.entries(o200k_base.special_tokens)) {
    idToBytes.set(rank, textEncoder.encode(text));
  }

  const tokenBytes = (id: number): Uint8Array => {
    const bytes = idToBytes.get(id);
    if (!bytes) throw new Error(`Unknown token id ${id}`);
    return bytes;
  };

  return {
    encode: (text) => enc.encode(text, 'all'),
    tokenBytes,
    eotToken: o200k_base.special_tokens['<|endoftext|>']!,
    tokenCharOffsets(ids) {
      const pieces = ids.map(tokenBytes);
      let total = 0;
      for (const p of pieces) total += p.length;
      const merged = new Uint8Array(total);
      let cursor = 0;
      for (const p of pieces) {
        merged.set(p, cursor);
        cursor += p.length;
      }
      const text = new TextDecoder('utf-8').decode(merged);

      // Byte start/end per UTF-16 index (surrogate pairs share the pair's bytes).
      const charByteStarts: number[] = [];
      const charByteEnds: number[] = [];
      let byteCursor = 0;
      for (const ch of text) {
        const byteLen = textEncoder.encode(ch).length;
        for (let u = 0; u < ch.length; u++) {
          charByteStarts.push(byteCursor);
          charByteEnds.push(byteCursor + byteLen);
        }
        byteCursor += byteLen;
      }

      const bisectRight = (arr: number[], x: number) => {
        let lo = 0,
          hi = arr.length;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (arr[mid]! <= x) lo = mid + 1;
          else hi = mid;
        }
        return lo;
      };
      const bisectLeft = (arr: number[], x: number) => {
        let lo = 0,
          hi = arr.length;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (arr[mid]! < x) lo = mid + 1;
          else hi = mid;
        }
        return lo;
      };

      const charStarts: number[] = [];
      const charEnds: number[] = [];
      let tokenByteCursor = 0;
      for (const raw of pieces) {
        const tokenByteStart = tokenByteCursor;
        const tokenByteEnd = tokenByteStart + raw.length;
        tokenByteCursor = tokenByteEnd;
        const startIdx = bisectRight(charByteEnds, tokenByteStart);
        let endIdx = bisectLeft(charByteStarts, tokenByteEnd);
        if (endIdx < startIdx) endIdx = startIdx;
        charStarts.push(startIdx);
        charEnds.push(endIdx);
      }
      return { text, charStarts, charEnds };
    },
  };
}
