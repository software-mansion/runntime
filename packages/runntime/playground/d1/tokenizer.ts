/** transformers.js tokenizer adapted to the d1 prompt encoder. */
import { AutoTokenizer } from '@huggingface/transformers';
import type { D1Tokenizer } from '../../src/zoo/d1/prompt.ts';

export async function loadD1Tokenizer(repo = 'LiquidAI/d1-omni-600M'): Promise<D1Tokenizer> {
  const tok = await AutoTokenizer.from_pretrained(repo);
  const id = (t: string): number => {
    const v = tok.convert_tokens_to_ids(t);
    if (typeof v !== 'number') throw new Error(`no id for ${t}`);
    return v;
  };
  return {
    encode: (text) => tok.encode(text, { add_special_tokens: false }),
    tokenId: id,
    bosId: id('<|startoftext|>'),
  };
}
