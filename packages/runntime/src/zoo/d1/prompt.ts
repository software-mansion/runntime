/** d1 questions to token sequences and probabilities back to answers, ported
 *  from the checkpoint's prompt.py. One question over one state renders as
 *
 *      <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> … <decide>
 *
 *  and the head scores the hidden state at each `<mask>`. */

/** Question type to the head's type-embedding row. */
export const QTYPES = { choice: 0, score: 1, noul: 2 } as const;
export type QuestionType = keyof typeof QTYPES;

/** A question in the Decision Index schema. */
export type Question =
  | { type: 'noul'; instructions: string; criteria?: { true?: string; false?: string } }
  | { type: 'choice'; instructions: string; criteria: Record<string, string> }
  | { type: 'score'; instructions: string; criteria: readonly string[] };

export type Answer =
  | { type: 'noul'; noul: number }
  | { type: 'choice'; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: 'score'; score: number; confidence: number; probabilities: Record<string, number> };

/** The tokenizer surface the encoder needs. */
export interface D1Tokenizer {
  encode(text: string): number[];
  tokenId(token: string): number;
  bosId: number;
}

const DELIM = {
  state: '<|reserved_7|>',
  q: '<|reserved_8|>',
  opt: '<|reserved_9|>',
  optEnd: '<|reserved_10|>',
  decide: '<|reserved_11|>',
};
const MARKER = '<|mask|>';
/** Tokens each option text may take before it is cut. */
const PER_OPTION = 24;
/** How image questions were trained: noul options read as plain yes/no. */
export const YES_NO = { false: 'no', true: 'yes' } as const;

/** `<|name|>` to `<¦name¦>`, so caller text cannot forge a delimiter. */
const escape = (text: string): string => text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>');

export const optionCount = (q: Question): number =>
  q.type === 'noul' ? 2 : q.type === 'choice' ? Object.keys(q.criteria).length : q.criteria.length;

/** Option texts in the model's order; a noul reads as [false, true]. */
function renderOptions(q: Question, noulDefault?: { false: string; true: string }): string[] {
  if (q.type === 'choice') {
    return Object.entries(q.criteria).map(([k, v]) => (v ? `${k}: ${v}` : k));
  }
  if (q.type === 'score') return q.criteria.map((c, i) => `level ${i}: ${c}`);
  const crit = q.criteria ?? noulDefault ?? {};
  return [
    'false: ' + (crit.false || 'no, the statement does not hold'),
    'true: ' + (crit.true || 'yes, the statement holds'),
  ];
}

/** Token ids of one question over one state, and each option marker's
 *  position. The option block gets max(96, min(24k + 32, maxLen / 2)) tokens;
 *  the state is cut on the right to what is left. */
export function encodeQuestion(
  tok: D1Tokenizer,
  state: string,
  q: Question,
  maxLen: number,
  noulDefault?: { false: string; true: string },
): { ids: number[]; markers: number[] } {
  const enc = (s: string) => tok.encode(escape(s));
  const opts = renderOptions(q, noulDefault);
  const budget = Math.max(96, Math.min(opts.length * PER_OPTION + 32, Math.floor(maxLen / 2)));
  const per = Math.max(2, Math.floor((budget - 3 * opts.length) / opts.length));
  const question = [tok.tokenId(DELIM.q), ...enc(q.instructions)].slice(0, Math.max(16, budget));
  const markers: number[] = [];
  for (const text of opts) {
    markers.push(question.length + 1);
    question.push(
      tok.tokenId(DELIM.opt),
      tok.tokenId(MARKER),
      ...enc(' ' + text).slice(0, per),
      tok.tokenId(DELIM.optEnd),
    );
  }
  question.push(tok.tokenId(DELIM.decide));
  const room = Math.max(0, maxLen - question.length - 2);
  const stateIds = [tok.tokenId(DELIM.state), ...enc(state).slice(0, room)];
  const ids = [tok.bosId, ...stateIds, ...question].slice(0, maxLen);
  const shifted = markers.map((m) => m + 1 + stateIds.length);
  if (shifted[shifted.length - 1]! >= maxLen)
    throw new Error('the options do not fit in the context');
  return { ids, markers: shifted };
}

/** Temperature-table key for a question with k options. */
export function temperatureKey(q: Question): string {
  const k = optionCount(q);
  return `${q.type}:` + (k <= 2 ? '2' : k <= 5 ? '3-5' : k <= 10 ? '6-10' : '11+');
}

/** Softmax over the option logits, divided by `temperature` first. */
export function softmaxT(logits: readonly number[], temperature: number): number[] {
  const z = logits.map((l) => l / temperature);
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** Probabilities in option order (noul as [yes, no]) to the typed answer. */
export function toAnswer(q: Question, probs: readonly number[]): Answer {
  if (q.type === 'noul') return { type: 'noul', noul: probs[0]! };
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i]! > probs[best]!) best = i;
  if (q.type === 'choice') {
    const names = Object.keys(q.criteria);
    return {
      type: 'choice',
      choice: names[best]!,
      confidence: probs[best]!,
      probabilities: Object.fromEntries(names.map((n, i) => [n, probs[i]!])),
    };
  }
  return {
    type: 'score',
    score: probs.reduce((s, p, i) => s + i * p, 0),
    confidence: probs[best]!,
    probabilities: Object.fromEntries(probs.map((p, i) => [String(i), p])),
  };
}
