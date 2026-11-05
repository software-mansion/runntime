/** Compares the TS prompt encoding with the PyTorch reference's token ids. */
import { readFileSync } from 'node:fs';
import { encodeQuestion, YES_NO, type Question } from '../../src/zoo/d1/prompt.ts';
import { loadD1Tokenizer } from './tokenizer.ts';

const ref = JSON.parse(readFileSync(process.argv[2]!, 'utf8'));
const tok = await loadD1Tokenizer();

const textQs: Question[] = [
  { type: 'noul', instructions: 'Is the customer asking for a refund?' },
  {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: {
      billing: 'Charges, refunds, invoices',
      technical: 'App or site faults',
      fraud: 'Suspected unauthorised use',
    },
  },
  {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['Can wait', 'Today', 'Blocking the customer now'],
  },
];
const imageQs: Question[] = [
  {
    type: 'choice',
    instructions: 'How many cats are there?',
    criteria: { one: 'One', two: 'Two', more: 'Three or more' },
  },
  { type: 'noul', instructions: 'Is there a dog in the picture?' },
];
const state = 'I was charged twice this month, please refund one of them.';

let ok = true;
const check = (
  label: string,
  got: { ids: number[]; markers: number[] },
  ids: number[],
  markers: number[],
) => {
  const same =
    JSON.stringify(got.ids) === JSON.stringify(ids) &&
    JSON.stringify(got.markers) === JSON.stringify(markers);
  ok &&= same;
  console.log(label, same ? 'match' : `MISMATCH\n got ${got.ids}\nwant ${ids}`);
};
textQs.forEach((q, i) =>
  check(`text ${i}`, encodeQuestion(tok, state, q, 16384), ref.text.ids[i], ref.text.markers[i]),
);
imageQs.forEach((q, i) =>
  check(
    `image ${i}`,
    encodeQuestion(tok, '', q, 896, YES_NO),
    ref.image.ids[i],
    ref.image.markers[i],
  ),
);
process.exit(ok ? 0 : 1);
