/** Times systemOne and reports input tokens per second. d1 generates no
 *  tokens: each question is one trunk pass over the input. */
import type { Question } from '../../src/zoo/d1/prompt.ts';
import { loadDecider } from './shared.ts';

const out = document.getElementById('out')!;
const log = (s: string) => {
  out.textContent += s + '\n';
  console.log(s);
};

async function loadRgb(url: string) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const data = new Uint8Array(bmp.width * bmp.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    data[j] = rgba[i]!;
    data[j + 1] = rgba[i + 1]!;
    data[j + 2] = rgba[i + 2]!;
  }
  return { width: bmp.width, height: bmp.height, data };
}

const question: Question = {
  type: 'choice',
  instructions: 'Which team should handle this?',
  criteria: {
    billing: 'Charges, refunds, invoices',
    technical: 'App or site faults',
    fraud: 'Suspected unauthorised use',
  },
};
const sentence = 'I was charged twice this month, please refund one of them. ';

const WARMUP = 2;
const RUNS = 5;

async function main() {
  const decider = await loadDecider(() => {});
  const img = await loadRgb('/cats.jpg');
  const cases: [string, () => ReturnType<typeof decider.systemOne>][] = [
    ['text, short', () => decider.systemOne(sentence, { q: question })],
    ['text, ~500 tok', () => decider.systemOne(sentence.repeat(36), { q: question })],
    ['text, ~2000 tok', () => decider.systemOne(sentence.repeat(145), { q: question })],
    ['text, ~3500 tok', () => decider.systemOne(sentence.repeat(250), { q: question })],
    [
      'image + question',
      () => decider.systemOne('', { q: { type: 'noul', instructions: 'Is there a cat?' } }, [img]),
    ],
  ];
  const rows: Record<string, unknown>[] = [];
  for (const [name, run] of cases) {
    for (let i = 0; i < WARMUP; i++) await run();
    const times: number[] = [];
    let tokens = 0;
    for (let i = 0; i < RUNS; i++) {
      const t = performance.now();
      tokens = (await run()).inputTokens;
      times.push(performance.now() - t);
    }
    times.sort((a, b) => a - b);
    const median = times[Math.floor(RUNS / 2)]!;
    const row = {
      name,
      tokens,
      medianMs: +median.toFixed(1),
      tokPerSec: Math.round((tokens / median) * 1000),
    };
    rows.push(row);
    log(JSON.stringify(row));
  }
  (window as unknown as { __bench: unknown }).__bench = rows;
  log('DONE');
}

main().catch((err) => log(`ERROR ${err instanceof Error ? err.stack : String(err)}`));
