/** Runs d1-omni-600M text + image on WebGPU and prints the answers. */
import tgpu from 'typegpu';
import { fromSafetensors, initRunntime } from '../../src/core/index.ts';
import { createD1Decider } from '../../src/zoo/d1/decider.ts';
import type { Question } from '../../src/zoo/d1/prompt.ts';
import { loadD1Tokenizer } from './tokenizer.ts';

const out = document.getElementById('out')!;
const log = (s: string) => {
  out.textContent += s + '\n';
  console.log(s);
};

/** Config.json's text temperatures. */
const TEMPERATURES = {
  'choice:11+': 1.372515082359314,
  'choice:2': 1.7465145587921143,
  'choice:3-5': 1.3998981714248657,
  'choice:6-10': 1.1751071214675903,
  'noul:2': 1.6663223505020142,
  'score:3-5': 1.7301132678985596,
  'score:6-10': 1.0,
  choice: 1.0,
  score: 1.0,
  noul: 1.0,
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

async function main() {
  initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
  const t0 = performance.now();
  const tokenizer = await loadD1Tokenizer();
  const sd = await fromSafetensors('/model.safetensors');
  let last = 0;
  const decider = await createD1Decider(sd, tokenizer, {
    temperatures: TEMPERATURES,
    onProgress: (_, done, total) => {
      const pct = Math.floor((100 * done) / total);
      if (pct >= last + 10) log(`loading ${(last = pct)}%`);
    },
  });
  log(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  const textQs: Record<string, Question> = {
    refund: { type: 'noul', instructions: 'Is the customer asking for a refund?' },
    team: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: {
        billing: 'Charges, refunds, invoices',
        technical: 'App or site faults',
        fraud: 'Suspected unauthorised use',
      },
    },
    urgency: {
      type: 'score',
      instructions: 'How urgent is this?',
      criteria: ['Can wait', 'Today', 'Blocking the customer now'],
    },
  };
  const imageQs: Record<string, Question> = {
    cats: {
      type: 'choice',
      instructions: 'How many cats are there?',
      criteria: { one: 'One', two: 'Two', more: 'Three or more' },
    },
    animal: { type: 'noul', instructions: 'Is there a dog in the picture?' },
  };

  const results: Record<string, unknown> = {};
  for (let run = 0; run < 2; run++) {
    let t = performance.now();
    results.text = await decider.systemOne(
      'I was charged twice this month, please refund one of them.',
      textQs,
    );
    log(`text (${(performance.now() - t).toFixed(0)} ms): ${JSON.stringify(results.text)}`);
    const img = await loadRgb('/cats.jpg');
    t = performance.now();
    results.image = await decider.systemOne('', imageQs, [img]);
    log(`image (${(performance.now() - t).toFixed(0)} ms): ${JSON.stringify(results.image)}`);
  }
  (window as unknown as { __d1: unknown }).__d1 = results;
  log('DONE');
}

main().catch((err) => {
  log(`ERROR ${err instanceof Error ? err.stack : String(err)}`);
});
