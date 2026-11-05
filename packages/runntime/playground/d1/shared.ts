/** Model loading and answer rendering shared by the playground pages. */
import tgpu from 'typegpu';
import { fromSafetensors, initRunntime } from '../../src/core/index.ts';
import { createD1Decider, type D1Decider } from '../../src/zoo/d1/decider.ts';
import type { RgbImage } from '../../src/zoo/d1/image.ts';
import type { Answer } from '../../src/zoo/d1/prompt.ts';
import { loadD1Tokenizer } from './tokenizer.ts';

/** config.json's text temperatures. */
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

export async function loadDecider(status: (s: string) => void): Promise<D1Decider> {
  if (!navigator.gpu) throw new Error('no WebGPU in this browser');
  initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
  status('loading tokenizer…');
  const tokenizer = await loadD1Tokenizer();
  return createD1Decider(await fromSafetensors('/model.safetensors'), tokenizer, {
    temperatures: TEMPERATURES,
    onProgress: (_, done, total) => status(`loading model ${Math.floor((100 * done) / total)}%`),
  });
}

/** Pixels of a canvas region as RGB bytes. */
export function canvasRgb(ctx: CanvasRenderingContext2D, w: number, h: number): RgbImage {
  const { data } = ctx.getImageData(0, 0, w, h);
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i]!;
    rgb[j + 1] = data[i + 1]!;
    rgb[j + 2] = data[i + 2]!;
  }
  return { width: w, height: h, data: rgb };
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

function bar(label: string, p: number, strong: boolean): string {
  return `<div class="bar${strong ? ' top' : ''}"><span>${esc(label)}</span><i style="--p:${(p * 100).toFixed(1)}%"></i><b>${(p * 100).toFixed(1)}%</b></div>`;
}

/** One answer card: the question name, the verdict and its bars. */
export function renderAnswer(name: string, a: Answer, legend?: readonly string[]): string {
  if (a.type === 'noul') {
    const yes = a.noul >= 0.5;
    return `<section><h3>${esc(name)} <em class="${yes ? 'yes' : 'no'}">${yes ? 'yes' : 'no'}</em></h3>${bar('yes', a.noul, yes)}</section>`;
  }
  const entries = Object.entries(a.probabilities);
  const best = a.type === 'choice' ? a.choice : String(Math.round(a.score));
  const verdict = a.type === 'choice' ? a.choice : `${a.score.toFixed(2)}`;
  return `<section><h3>${esc(name)} <em>${esc(verdict)}</em></h3>${entries
    .map(([k, p]) =>
      bar(a.type === 'score' && legend ? `${k}: ${legend[Number(k)]}` : k, p, k === best),
    )
    .join('')}</section>`;
}

export const STYLE = `
  :root { color-scheme: dark; }
  body { font: 14px system-ui, sans-serif; background: #17171b; color: #eee; margin: 0; padding: 24px; }
  a { color: #8cf; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #999; margin: 0 0 20px; }
  section { background: #222228; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
  h3 { margin: 0 0 8px; font-size: 14px; display: flex; justify-content: space-between; }
  em { font-style: normal; color: #8cf; }
  em.yes { color: #5d5; } em.no { color: #e66; }
  .bar { display: grid; grid-template-columns: minmax(60px, 180px) 1fr 52px; gap: 8px; align-items: center; color: #aaa; font-size: 13px; }
  .bar span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar i { height: 10px; background: linear-gradient(90deg, #4a8 var(--p), #333 var(--p)); border-radius: 5px; }
  .bar.top { color: #fff; }
  .bar b { font-weight: normal; text-align: right; }
  .status { color: #999; font-size: 13px; min-height: 18px; }
`;
