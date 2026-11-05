/** d1 watches the webcam: four yes/no questions about every frame, live. */
import type { Question } from '../../src/zoo/d1/prompt.ts';
import { canvasRgb, loadDecider } from './shared.ts';

/** Frame side fed to the model, px: the smallest crop LFM2-VL's resize keeps (64 tokens). */
const FRAME = 256;

const ICONS = {
  person: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>',
  phone: '<rect x="6" y="2" width="12" height="20" rx="2.5"/><path d="M11 18h2"/>',
  thumb:
    '<path d="M7 10v11H3V10h4zM7 10l4-8c1.7 0 3 1.3 3 3v4h5.5c1.2 0 2.1 1.1 1.9 2.3l-1.4 8c-.2 1-1 1.7-2 1.7H7"/>',
};

const CHECKS: { key: string; label: string; icon: keyof typeof ICONS; q: Question }[] = [
  {
    key: 'present',
    label: 'Someone in front of the camera',
    icon: 'person',
    q: { type: 'noul', instructions: 'Is there a person in the picture?' },
  },
  {
    key: 'phone',
    label: 'Holding a phone',
    icon: 'phone',
    q: { type: 'noul', instructions: 'Is someone holding a phone?' },
  },
  {
    key: 'thumb',
    label: 'Showing a thumbs up',
    icon: 'thumb',
    q: { type: 'noul', instructions: 'Is the person showing a thumbs up with their hand?' },
  },
];
const QUESTIONS = Object.fromEntries(CHECKS.map((c) => [c.key, c.q]));

document.head.insertAdjacentHTML(
  'beforeend',
  `<style>
  :root {
    color-scheme: dark;
    --bg: #0d0e12; --panel: #16181f; --line: #262a35; --text: #eef0f5; --dim: #8a90a0;
    --yes: #3ddc84; --no: #4a5060;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: radial-gradient(1200px 600px at 20% -10%, #1c2333, var(--bg));
    color: var(--text); font: 15px/1.4 Inter, system-ui, sans-serif; display: grid; place-items: center; padding: 32px 16px;
  }
  .app { width: min(1080px, 100%); }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }
  h1 span { color: var(--dim); font-weight: 400; }
  header a { color: var(--dim); font-size: 13px; text-decoration: none; }
  main { display: grid; grid-template-columns: 1.35fr 1fr; gap: 20px; }
  @media (max-width: 820px) { main { grid-template-columns: 1fr; } }
  .stage { position: relative; aspect-ratio: 1; border-radius: 20px; overflow: hidden; background: #000; border: 1px solid var(--line); }
  .stage video, .stage canvas { width: 100%; height: 100%; object-fit: cover; display: block; transform: scaleX(-1); }
  .stage canvas { transform: none; }
  .badge {
    position: absolute; top: 14px; left: 14px; display: flex; align-items: center; gap: 8px;
    background: rgba(0,0,0,.55); backdrop-filter: blur(8px); padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: .06em;
  }
  .badge i { width: 8px; height: 8px; border-radius: 50%; background: #ff4d4d; animation: pulse 1.6s infinite; }
  @keyframes pulse { 50% { opacity: .35; } }
  .stats { position: absolute; bottom: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between; font-size: 12px; color: #cfd3dc; }
  .stats span { background: rgba(0,0,0,.55); backdrop-filter: blur(8px); padding: 5px 10px; border-radius: 999px; }
  .loading { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; color: var(--dim); background: var(--panel); }
  .loading b { display: block; width: 220px; height: 4px; border-radius: 2px; background: var(--line); margin: 14px auto 0; overflow: hidden; }
  .loading b i { display: block; height: 100%; width: var(--p, 0%); background: var(--yes); transition: width .2s; }
  .checks { display: grid; gap: 12px; align-content: start; }
  .check {
    display: grid; grid-template-columns: 48px 1fr auto; align-items: center; gap: 14px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px;
    transition: border-color .25s, box-shadow .25s, background .25s;
  }
  .check.on { border-color: color-mix(in srgb, var(--yes) 55%, transparent); background: color-mix(in srgb, var(--yes) 7%, var(--panel)); box-shadow: 0 0 28px -8px color-mix(in srgb, var(--yes) 50%, transparent); }
  .icon { width: 48px; height: 48px; border-radius: 12px; display: grid; place-items: center; background: #1f232d; color: var(--dim); transition: color .25s, background .25s; }
  .check.on .icon { color: var(--yes); background: color-mix(in srgb, var(--yes) 14%, #1f232d); }
  .icon svg { width: 24px; height: 24px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
  .label { font-weight: 550; }
  .meter { margin-top: 8px; height: 6px; border-radius: 3px; background: var(--line); overflow: hidden; }
  .meter i { display: block; height: 100%; width: 0; border-radius: 3px; background: var(--no); transition: width .3s ease, background .25s; }
  .check.on .meter i { background: var(--yes); }
  .verdict { text-align: right; min-width: 64px; }
  .verdict strong { display: block; font-size: 18px; color: var(--dim); transition: color .25s; }
  .check.on .verdict strong { color: var(--yes); }
  .verdict small { color: var(--dim); font-variant-numeric: tabular-nums; }
  footer { margin-top: 16px; color: var(--dim); font-size: 13px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  footer label { cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
  footer input { display: none; }
</style>`,
);

document.body.innerHTML = `
  <div class="app">
    <header>
      <h1>d1-omni-600M <span>watches your camera</span></h1>
      <a href="./">← all demos</a>
    </header>
    <main>
      <div class="stage">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="frame" width="${FRAME}" height="${FRAME}" hidden></canvas>
        <div class="badge"><i></i>LIVE</div>
        <div class="stats"><span id="speed">–</span><span>on-device · WebGPU</span></div>
        <div class="loading" id="loading"><div><span id="loadText">Starting…</span><b><i id="loadBar"></i></b></div></div>
      </div>
      <div class="checks">
        ${CHECKS.map(
          (c) => `
          <div class="check" id="c-${c.key}">
            <div class="icon"><svg viewBox="0 0 24 24">${ICONS[c.icon]}</svg></div>
            <div><div class="label">${c.label}</div><div class="meter"><i></i></div></div>
            <div class="verdict"><strong>–</strong><small>–</small></div>
          </div>`,
        ).join('')}
      </div>
    </main>
    <footer>
      <span>Every frame runs through the SigLIP2 tower and the LFM2 trunk on your GPU. Nothing leaves the browser.</span>
      <label>No camera? Use a photo<input type="file" id="file" accept="image/*" /></label>
    </footer>
  </div>`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const frame = $<HTMLCanvasElement>('frame');
const ctx = frame.getContext('2d', { willReadFrequently: true })!;
let still: ImageBitmap | undefined;

function loading(text: string) {
  $('loadText').textContent = text;
  const pct = /(\d+)%/.exec(text)?.[1];
  if (pct) $('loadBar').style.setProperty('--p', `${pct}%`);
}

/** Center-crops a source to a square and draws it at FRAME px. */
function drawSquare(src: CanvasImageSource, w: number, h: number) {
  const side = Math.min(w, h);
  ctx.drawImage(src, (w - side) / 2, (h - side) / 2, side, side, 0, 0, FRAME, FRAME);
}

function show(key: string, p: number) {
  const el = $(`c-${key}`);
  el.classList.toggle('on', p >= 0.5);
  el.querySelector<HTMLElement>('.meter i')!.style.width = `${(p * 100).toFixed(1)}%`;
  el.querySelector('strong')!.textContent = p >= 0.5 ? 'Yes' : 'No';
  el.querySelector('small')!.textContent = `${(p * 100).toFixed(0)}%`;
}

async function main() {
  const decider = await loadDecider(loading);
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
  } catch {
    loading('No camera found. Pick a photo below.');
  }
  $<HTMLInputElement>('file').onchange = async (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    still = await createImageBitmap(f);
    video.hidden = true;
    frame.hidden = false;
  };
  let first = true;
  for (;;) {
    if (still) drawSquare(still, still.width, still.height);
    else if (video.videoWidth) drawSquare(video, video.videoWidth, video.videoHeight);
    else {
      await new Promise((r) => setTimeout(r, 200));
      continue;
    }
    const t = performance.now();
    const { answers } = await decider.systemOne('', QUESTIONS, [canvasRgb(ctx, FRAME, FRAME)]);
    for (const [key, a] of Object.entries(answers)) if (a.type === 'noul') show(key, a.noul);
    const ms = performance.now() - t;
    $('speed').textContent = `${ms.toFixed(0)} ms · ${(1000 / ms).toFixed(1)} fps`;
    if (first) {
      $('loading').remove();
      first = false;
    }
    if (still) await new Promise((r) => setTimeout(r, 1000));
  }
}

main().catch((err) => loading(`Error: ${err instanceof Error ? err.message : String(err)}`));
