/** d1-omni plays snake: every tick the board goes in as a text or an image
 *  state, and one choice question picks the move. */
import tgpu from 'typegpu';
import { fromSafetensors, initRunntime } from '../../src/core/index.ts';
import { createD1Decider, type D1Decider } from '../../src/zoo/d1/decider.ts';
import type { Question } from '../../src/zoo/d1/prompt.ts';
import { loadD1Tokenizer } from './tokenizer.ts';

const N = 10;
const CELL = 32;
type Dir = 'up' | 'down' | 'left' | 'right';
type P = { r: number; c: number };
const STEP: Record<Dir, P> = {
  up: { r: -1, c: 0 },
  down: { r: 1, c: 0 },
  left: { r: 0, c: -1 },
  right: { r: 0, c: 1 },
};
const DIRS = Object.keys(STEP) as Dir[];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
const statusEl = $('status');
const probsEl = $('probs');
const startBtn = $<HTMLButtonElement>('start');
const modeSel = $<HTMLSelectElement>('mode');
const safeBox = $<HTMLInputElement>('safe');

let snake: P[] = [];
let dir: Dir = 'right';
let food: P = { r: 0, c: 0 };
let score = 0;
let steps = 0;
let running = false;
let stopReason = '';
/** Board positions since the last food, to end a game that circles forever. */
let seen = new Set<string>();

function reset() {
  snake = [
    { r: 5, c: 4 },
    { r: 5, c: 3 },
    { r: 5, c: 2 },
  ];
  dir = 'right';
  score = 0;
  steps = 0;
  placeFood();
  seen = new Set();
}

function placeFood() {
  const free: P[] = [];
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++) if (!snake.some((s) => s.r === r && s.c === c)) free.push({ r, c });
  food = free[Math.floor(Math.random() * free.length)]!;
}

const next = (d: Dir): P => ({ r: snake[0]!.r + STEP[d].r, c: snake[0]!.c + STEP[d].c });
/** A move that ends the game: a wall, or the body (the tail cell moves away). */
function deadly(d: Dir): string | undefined {
  const p = next(d);
  if (p.r < 0 || p.r >= N || p.c < 0 || p.c >= N) return 'a wall';
  if (snake.slice(0, -1).some((s) => s.r === p.r && s.c === p.c)) return 'its own body';
  return undefined;
}

function draw() {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, N * CELL, N * CELL);
  ctx.fillStyle = '#e33';
  ctx.fillRect(food.c * CELL + 4, food.r * CELL + 4, CELL - 8, CELL - 8);
  snake.forEach((s, i) => {
    ctx.fillStyle = i === 0 ? '#ff0' : '#3c3';
    ctx.fillRect(s.c * CELL + 1, s.r * CELL + 1, CELL - 2, CELL - 2);
  });
}

function textState(): string {
  const rows = [];
  for (let r = 0; r < N; r++) {
    let line = '';
    for (let c = 0; c < N; c++) {
      const i = snake.findIndex((s) => s.r === r && s.c === c);
      line += i === 0 ? 'H' : i > 0 ? 'S' : food.r === r && food.c === c ? 'F' : '.';
    }
    rows.push(line);
  }
  const h = snake[0]!;
  const dangers = DIRS.map((d) => [d, deadly(d)] as const).filter(([, why]) => why);
  return [
    `Snake game on a ${N}x${N} grid. Row 0 is the top, column 0 is the left.`,
    `H is the snake's head, S its body, F the food, . empty.`,
    rows.join('\n'),
    `The head is at row ${h.r}, column ${h.c}, moving ${dir}. The food is at row ${food.r}, column ${food.c}.`,
    dangers.length
      ? dangers.map(([d, why]) => `Moving ${d} hits ${why} and loses.`).join(' ')
      : 'No move hits anything right now.',
  ].join('\n');
}

const IMAGE_STATE =
  'A snake game. The yellow square is the head of the snake, green squares are its body, the red square is the food. The snake loses if it leaves the board or runs into itself.';

/** Steps from p to the food, ignoring the body. */
const distance = (p: P) => Math.abs(p.r - food.r) + Math.abs(p.c - food.c);

/** Each option says where the move goes and what it does to the food
 *  distance, so the model compares consequences instead of reading geometry. */
function question(options: Dir[]): Question {
  const now = distance(snake[0]!);
  const describe = (d: Dir) => {
    const after = distance(next(d));
    const effect =
      next(d).r === food.r && next(d).c === food.c
        ? 'eats the food'
        : after < now
          ? `gets closer to the food (${after} steps away)`
          : `moves away from the food (${after} steps away)`;
    const danger = deadly(d);
    return `Move ${d}: ${danger ? `hits ${danger} and loses the game` : effect}`;
  };
  return {
    type: 'choice',
    instructions: 'Which move is best for the snake: eat the food soon and never lose the game?',
    criteria: Object.fromEntries(options.map((d) => [d, describe(d)])),
  };
}

function boardImage() {
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const rgb = new Uint8Array(canvas.width * canvas.height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i]!;
    rgb[j + 1] = data[i + 1]!;
    rgb[j + 2] = data[i + 2]!;
  }
  return { width: canvas.width, height: canvas.height, data: rgb };
}

async function tick(decider: D1Decider): Promise<boolean> {
  // The snake cannot reverse into its neck; the safety net also drops moves that lose at once.
  const reverse = DIRS.find((d) => STEP[d].r === -STEP[dir].r && STEP[d].c === -STEP[dir].c)!;
  let options = DIRS.filter((d) => d !== reverse);
  if (safeBox.checked) {
    const safe = options.filter((d) => !deadly(d));
    if (safe.length === 0) return false;
    options = safe;
  }
  const image = modeSel.value === 'image';
  const t = performance.now();
  const { answers } = await decider.systemOne(
    image ? IMAGE_STATE : textState(),
    { move: question(options) },
    image ? [boardImage()] : undefined,
  );
  const ms = performance.now() - t;
  const a = answers.move!;
  if (a.type !== 'choice') return false;
  probsEl.innerHTML = Object.entries(a.probabilities)
    .sort((x, y) => y[1] - x[1])
    .map(
      ([d, p]) =>
        `<div class="bar"><span>${d}</span><i style="width:${(p * 100).toFixed(0)}%"></i><b>${(p * 100).toFixed(1)}%</b></div>`,
    )
    .join('');
  statusEl.textContent = `score ${score} · step ${steps} · ${ms.toFixed(0)} ms/move · ${a.choice}`;
  dir = a.choice as Dir;
  if (deadly(dir)) return false;
  const head = next(dir);
  snake.unshift(head);
  if (head.r === food.r && head.c === food.c) {
    score++;
    if (snake.length === N * N) return false;
    placeFood();
    seen = new Set();
  } else {
    snake.pop();
  }
  steps++;
  draw();
  const key = snake.map((p) => `${p.r},${p.c}`).join(' ');
  if (seen.has(key)) {
    stopReason = 'stuck in a loop';
    return false;
  }
  seen.add(key);
  return true;
}

async function main() {
  canvas.width = canvas.height = N * CELL;
  reset();
  draw();
  statusEl.textContent = 'loading model…';
  initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
  const decider = await createD1Decider(
    await fromSafetensors('/model.safetensors'),
    await loadD1Tokenizer(),
    {
      onProgress: (_, done, total) =>
        (statusEl.textContent = `loading model ${Math.floor((100 * done) / total)}%`),
    },
  );
  statusEl.textContent = 'ready';
  startBtn.disabled = false;
  startBtn.onclick = async () => {
    if (running) {
      running = false;
      return;
    }
    running = true;
    startBtn.textContent = 'Stop';
    reset();
    draw();
    stopReason = '';
    while (running && (await tick(decider))) await new Promise((r) => setTimeout(r, 60));
    if (running)
      statusEl.textContent = `game over (${stopReason || 'crashed'}) · score ${score} in ${steps} moves`;
    running = false;
    startBtn.textContent = 'Play';
  };
}

main().catch((err) => {
  statusEl.textContent = `ERROR ${err instanceof Error ? err.message : String(err)}`;
  console.error(err);
});
