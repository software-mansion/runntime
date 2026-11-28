/** The hero button's hover: its fill dissolves into a grid of square pixels,
 *  a few of which flash light and fade, each on its own clock. Drawn on a 2D
 *  canvas inside the button, only while it is hovered or keyboard-focused. */

/** Side of one pixel cell and the gap between cells, in CSS pixels. */
const CELL = 3;
const GAP = 1;
/** Share of cells that twinkle; the rest hold a dim dither pattern. */
const TWINKLE_SHARE = 0.2;
/** Twinkle rates, in radians per second; each cell picks one in this range. */
const RATE_MIN = 1.2;
const RATE_MAX = 3.4;
/** Exponent on the twinkle sine: higher means shorter flashes. */
const FLASH_SHARPNESS = 14;
/** Gray levels, 0 to 255: the ground, the lighter dither cells, and the flash peak. */
const GROUND = 34;
const DITHER = 64;
const PEAK = 236;
/** Share of the dim cells that take the lighter dither level, by a 4×4 Bayer threshold. */
const DITHER_SHARE = 0.25;

const BAYER_4 = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

interface Cell {
  x: number;
  y: number;
  /** Radians per second, or 0 for a cell that does not twinkle. */
  rate: number;
  phase: number;
  base: number;
}

/** Lays out the cells for a button of `width`×`height` CSS pixels. */
function layoutCells(width: number, height: number): Cell[] {
  const pitch = CELL + GAP;
  const cells: Cell[] = [];
  for (let row = 0; row * pitch < height; row++) {
    for (let col = 0; col * pitch < width; col++) {
      const threshold = BAYER_4[(row % 4) * 4 + (col % 4)]! / 16;
      const twinkles = Math.random() < TWINKLE_SHARE;
      cells.push({
        x: col * pitch,
        y: row * pitch,
        rate: twinkles ? RATE_MIN + Math.random() * (RATE_MAX - RATE_MIN) : 0,
        phase: Math.random() * Math.PI * 2,
        base: threshold < DITHER_SHARE ? DITHER : GROUND,
      });
    }
  }
  return cells;
}

/** Gray level of a cell at `time` seconds. */
function cellLevel(cell: Cell, time: number): number {
  if (cell.rate === 0) return cell.base;
  const flash = Math.max(0, Math.sin(time * cell.rate + cell.phase)) ** FLASH_SHARPNESS;
  return Math.round(cell.base + (PEAK - cell.base) * flash);
}

/** Wires the effect to `button`, drawing on `canvas`. `still` draws the pattern
 *  once without twinkling. Returns a function that removes it. */
export function attachPixelGlitter(
  button: HTMLElement,
  canvas: HTMLCanvasElement,
  { still }: { still: boolean },
): () => void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};
  let cells: Cell[] = [];
  let frame = 0;
  let start = 0;

  const resize = () => {
    const { width, height } = button.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cells = layoutCells(width, height);
  };

  const draw = (time: number) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const cell of cells) {
      const level = cellLevel(cell, time);
      ctx.fillStyle = `rgb(${level} ${level} ${level})`;
      ctx.fillRect(cell.x, cell.y, CELL, CELL);
    }
  };

  const tick = (now: number) => {
    draw((now - start) / 1000);
    frame = requestAnimationFrame(tick);
  };

  const active = () => button.matches(':hover') || button.matches(':focus-visible');

  const begin = () => {
    if (frame || !active()) return;
    resize();
    if (still) {
      draw(0);
      return;
    }
    start = performance.now();
    frame = requestAnimationFrame(tick);
  };

  const end = () => {
    if (active()) return;
    cancelAnimationFrame(frame);
    frame = 0;
  };

  button.addEventListener('pointerenter', begin);
  button.addEventListener('focus', begin);
  button.addEventListener('pointerleave', end);
  button.addEventListener('blur', end);

  return () => {
    cancelAnimationFrame(frame);
    button.removeEventListener('pointerenter', begin);
    button.removeEventListener('focus', begin);
    button.removeEventListener('pointerleave', end);
    button.removeEventListener('blur', end);
  };
}
