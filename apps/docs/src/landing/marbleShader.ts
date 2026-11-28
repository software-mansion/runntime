/** Slow liquid marble in black and silver, from domain-warped noise, with thin
 *  dark veins in the light bands and film grain. It fills the navbar link pill
 *  and the NN letters of the nav mark and the hero wordmark, all from one
 *  pipeline and one frame loop. */

import tgpu, { common, d, std, type TgpuRoot } from 'typegpu';
import { runCanvasLoop } from './canvasLoop';
import { hash21, valueNoise } from './noise';

/** Most canvas pixels per CSS pixel. The targets are small, so they render at
 *  the device's pixel ratio, up to this, and the grain stays one pixel fine. */
const MAX_PIXEL_RATIO = 2;
/** How far each warp layer pushes the next one, in noise cells. */
const WARP = 1.6;
/** Seconds into the flow the first frame shows. At 0 the bands start in a
 *  plain arrangement; later the warp has folded them into marble. */
const START_TIME = 40;

const Params = d.struct({
  resolution: d.vec2f,
  time: d.f32,
  /** Noise cells across the canvas width (x), and down the same distance (y). */
  cells: d.vec2f,
  /** Darkest and lightest gray of this target's marble. */
  dark: d.f32,
  light: d.f32,
  /** How fast the marble flows, in noise cells per second. */
  flowSpeed: d.f32,
});
const layout = tgpu.bindGroupLayout({ params: { uniform: Params } });

const fbm = tgpu.fn(
  [d.vec2f],
  d.f32,
)((p) => {
  'use gpu';
  let sum = d.f32(0);
  let q = d.vec2f(p);
  for (const octave of tgpu.unroll(std.range(3))) {
    sum += valueNoise(q) * 0.5 ** (octave + 1);
    q = q * 2.03 + d.vec2f(1.7, 9.2);
  }
  return sum;
});

const marbleFragment = tgpu.fragmentFn({
  in: { pos: d.builtin.position },
  out: d.vec4f,
})(({ pos }) => {
  'use gpu';
  const params = layout.$.params;
  const flow = params.time * params.flowSpeed;
  // Stretched across, so the bands run mostly up and down like poured paint.
  const point = (pos.xy / params.resolution.x) * params.cells;
  const warp1 = d.vec2f(fbm(point + d.vec2f(flow, 0)), fbm(point + d.vec2f(5.2, 1.3 - flow)));
  const warp2 = d.vec2f(
    fbm(point + warp1 * WARP + d.vec2f(1.7, 9.2)),
    fbm(point + warp1 * WARP + d.vec2f(8.3, 2.8 + flow)),
  );
  const marble = fbm(point + warp2 * WARP);

  const light = std.smoothstep(0.27, 0.45, marble);
  const veins =
    std.exp(-std.pow((marble - 0.52) * 70, 2)) + std.exp(-std.pow((marble - 0.6) * 90, 2)) * 0.7;
  let value = std.mix(params.dark, params.light, light) - veins * light * 0.5;
  value += (hash21(pos.xy + std.fract(params.time) * 57) - 0.5) * 0.05;
  return d.vec4f(d.vec3f(std.clamp(value, 0, 1)), 1);
});

/** One canvas the marble fills, and how its marble looks. */
export interface MarbleTarget {
  canvas: HTMLCanvasElement;
  /** Noise cells across the canvas width, and down the same distance. */
  cells: readonly [number, number];
  /** Darkest and lightest gray, 0 to 1. Letters take a lighter range than
   *  the pill, so no part of a glyph sinks into the black page. */
  dark: number;
  light: number;
  /** Noise cells per second. A target that shows few cells needs more, or
   *  its flow moves under a pixel per second and reads as still. */
  flowSpeed: number;
}

/** Starts the marble on every target and returns a function that stops it.
 *  The first target drives the frame loop and must stay on screen (the fixed
 *  navbar pill); the others are drawn in the same frame while they are
 *  visible. Throws when the shader does not resolve; WGSL validation errors
 *  arrive later, on the device's error scope. */
export function startMarble(
  root: TgpuRoot,
  targets: readonly MarbleTarget[],
  { still }: { still: boolean },
): () => void {
  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: marbleFragment,
    targets: { format: navigator.gpu.getPreferredCanvasFormat() },
  });

  const pixelRatio = Math.min(devicePixelRatio, MAX_PIXEL_RATIO);
  const drawers = targets.map(({ canvas, cells, dark, light, flowSpeed }) => {
    const context = root.configureContext({ canvas, alphaMode: 'opaque' });
    const paramsBuffer = root.createBuffer(Params).$usage('uniform');
    const bound = pipeline.with(root.createBindGroup(layout, { params: paramsBuffer }));
    // Reused every frame, so drawing allocates nothing.
    const values = {
      resolution: d.vec2f(),
      time: 0,
      cells: d.vec2f(...cells),
      dark,
      light,
      flowSpeed,
    };
    let onScreen = true;
    return {
      canvas,
      setOnScreen: (visible: boolean) => {
        onScreen = visible;
      },
      draw: (time: number) => {
        if (!onScreen || canvas.width === 0) return;
        values.resolution.x = canvas.width;
        values.resolution.y = canvas.height;
        values.time = time;
        paramsBuffer.write(values);
        bound.withColorAttachment({ view: context }).draw(3);
      },
    };
  });
  const [primary, ...others] = drawers;
  if (!primary) return () => {};

  let lastTime = START_TIME;
  const resizeOther = (canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * pixelRatio));
    canvas.height = Math.max(1, Math.round(rect.height * pixelRatio));
  };
  for (const other of others) resizeOther(other.canvas);

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const other = others.find((o) => o.canvas === entry.target);
      if (!other) continue;
      resizeOther(other.canvas);
      other.draw(lastTime);
    }
  });
  const screenObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const other = others.find((o) => o.canvas === entry.target);
      other?.setOnScreen(entry.isIntersecting);
      if (entry.isIntersecting) other?.draw(lastTime);
    }
  });
  for (const other of others) {
    resizeObserver.observe(other.canvas);
    screenObserver.observe(other.canvas);
  }

  const stopLoop = runCanvasLoop({
    canvas: primary.canvas,
    scale: pixelRatio,
    still,
    startTime: START_TIME,
    draw: (time) => {
      lastTime = time;
      for (const drawer of drawers) drawer.draw(time);
    },
  });

  return () => {
    stopLoop();
    resizeObserver.disconnect();
    screenObserver.disconnect();
  };
}
