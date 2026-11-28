/** The navbar fill: slow liquid marble in black and silver, from domain-warped
 *  noise, with thin dark veins in the light bands and film grain. */

import tgpu, { common, d, std, type TgpuRoot } from 'typegpu';
import { runCanvasLoop } from './canvasLoop';
import { hash21, valueNoise } from './noise';

/** Canvas pixels per CSS pixel. The marble is soft, so a low resolution is enough. */
const RENDER_SCALE = 0.5;
/** Darkest and lightest gray of the marble, kept low so the nav text stays legible. */
const DARK = 0.04;
const LIGHT = 0.62;
/** Noise cells across the width of the bar, and down the same distance. The
 *  bar shows a thin slice of a large, slow pattern. */
const CELLS_ACROSS = [3.4, 1.2] as const;
/** How far each warp layer pushes the next one, in noise cells. */
const WARP = 2.2;

const Params = d.struct({ resolution: d.vec2f, time: d.f32 });
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
  const P = layout.$.params;
  const t = P.time * 0.035;
  // Stretched across, so the bands run mostly up and down like poured paint.
  const p = (pos.xy / P.resolution.x) * d.vec2f(CELLS_ACROSS[0], CELLS_ACROSS[1]);
  const w1 = d.vec2f(fbm(p + d.vec2f(t, 0)), fbm(p + d.vec2f(5.2, 1.3 - t)));
  const w2 = d.vec2f(
    fbm(p + w1 * WARP + d.vec2f(1.7, 9.2)),
    fbm(p + w1 * WARP + d.vec2f(8.3, 2.8 + t)),
  );
  const v = fbm(p + w2 * WARP);

  const light = std.smoothstep(0.34, 0.52, v);
  const veins = std.exp(-std.pow((v - 0.58) * 70, 2)) + std.exp(-std.pow((v - 0.64) * 90, 2)) * 0.7;
  let value = std.mix(DARK, LIGHT, light) - veins * light * 0.5;
  value += (hash21(pos.xy + std.fract(P.time) * 57) - 0.5) * 0.05;
  return d.vec4f(d.vec3f(std.clamp(value, 0, 1)), 1);
});

/** Starts the marble on `canvas` and returns a function that stops it. Throws
 *  when the pipeline does not compile. */
export function startMarble(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  { still }: { still: boolean },
): () => void {
  const context = root.configureContext({ canvas, alphaMode: 'opaque' });
  const params = root.createBuffer(Params).$usage('uniform');
  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: marbleFragment,
      targets: { format: navigator.gpu.getPreferredCanvasFormat() },
    })
    .with(root.createBindGroup(layout, { params }));

  return runCanvasLoop({
    canvas,
    scale: RENDER_SCALE,
    still,
    startTime: 40,
    draw: (time) => {
      params.write({ resolution: [canvas.width, canvas.height], time });
      pipeline.withColorAttachment({ view: context.getCurrentTexture().createView() }).draw(3);
    },
  });
}
