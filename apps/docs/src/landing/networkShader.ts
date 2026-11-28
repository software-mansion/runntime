/** The landing page hero: a convolutional network looking at an abstract image, drawn as
 *  a stack of feature maps in perspective and ray-traced in one fragment shader. */

import tgpu, { common, d, std, type TgpuRoot } from 'typegpu';
import { runCanvasLoop } from './canvasLoop';
import { hash21, valueNoise } from './noise';

/** World distance between two feature maps along the network axis. */
const LAYER_GAP = 1.5;
/** Half width and half height of each feature map, in world units, front to back:
 *  input image, edges, pooled, sparse, output vector. */
const HALF_SIZES = [
  [1, 1],
  [0.86, 0.86],
  [0.72, 0.72],
  [0.58, 0.58],
  [0.14, 0.62],
] as const;
/** Half size of the receptive-field window on maps 0 to 3, as a fraction of the map. */
const WINDOW_HALF = [0.22, 0.17, 0.15, 0.12] as const;
/** Faint filaments per gap that carry the background activity. */
const FILAMENTS = 9;
/** Nodes in the output vector, and the one that wins, counted from the top. */
const OUTPUT_NODES = 7;
const OUTPUT_WINNER = 2;
/** Seconds per forward pass: the wave crosses every gap, then rests. */
const PASS_SECONDS = 5.2;
/** Gray levels the ordered dither quantizes to, black included. */
const DITHER_LEVELS = 6;
/** Opacity of a map's panel over what is behind it. */
const MAP_OPACITY = 0.86;
/** Canvas pixels per CSS pixel. The dither cells show at 2 CSS px. */
const RENDER_SCALE = 0.5;
/** Focal length of the camera, in halves of the screen's short side. */
const FOCAL = 2.6;

const outputHalf = HALF_SIZES[4][1];
/** The winning output node's center, as a fraction of the output map's half height. */
const WINNER_Y =
  (-outputHalf + (OUTPUT_NODES - OUTPUT_WINNER - 0.5) * ((2 * outputHalf) / OUTPUT_NODES)) /
  outputHalf;

const Params = d.struct({
  resolution: d.vec2f,
  time: d.f32,
  /** The receptive field's center on the input image, each axis in [-1, 1]. */
  focus: d.vec2f,
  eye: d.vec3f,
  lookAt: d.vec3f,
  /** Moves the stack across the screen so the copy has room, in halves of the short side. */
  shift: d.vec2f,
});

const layout = tgpu.bindGroupLayout({ params: { uniform: Params } });

/** The input image: soft blobs filled with an interference pattern, a ring and
 *  a few wave lines on a dark ground. `drift` moves the shapes slowly. `q`
 *  spans [-1, 1], y up. */
const inputImage = tgpu.fn(
  [d.vec2f, d.vec2f],
  d.f32,
)((q, drift) => {
  'use gpu';
  // Three metaballs merge into one organic shape with a crisp edge.
  const c0 = d.vec2f(-0.35, 0.2) + drift * 0.18;
  const c1 = d.vec2f(0.3, -0.15) - drift * d.vec2f(0.14, 0.2);
  const c2 = d.vec2f(0.15, 0.45) + d.vec2f(drift.y, -drift.x) * 0.16;
  const field =
    0.13 / std.max(std.dot(q - c0, q - c0), 1e-3) +
    0.1 / std.max(std.dot(q - c1, q - c1), 1e-3) +
    0.05 / std.max(std.dot(q - c2, q - c2), 1e-3);
  const blob = std.smoothstep(0.95, 1.05, field);
  const rim = std.exp(-std.pow((field - 1) * 9, 2));

  // Two ring sources beat against each other inside the blob.
  const waveA = std.sin(std.length(q - d.vec2f(-0.8, -0.6)) * 34);
  const waveB = std.sin(std.length(q - d.vec2f(0.9, 0.7)) * 31);
  const moire = 0.5 + 0.5 * waveA * waveB;
  const fill = 0.38 + 0.6 * moire * std.smoothstep(1, 2.4, field);

  const ringCenter = d.vec2f(0.5, 0.48) - drift * 0.1;
  const ring = 1 - std.smoothstep(0.012, 0.026, std.abs(std.length(q - ringCenter) - 0.22));
  const lineY = q.y + 0.55 + 0.12 * std.sin(q.x * 4.2 + drift.x * 2);
  const lines = std.exp(-std.pow(lineY * 38, 2)) + std.exp(-std.pow((lineY - 0.12) * 38, 2)) * 0.6;
  const ground = 0.06 + 0.05 * valueNoise(q * 5 + drift);

  let value = std.mix(ground + lines * 0.55, fill, blob) + rim * 0.35;
  value = std.max(value, ring * 0.9);
  return std.clamp(value, 0, 1);
});

/** The gradient of the input image, by central differences of width `e`. */
const imageGradient = tgpu.fn(
  [d.vec2f, d.vec2f, d.f32],
  d.vec2f,
)((q, drift, e) => {
  'use gpu';
  const dx = inputImage(q + d.vec2f(e, 0), drift) - inputImage(q - d.vec2f(e, 0), drift);
  const dy = inputImage(q + d.vec2f(0, e), drift) - inputImage(q - d.vec2f(0, e), drift);
  return d.vec2f(dx, dy) / (2 * e);
});

/** Splits a map into `tiles`×`tiles` channels. Returns the position inside the
 *  channel in [-1, 1] (xy), the channel index (z) and whether it is inside (w). */
const channelTile = tgpu.fn(
  [d.vec2f, d.f32],
  d.vec4f,
)((q, tiles) => {
  'use gpu';
  const u = (q * 0.5 + 0.5) * tiles;
  const cell = std.floor(u);
  const inner = (std.fract(u) * 2 - 1) * 1.1;
  const inside = std.step(std.max(std.abs(inner.x), std.abs(inner.y)), 1);
  return d.vec4f(inner, cell.x + cell.y * tiles, inside);
});

/** First conv: four oriented edge filters, one per channel. */
const edgeMap = tgpu.fn(
  [d.vec2f, d.vec2f],
  d.f32,
)((q, drift) => {
  'use gpu';
  const tile = channelTile(q, 2);
  const grid = (std.floor((tile.xy * 0.5 + 0.5) * 56) + 0.5) / 56;
  const g = imageGradient(grid * 2 - 1, drift, 0.025);
  const angle = tile.z * (Math.PI / 4);
  const response = std.abs(std.dot(g, d.vec2f(std.cos(angle), std.sin(angle))));
  return std.min(response * 0.16, 1) * tile.w;
});

/** After pooling: nine rectified orientations on a coarse grid of blocks. */
const pooledMap = tgpu.fn(
  [d.vec2f, d.vec2f],
  d.f32,
)((q, drift) => {
  'use gpu';
  const tile = channelTile(q, 3);
  const cells = d.f32(13);
  const u = (tile.xy * 0.5 + 0.5) * cells;
  const center = ((std.floor(u) + 0.5) / cells) * 2 - 1;
  const g = imageGradient(center, drift, 0.07);
  const angle = tile.z * ((2 * Math.PI) / 9);
  const response = std.max(std.dot(g, d.vec2f(std.cos(angle), std.sin(angle))), 0);
  const f = std.abs(std.fract(u) - 0.5);
  const block = std.step(std.max(f.x, f.y), 0.38);
  return std.min(response * 0.7, 1) * block * tile.w;
});

/** Deep features: a sparse grid where few units respond, most near the focus. */
const sparseMap = tgpu.fn(
  [d.vec2f, d.vec2f, d.vec2f],
  d.f32,
)((q, drift, focus) => {
  'use gpu';
  const cells = d.f32(9);
  const u = (q * 0.5 + 0.5) * cells;
  const cell = std.floor(u);
  const center = ((cell + 0.5) / cells) * 2 - 1;
  const g = imageGradient(center, drift, 0.12);
  const near = 1 - std.smoothstep(0.1, 0.45, std.length(center - focus));
  const unit = hash21(cell + 7.1);
  const act = std.clamp(std.smoothstep(1.2, 3.2, std.length(g)) * unit * 1.4 + near * 0.7, 0, 1);
  const radius = 0.12 + 0.26 * act;
  const disc = 1 - std.smoothstep(radius - 0.05, radius, std.length(std.fract(u) - 0.5));
  return (0.12 + 0.88 * act) * disc;
});

/** The output vector, at `lp` in world units around the map center. `pulse`
 *  is the arrival of the forward pass, 0 to 1. */
const outputMap = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((lp, pulse) => {
  'use gpu';
  const pitch = (2 * outputHalf) / OUTPUT_NODES;
  const slot = std.clamp(std.floor((lp.y + outputHalf) / pitch), 0, OUTPUT_NODES - 1);
  const center = d.vec2f(0, -outputHalf + (slot + 0.5) * pitch);
  const winner = 1 - std.step(0.5, std.abs(slot - (OUTPUT_NODES - 1 - OUTPUT_WINNER)));
  const act = std.mix(0.12 + 0.3 * hash21(d.vec2f(slot, 3.3)), 1, winner * (0.3 + 0.7 * pulse));
  const r = std.length(lp - center);
  const node = 1 - std.smoothstep(0.045, 0.055, r);
  return act * node + std.exp(-r * 22) * winner * pulse * 0.7;
});

/** Where the ray meets the map at depth `z`: local position (xy), inside (z), ray distance (w). */
const hitMap = tgpu.fn(
  [d.vec3f, d.vec3f, d.f32, d.vec2f],
  d.vec4f,
)((ro, rd, z, half) => {
  'use gpu';
  const t = (z - ro.z) / rd.z;
  const p = ro + rd * t;
  const inside = std.step(std.abs(p.x), half.x) * std.step(std.abs(p.y), half.y) * std.step(0, t);
  return d.vec4f(p.xy, inside, t);
});

/** The panel a map is printed on: dark fill, hairline border, the focus
 *  window, the marker where the window below converges, and the content,
 *  lit up inside the window and when the pass reaches the map. */
const shadeMap = tgpu.fn(
  [d.vec2f, d.vec2f, d.f32, d.vec2f, d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((lp, half, content, focus, windowHalf, markerStrength, pixel, flash) => {
  'use gpu';
  const edge = std.min(half.x - std.abs(lp.x), half.y - std.abs(lp.y));
  const border = 1 - std.smoothstep(pixel * 0.5, pixel * 1.5, edge);
  const q = lp / half;
  const w = std.abs(q - focus);
  const box = std.max(w.x, w.y);
  const inWindow =
    (1 - std.smoothstep(windowHalf - 0.01, windowHalf, box)) * std.step(0.001, windowHalf);
  const outline =
    (1 - std.smoothstep(0, (pixel * 1.4) / half.x, std.abs(box - windowHalf))) *
    std.step(0.001, windowHalf);
  const marker = 1 - std.smoothstep(0.025, 0.04, std.length(q - focus));
  const lit = content * (0.6 + 0.6 * inWindow + 0.45 * flash);
  return 0.03 + lit + border * 0.3 + outline * 0.7 + marker * markerStrength;
});

/** Glow of a 3D segment seen along a ray (`base` sets its brightness), with a
 *  pulse of light at `pulseAt` along it, 0 to 1. */
const segmentGlow = tgpu.fn(
  [d.vec3f, d.vec3f, d.vec3f, d.vec3f, d.f32, d.f32, d.f32],
  d.f32,
)((ro, rd, a, b, pixelAngle, base, pulseAt) => {
  'use gpu';
  const ab = b - a;
  const w0 = ro - a;
  const bd = std.dot(rd, ab);
  const aa = std.dot(ab, ab);
  const u = std.clamp((std.dot(ab, w0) - bd * std.dot(rd, w0)) / std.max(aa - bd * bd, 1e-5), 0, 1);
  const onSegment = a + ab * u;
  const s = std.max(std.dot(onSegment - ro, rd), 0);
  const dist = std.length(ro + rd * s - onSegment);
  const width = s * pixelAngle;
  const line = std.exp(-std.pow(dist / width, 2));
  const pulse =
    std.exp(-std.pow((u - pulseAt) * 14, 2)) * std.exp(-std.pow(dist / (width * 1.6), 2));
  return line * base + pulse * 0.8;
});

/** The point at map-relative `q` on the map `layer`, in world space. */
const mapPoint = tgpu.fn(
  [d.vec2f, d.vec2f, d.f32],
  d.vec3f,
)((q, half, layer) => {
  'use gpu';
  return d.vec3f(q * half, layer * LAYER_GAP);
});

/** Light in the gap after the map `layer`: the receptive-field cone from the
 *  window on this map to one unit of the next, and the background filaments.
 *  `progress` is where the forward pass is along this gap, 0 to 1. */
const gapGlow = tgpu.fn(
  [d.vec3f, d.vec3f, d.f32, d.vec2f, d.vec2f, d.vec2f, d.vec2f, d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((ro, rd, layer, half, nextHalf, focus, nextFocus, windowHalf, progress, pixelAngle, time) => {
  'use gpu';
  const tip = mapPoint(nextFocus, nextHalf, layer + 1);
  let glow = segmentGlow(
    ro,
    rd,
    mapPoint(focus + d.vec2f(-windowHalf, -windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    ro,
    rd,
    mapPoint(focus + d.vec2f(windowHalf, -windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    ro,
    rd,
    mapPoint(focus + d.vec2f(windowHalf, windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    ro,
    rd,
    mapPoint(focus + d.vec2f(-windowHalf, windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  for (const i of std.range(FILAMENTS)) {
    const seed = d.vec2f(d.f32(i), layer);
    const from = d.vec2f(hash21(seed), hash21(seed + 1.7)) * 1.8 - 0.9;
    const to = d.vec2f(hash21(seed + 3.1), hash21(seed + 4.9)) * 1.8 - 0.9;
    const travel = std.fract(time * 0.21 + hash21(seed + 8.3));
    const a = mapPoint(from, half, layer);
    const b = mapPoint(to, nextHalf, layer + 1);
    glow += segmentGlow(ro, rd, a, b, pixelAngle, 0.14, travel) * 0.22;
    glow += segmentGlow(ro, rd, a, b, pixelAngle, 0, progress) * 0.4;
  }
  return glow;
});

/** 4×4 Bayer threshold in [0, 1), from the pixel coordinate. */
const bayer4 = tgpu.fn(
  [d.vec2f],
  d.f32,
)((p) => {
  'use gpu';
  const a = std.floor(std.fract(p / 4) * 4);
  const b = std.floor(a * 0.5);
  const fine = std.fract(a.x * 0.5 + a.y * a.y * 0.75);
  const coarse = std.fract(b.x * 0.5 + b.y * b.y * 0.75);
  return coarse * 0.25 + fine;
});

/** How bright a map flashes when the pass reaches it. `wave` counts gaps crossed. */
const flashAt = tgpu.fn(
  [d.f32, d.f32],
  d.f32,
)((wave, layer) => {
  'use gpu';
  return std.exp(-std.pow((wave - layer) * 2.2, 2));
});

const heroFragment = tgpu.fragmentFn({
  in: { pos: d.builtin.position },
  out: d.vec4f,
})(({ pos }) => {
  'use gpu';
  const P = layout.$.params;
  const res = P.resolution;
  // Units of half the short side, so a portrait screen still fits the stack.
  const shortSide = std.min(res.x, res.y);
  const screen = d.vec2f(pos.x * 2 - res.x, res.y - pos.y * 2) / shortSide - P.shift;

  const forward = std.normalize(P.lookAt - P.eye);
  const right = std.normalize(std.cross(d.vec3f(0, 1, 0), forward));
  const up = std.cross(forward, right);
  const ro = P.eye;
  const rd = std.normalize(forward * FOCAL + right * screen.x + up * screen.y);
  const pixelAngle = 2 / (shortSide * FOCAL);

  const wave = std.fract(P.time / PASS_SECONDS) * 5.4 - 0.2;
  const focus = P.focus;
  // The input image drifts slowly and leans toward the focus.
  const drift = d.vec2f(std.sin(P.time * 0.13), std.cos(P.time * 0.11)) + focus * 0.3;

  const half0 = d.vec2f(HALF_SIZES[0][0], HALF_SIZES[0][1]);
  const half1 = d.vec2f(HALF_SIZES[1][0], HALF_SIZES[1][1]);
  const half2 = d.vec2f(HALF_SIZES[2][0], HALF_SIZES[2][1]);
  const half3 = d.vec2f(HALF_SIZES[3][0], HALF_SIZES[3][1]);
  const half4 = d.vec2f(HALF_SIZES[4][0], HALF_SIZES[4][1]);
  const winner = d.vec2f(0, WINNER_Y);

  // Back to front: each map covers the light behind it.
  let color = d.f32(0);
  const h4 = hitMap(ro, rd, 4 * LAYER_GAP, half4);
  if (h4.z > 0) {
    const pulse = std.smoothstep(3.7, 4.1, wave);
    const shade = shadeMap(
      h4.xy,
      half4,
      outputMap(h4.xy, pulse),
      winner,
      0,
      0,
      h4.w * pixelAngle,
      0,
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    ro,
    rd,
    3,
    half3,
    half4,
    focus,
    winner,
    WINDOW_HALF[3],
    wave - 3,
    pixelAngle,
    P.time,
  );
  const h3 = hitMap(ro, rd, 3 * LAYER_GAP, half3);
  if (h3.z > 0) {
    const content = sparseMap(h3.xy / half3, drift, focus);
    const shade = shadeMap(
      h3.xy,
      half3,
      content,
      focus,
      WINDOW_HALF[3],
      0.5,
      h3.w * pixelAngle,
      flashAt(wave, 3),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    ro,
    rd,
    2,
    half2,
    half3,
    focus,
    focus,
    WINDOW_HALF[2],
    wave - 2,
    pixelAngle,
    P.time,
  );
  const h2 = hitMap(ro, rd, 2 * LAYER_GAP, half2);
  if (h2.z > 0) {
    const content = pooledMap(h2.xy / half2, drift);
    const shade = shadeMap(
      h2.xy,
      half2,
      content,
      focus,
      WINDOW_HALF[2],
      0.5,
      h2.w * pixelAngle,
      flashAt(wave, 2),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    ro,
    rd,
    1,
    half1,
    half2,
    focus,
    focus,
    WINDOW_HALF[1],
    wave - 1,
    pixelAngle,
    P.time,
  );
  const h1 = hitMap(ro, rd, LAYER_GAP, half1);
  if (h1.z > 0) {
    const content = edgeMap(h1.xy / half1, drift);
    const shade = shadeMap(
      h1.xy,
      half1,
      content,
      focus,
      WINDOW_HALF[1],
      0.5,
      h1.w * pixelAngle,
      flashAt(wave, 1),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(ro, rd, 0, half0, half1, focus, focus, WINDOW_HALF[0], wave, pixelAngle, P.time);
  const h0 = hitMap(ro, rd, 0, half0);
  if (h0.z > 0) {
    const content = inputImage(h0.xy / half0, drift);
    const shade = shadeMap(
      h0.xy,
      half0,
      content,
      focus,
      WINDOW_HALF[0],
      0,
      h0.w * pixelAngle,
      flashAt(wave, 0),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }

  const uv = pos.xy / res;
  const vignette =
    1 - 0.55 * std.smoothstep(0.35, 0.95, std.length((uv - 0.5) * d.vec2f(1.1, 1.4)));
  color = std.clamp(color * vignette, 0, 1);

  // Ordered dither with a little moving grain, so it reads as print, not glow.
  const grain = (hash21(pos.xy + std.fract(P.time * 7.3) * 91) - 0.5) * 0.35;
  const level = std.floor(
    color * (DITHER_LEVELS - 1) + bayer4(pos.xy) + grain * std.step(0.01, color),
  );
  const v = std.clamp(level / (DITHER_LEVELS - 1), 0, 1);
  return d.vec4f(v, v, v, 1);
});

/** Where the camera sits and looks, in world units. From the right of the
 *  stack, so the input reads first and the output recedes. */
const CAMERA_EYE = [4.4, 1.7, -2.4] as const;
const CAMERA_LOOK_AT = [0, -0.1, 2.9] as const;
/** Camera drift, in seconds of period per axis. */
const DRIFT_PERIODS = [41, 29, 53] as const;
/** How fast the focus follows the pointer, per second. */
const FOCUS_EASE = 3;

/** A frozen moment for reduced motion: the pass mid-way through the network. */
const STILL_TIME = 2.1;

/** Starts the hero on `canvas` and returns a function that stops it. Throws
 *  when the pipeline does not compile. */
export function startHero(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  { still }: { still: boolean },
): () => void {
  const context = root.configureContext({ canvas, alphaMode: 'opaque' });
  const params = root.createBuffer(Params).$usage('uniform');
  const bindGroup = root.createBindGroup(layout, { params });
  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: heroFragment,
      targets: { format: navigator.gpu.getPreferredCanvasFormat() },
    })
    .with(bindGroup);

  const focus = d.vec2f();
  let pointer: [number, number] | undefined;

  const draw = (time: number, dt: number) => {
    // Without a pointer the focus wanders over the image on its own.
    const target = pointer ?? [0.55 * Math.sin(time * 0.37), 0.35 * Math.sin(time * 0.53 + 1)];
    const ease = 1 - Math.exp(-FOCUS_EASE * dt);
    focus.x += (target[0] - focus.x) * ease;
    focus.y += (target[1] - focus.y) * ease;

    const width = canvas.width;
    const height = canvas.height;
    const aspect = width / height;
    // Wide screens put the stack right of the copy, narrow ones above it.
    const shift = aspect > 1.1 ? d.vec2f(Math.min(aspect * 0.55, 1.2), 0.2) : d.vec2f(0.4, 1.1);
    const drift = DRIFT_PERIODS.map((period) => Math.sin((2 * Math.PI * time) / period));
    params.write({
      resolution: d.vec2f(width, height),
      time,
      focus,
      eye: d.vec3f(
        CAMERA_EYE[0] + 0.45 * drift[0]! + focus.x * 0.25,
        CAMERA_EYE[1] + 0.3 * drift[1]! + focus.y * 0.2,
        CAMERA_EYE[2] + 0.3 * drift[2]!,
      ),
      lookAt: d.vec3f(...CAMERA_LOOK_AT),
      shift,
    });
    pipeline.withColorAttachment({ view: context.getCurrentTexture().createView() }).draw(3);
  };

  const onPointer = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = 1 - ((event.clientY - rect.top) / rect.height) * 2;
    pointer = [Math.max(-0.75, Math.min(0.75, x)), Math.max(-0.75, Math.min(0.75, y))];
  };
  const onLeave = () => {
    pointer = undefined;
  };

  const stopLoop = runCanvasLoop({
    canvas,
    scale: RENDER_SCALE,
    still,
    startTime: still ? STILL_TIME : 0,
    draw,
  });
  if (!still) {
    window.addEventListener('pointermove', onPointer, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);
  }

  return () => {
    stopLoop();
    window.removeEventListener('pointermove', onPointer);
    document.documentElement.removeEventListener('pointerleave', onLeave);
  };
}
