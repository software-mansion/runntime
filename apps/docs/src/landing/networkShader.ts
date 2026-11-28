/** The landing page hero: a convolutional network looking at an abstract image, drawn as
 *  a stack of feature maps in perspective and ray-traced in one fragment shader.
 *  The canvas is fixed behind the whole page; scrolling moves the camera deeper. */

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

/** The gradient of the input image, by central differences `spacing` apart. */
const imageGradient = tgpu.fn(
  [d.vec2f, d.vec2f, d.f32],
  d.vec2f,
)((q, drift, spacing) => {
  'use gpu';
  const right = d.vec2f(spacing, 0);
  const up = d.vec2f(0, spacing);
  const dx = inputImage(q + right, drift) - inputImage(q - right, drift);
  const dy = inputImage(q + up, drift) - inputImage(q - up, drift);
  return d.vec2f(dx, dy) / (2 * spacing);
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
  const gradient = imageGradient(grid * 2 - 1, drift, 0.025);
  const angle = tile.z * (Math.PI / 4);
  const response = std.abs(std.dot(gradient, d.vec2f(std.cos(angle), std.sin(angle))));
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
  const gradient = imageGradient(center, drift, 0.07);
  const angle = tile.z * ((2 * Math.PI) / 9);
  const response = std.max(std.dot(gradient, d.vec2f(std.cos(angle), std.sin(angle))), 0);
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
  const gradient = imageGradient(center, drift, 0.12);
  const near = 1 - std.smoothstep(0.1, 0.45, std.length(center - focus));
  const unit = hash21(cell + 7.1);
  const activation = std.clamp(
    std.smoothstep(1.2, 3.2, std.length(gradient)) * unit * 1.4 + near * 0.7,
    0,
    1,
  );
  const radius = 0.12 + 0.26 * activation;
  const disc = 1 - std.smoothstep(radius - 0.05, radius, std.length(std.fract(u) - 0.5));
  return (0.12 + 0.88 * activation) * disc;
});

/** The output vector, at `localPosition` in world units around the map center. `pulse`
 *  is the arrival of the forward pass, 0 to 1. */
const outputMap = tgpu.fn(
  [d.vec2f, d.f32],
  d.f32,
)((localPosition, pulse) => {
  'use gpu';
  const pitch = (2 * outputHalf) / OUTPUT_NODES;
  const slot = std.clamp(std.floor((localPosition.y + outputHalf) / pitch), 0, OUTPUT_NODES - 1);
  const center = d.vec2f(0, -outputHalf + (slot + 0.5) * pitch);
  const winner = 1 - std.step(0.5, std.abs(slot - (OUTPUT_NODES - 1 - OUTPUT_WINNER)));
  const activation = std.mix(
    0.12 + 0.3 * hash21(d.vec2f(slot, 3.3)),
    1,
    winner * (0.3 + 0.7 * pulse),
  );
  const r = std.length(localPosition - center);
  const node = 1 - std.smoothstep(0.045, 0.055, r);
  return activation * node + std.exp(-r * 22) * winner * pulse * 0.7;
});

/** Where the ray meets the map at depth `z`: local position (xy), inside (z), ray distance (w). */
const hitMap = tgpu.fn(
  [d.vec3f, d.vec3f, d.f32, d.vec2f],
  d.vec4f,
)((rayOrigin, rayDirection, z, half) => {
  'use gpu';
  const t = (z - rayOrigin.z) / rayDirection.z;
  const p = rayOrigin + rayDirection * t;
  const inside = std.step(std.abs(p.x), half.x) * std.step(std.abs(p.y), half.y) * std.step(0, t);
  return d.vec4f(p.xy, inside, t);
});

/** The panel a map is printed on: dark fill, hairline border, the focus
 *  window, the marker where the window below converges, and the content,
 *  lit up inside the window and when the pass reaches the map. */
const shadeMap = tgpu.fn(
  [d.vec2f, d.vec2f, d.f32, d.vec2f, d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((localPosition, half, content, focus, windowHalf, markerStrength, pixel, flash) => {
  'use gpu';
  const edge = std.min(half.x - std.abs(localPosition.x), half.y - std.abs(localPosition.y));
  const border = 1 - std.smoothstep(pixel * 0.5, pixel * 1.5, edge);
  const q = localPosition / half;
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
)((rayOrigin, rayDirection, a, b, pixelAngle, base, pulseAt) => {
  'use gpu';
  const segment = b - a;
  const startToOrigin = rayOrigin - a;
  const rayAlongSegment = std.dot(rayDirection, segment);
  const segmentLengthSquared = std.dot(segment, segment);
  const u = std.clamp(
    (std.dot(segment, startToOrigin) - rayAlongSegment * std.dot(rayDirection, startToOrigin)) /
      std.max(segmentLengthSquared - rayAlongSegment * rayAlongSegment, 1e-5),
    0,
    1,
  );
  const onSegment = a + segment * u;
  const alongRay = std.max(std.dot(onSegment - rayOrigin, rayDirection), 0);
  const dist = std.length(rayOrigin + rayDirection * alongRay - onSegment);
  const width = alongRay * pixelAngle;
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
)((
  rayOrigin,
  rayDirection,
  layer,
  half,
  nextHalf,
  focus,
  nextFocus,
  windowHalf,
  progress,
  pixelAngle,
  time,
) => {
  'use gpu';
  const tip = mapPoint(nextFocus, nextHalf, layer + 1);
  let glow = segmentGlow(
    rayOrigin,
    rayDirection,
    mapPoint(focus + d.vec2f(-windowHalf, -windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    rayOrigin,
    rayDirection,
    mapPoint(focus + d.vec2f(windowHalf, -windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    rayOrigin,
    rayDirection,
    mapPoint(focus + d.vec2f(windowHalf, windowHalf), half, layer),
    tip,
    pixelAngle,
    0.5,
    progress,
  );
  glow += segmentGlow(
    rayOrigin,
    rayDirection,
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
    glow += segmentGlow(rayOrigin, rayDirection, a, b, pixelAngle, 0.14, travel) * 0.22;
    glow += segmentGlow(rayOrigin, rayDirection, a, b, pixelAngle, 0, progress) * 0.4;
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
  const params = layout.$.params;
  const resolution = params.resolution;
  // Units of half the short side, so a portrait screen still fits the stack.
  const shortSide = std.min(resolution.x, resolution.y);
  const screen =
    d.vec2f(pos.x * 2 - resolution.x, resolution.y - pos.y * 2) / shortSide - params.shift;

  const forward = std.normalize(params.lookAt - params.eye);
  const right = std.normalize(std.cross(d.vec3f(0, 1, 0), forward));
  const up = std.cross(forward, right);
  const rayOrigin = params.eye;
  const rayDirection = std.normalize(forward * FOCAL + right * screen.x + up * screen.y);
  const pixelAngle = 2 / (shortSide * FOCAL);

  const wave = std.fract(params.time / PASS_SECONDS) * 5.4 - 0.2;
  const focus = params.focus;
  // The input image drifts slowly and leans toward the focus.
  const drift = d.vec2f(std.sin(params.time * 0.13), std.cos(params.time * 0.11)) + focus * 0.3;

  const half0 = d.vec2f(HALF_SIZES[0][0], HALF_SIZES[0][1]);
  const half1 = d.vec2f(HALF_SIZES[1][0], HALF_SIZES[1][1]);
  const half2 = d.vec2f(HALF_SIZES[2][0], HALF_SIZES[2][1]);
  const half3 = d.vec2f(HALF_SIZES[3][0], HALF_SIZES[3][1]);
  const half4 = d.vec2f(HALF_SIZES[4][0], HALF_SIZES[4][1]);
  const winner = d.vec2f(0, WINNER_Y);

  // Back to front: each map covers the light behind it.
  let color = d.f32(0);
  const hit4 = hitMap(rayOrigin, rayDirection, 4 * LAYER_GAP, half4);
  if (hit4.z > 0) {
    const pulse = std.smoothstep(3.7, 4.1, wave);
    const content = outputMap(hit4.xy, pulse);
    const shade = shadeMap(hit4.xy, half4, content, winner, 0, 0, hit4.w * pixelAngle, 0);
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    rayOrigin,
    rayDirection,
    3,
    half3,
    half4,
    focus,
    winner,
    WINDOW_HALF[3],
    wave - 3,
    pixelAngle,
    params.time,
  );
  const hit3 = hitMap(rayOrigin, rayDirection, 3 * LAYER_GAP, half3);
  if (hit3.z > 0) {
    const content = sparseMap(hit3.xy / half3, drift, focus);
    const shade = shadeMap(
      hit3.xy,
      half3,
      content,
      focus,
      WINDOW_HALF[3],
      0.5,
      hit3.w * pixelAngle,
      flashAt(wave, 3),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    rayOrigin,
    rayDirection,
    2,
    half2,
    half3,
    focus,
    focus,
    WINDOW_HALF[2],
    wave - 2,
    pixelAngle,
    params.time,
  );
  const hit2 = hitMap(rayOrigin, rayDirection, 2 * LAYER_GAP, half2);
  if (hit2.z > 0) {
    const content = pooledMap(hit2.xy / half2, drift);
    const shade = shadeMap(
      hit2.xy,
      half2,
      content,
      focus,
      WINDOW_HALF[2],
      0.5,
      hit2.w * pixelAngle,
      flashAt(wave, 2),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    rayOrigin,
    rayDirection,
    1,
    half1,
    half2,
    focus,
    focus,
    WINDOW_HALF[1],
    wave - 1,
    pixelAngle,
    params.time,
  );
  const hit1 = hitMap(rayOrigin, rayDirection, LAYER_GAP, half1);
  if (hit1.z > 0) {
    const content = edgeMap(hit1.xy / half1, drift);
    const shade = shadeMap(
      hit1.xy,
      half1,
      content,
      focus,
      WINDOW_HALF[1],
      0.5,
      hit1.w * pixelAngle,
      flashAt(wave, 1),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }
  color += gapGlow(
    rayOrigin,
    rayDirection,
    0,
    half0,
    half1,
    focus,
    focus,
    WINDOW_HALF[0],
    wave,
    pixelAngle,
    params.time,
  );
  const hit0 = hitMap(rayOrigin, rayDirection, 0, half0);
  if (hit0.z > 0) {
    const content = inputImage(hit0.xy / half0, drift);
    const shade = shadeMap(
      hit0.xy,
      half0,
      content,
      focus,
      WINDOW_HALF[0],
      0,
      hit0.w * pixelAngle,
      flashAt(wave, 0),
    );
    color = std.mix(color, shade, MAP_OPACITY);
  }

  const uv = pos.xy / resolution;
  const vignette =
    1 - 0.55 * std.smoothstep(0.35, 0.95, std.length((uv - 0.5) * d.vec2f(1.1, 1.4)));
  color = std.clamp(color * vignette, 0, 1);

  // Ordered dither with a little moving grain, so it reads as print, not glow.
  const grain = (hash21(pos.xy + std.fract(params.time * 7.3) * 91) - 0.5) * 0.35;
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
/** Camera drift: seconds of period per axis, and how far it swings, in world units. */
const DRIFT_PERIODS = [41, 29, 53] as const;
const DRIFT_SWING = [0.45, 0.3, 0.3] as const;
/** How far the camera leans with the focus, in world units per unit of focus. */
const FOCUS_LEAN = [0.25, 0.2] as const;
/** How far the camera moves into the stack over the whole page scroll, in world units. */
const SCROLL_DOLLY = [-1.6, -0.3, 2.2] as const;
/** How fast the focus and the scroll position are followed, per second. */
const FOCUS_EASE = 3;
const SCROLL_EASE = 4;
/** The idle focus: a slow figure-eight over the input image. Amplitude per
 *  axis, in map halves, and rate per axis, in radians per second. */
const WANDER_AMPLITUDE = [0.55, 0.35] as const;
const WANDER_RATE = [0.37, 0.53] as const;
/** How far the pointer can move the focus from the center, in map halves. */
const POINTER_REACH = 0.75;
/** Aspect ratio above which the stack sits right of the copy instead of above it. */
const WIDE_ASPECT = 1.1;
/** Where the stack sits on wide and narrow screens, in halves of the short side.
 *  On wide screens x grows with the aspect ratio, up to WIDE_SHIFT_MAX_X. */
const WIDE_SHIFT_PER_ASPECT = 0.55;
const WIDE_SHIFT_MAX_X = 1.2;
const WIDE_SHIFT_Y = 0.2;
const NARROW_SHIFT = [0.65, 1.1] as const;

/** A frozen moment for reduced motion: the pass mid-way through the network. */
const STILL_TIME = 2.1;

/** Starts the network on `canvas` and returns a function that stops it.
 *  Throws when the shader does not resolve; WGSL validation errors arrive
 *  later, on the device's error scope. */
export function startHero(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  { still }: { still: boolean },
): () => void {
  const context = root.configureContext({ canvas, alphaMode: 'opaque' });
  const paramsBuffer = root.createBuffer(Params).$usage('uniform');
  const pipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: heroFragment,
      targets: { format: navigator.gpu.getPreferredCanvasFormat() },
    })
    .with(root.createBindGroup(layout, { params: paramsBuffer }));

  // Reused every frame, so drawing allocates nothing.
  const values = {
    resolution: d.vec2f(),
    time: 0,
    focus: d.vec2f(),
    eye: d.vec3f(),
    lookAt: d.vec3f(...CAMERA_LOOK_AT),
    shift: d.vec2f(),
  };
  let pointer: [number, number] | undefined;
  let scroll = 0;

  const scrollProgress = () => {
    const range = document.documentElement.scrollHeight - innerHeight;
    return range > 0 ? Math.min(Math.max(scrollY / range, 0), 1) : 0;
  };

  const draw = (time: number, dt: number) => {
    const { focus, eye, shift, resolution } = values;
    // Without a pointer the focus wanders over the image on its own.
    const targetX = pointer?.[0] ?? WANDER_AMPLITUDE[0] * Math.sin(time * WANDER_RATE[0]);
    const targetY = pointer?.[1] ?? WANDER_AMPLITUDE[1] * Math.sin(time * WANDER_RATE[1] + 1);
    const focusEase = 1 - Math.exp(-FOCUS_EASE * dt);
    focus.x += (targetX - focus.x) * focusEase;
    focus.y += (targetY - focus.y) * focusEase;
    // A still frame jumps straight to the scroll position.
    scroll += (scrollProgress() - scroll) * (dt > 0 ? 1 - Math.exp(-SCROLL_EASE * dt) : 1);

    for (let axis = 0; axis < 3; axis++) {
      const swing = DRIFT_SWING[axis]! * Math.sin((2 * Math.PI * time) / DRIFT_PERIODS[axis]!);
      const lean = axis < 2 ? FOCUS_LEAN[axis]! * focus[axis]! : 0;
      eye[axis] = CAMERA_EYE[axis]! + swing + lean + SCROLL_DOLLY[axis]! * scroll;
    }

    resolution.x = canvas.width;
    resolution.y = canvas.height;
    const aspect = canvas.width / canvas.height;
    // Wide screens put the stack right of the copy, narrow ones above it.
    shift.x =
      aspect > WIDE_ASPECT
        ? Math.min(aspect * WIDE_SHIFT_PER_ASPECT, WIDE_SHIFT_MAX_X)
        : NARROW_SHIFT[0];
    shift.y = aspect > WIDE_ASPECT ? WIDE_SHIFT_Y : NARROW_SHIFT[1];
    values.time = time;

    paramsBuffer.write(values);
    pipeline.withColorAttachment({ view: context }).draw(3);
  };

  const onPointer = (event: PointerEvent) => {
    const x = (event.clientX / innerWidth) * 2 - 1;
    const y = 1 - (event.clientY / innerHeight) * 2;
    const clamp = (v: number) => Math.max(-POINTER_REACH, Math.min(POINTER_REACH, v));
    pointer = [clamp(x), clamp(y)];
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
