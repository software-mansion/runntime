import { d } from 'typegpu';
import { describe, it, onTestFinished } from 'vitest';
import { createMapToTexture } from '../../../../src/core/kernels/image/mapToTexture.ts';
import { expectClose, gpu, randomF32 } from '../../../gpu.ts';

const { root } = await gpu();
const mapToTexture = createMapToTexture(root);

/** Half-pixel bilinear resample of `map` to `outW`×`outH`, edges clamped,
 *  from the definition. Row 0 is the top of the image in both. */
function resampleRef(map: Float32Array, w: number, h: number, outW: number, outH: number) {
  const at = (x: number, y: number) =>
    map[Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)]!;
  const out = new Float64Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const sx = ((x + 0.5) / outW) * w - 0.5;
      const sy = ((y + 0.5) / outH) * h - 0.5;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx;
      const bot = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx;
      out[y * outW + x] = top * (1 - fy) + bot * fy;
    }
  }
  return out;
}

/** Draws `map` into an `outW`×`outH` rgba32float texture and reads back the
 *  red channel, row 0 first. `outW` * 16 must be a multiple of 256. */
async function draw(
  map: Float32Array,
  w: number,
  h: number,
  outW: number,
  outH: number,
  normalize: 'none' | 'minmax',
): Promise<Float32Array> {
  const buffer = root.createBuffer(d.arrayOf(d.f32, map.length), Array.from(map)).$usage('storage');
  const target = root.device.createTexture({
    size: [outW, outH],
    format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback = root.device.createBuffer({
    size: outW * outH * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  onTestFinished(() => {
    buffer.destroy();
    target.destroy();
    readback.destroy();
  });
  const enc = root.device.createCommandEncoder();
  mapToTexture.encode(enc, buffer, target, { width: w, height: h, normalize });
  enc.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: outW * 16 }, [
    outW,
    outH,
  ]);
  root.device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const rgba = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return rgba.filter((_, i) => i % 4 === 0);
}

describe('mapToTexture kernel', () => {
  it.each([
    // Same size: each pixel center lands on one map pixel, so this pins the
    // orientation (no flip in x or y) as well as the values.
    { w: 16, h: 16, outW: 16, outH: 16 },
    { w: 16, h: 9, outW: 16, outH: 9 },
    // Upsampling: bilinear between neighbours, clamped at the edges.
    { w: 5, h: 7, outW: 16, outH: 12 },
    // Downsampling.
    { w: 40, h: 33, outW: 16, outH: 8 },
    { w: 1, h: 1, outW: 16, outH: 4 },
  ])('normalize none, $w×$h to $outW×$outH', async ({ w, h, outW, outH }) => {
    const map = randomF32(w * h, w * 31 + h);
    const got = await draw(map, w, h, outW, outH, 'none');
    expectClose(got, resampleRef(map, w, h, outW, outH), { atol: 1e-5, rtol: 1e-5 });
  });

  it.each([
    { w: 16, h: 16, outW: 16, outH: 16 },
    // 300 values: more than one workgroup's worth, so threads stride the map.
    { w: 20, h: 15, outW: 32, outH: 8 },
  ])('normalize minmax, $w×$h to $outW×$outH', async ({ w, h, outW, outH }) => {
    const map = randomF32(w * h, 7).map((v) => v * 50 + 3);
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of map) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    const want = resampleRef(map, w, h, outW, outH).map((v) => (v - lo) / (hi - lo));
    const got = await draw(map, w, h, outW, outH, 'minmax');
    expectClose(got, want, { atol: 1e-5, rtol: 1e-5 });
  });

  it('normalize minmax draws a flat map as 0', async () => {
    const got = await draw(new Float32Array(64).fill(2.5), 8, 8, 16, 16, 'minmax');
    expectClose(got, new Float32Array(256), { atol: 0, rtol: 0 });
  });
});
