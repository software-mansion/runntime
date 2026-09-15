import {
  createDepthEstimator,
  models,
  type DepthEstimator,
  type DepthMap,
  type LoadOptions,
} from 'runntime/zoo';
import { useCallback, useMemo } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadEstimator = (opts: LoadOptions) =>
  createDepthEstimator(models.depthEstimation.DEPTHART.DEFAULT, opts);

/** Near to far: warm to cold. Each stop is r, g, b. */
const STOPS: [number, number, number][] = [
  [250, 220, 90],
  [230, 90, 60],
  [110, 40, 120],
  [20, 20, 60],
];

function shade(t: number): [number, number, number] {
  const x = t * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = STOPS[i]!;
  const b = STOPS[i + 1]!;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function Map({ estimator }: { estimator: DepthEstimator }) {
  const run = useCallback(
    (frame: Parameters<DepthEstimator['estimateDepth']>[0]) => estimator.estimateDepth(frame),
    [estimator],
  );

  // The map is painted here at its own size, then stretched over the frame.
  const layer = useMemo(() => new OffscreenCanvas(1, 1), []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, depth: DepthMap) => {
      layer.width = depth.width;
      layer.height = depth.height;
      const layerCtx = layer.getContext('2d')!;
      const pixels = layerCtx.createImageData(depth.width, depth.height);
      let min = Infinity;
      let max = -Infinity;
      for (const v of depth.data) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const inv = max > min ? 1 / (max - min) : 0;
      for (let i = 0; i < depth.data.length; i++) {
        // Smaller = nearer, so the nearest pixels take the first, warm stop.
        const [r, g, b] = shade((depth.data[i]! - min) * inv);
        pixels.data[i * 4] = r;
        pixels.data[i * 4 + 1] = g;
        pixels.data[i * 4 + 2] = b;
        pixels.data[i * 4 + 3] = 255;
      }
      layerCtx.putImageData(pixels, 0, 0);
      ctx.drawImage(layer, 0, 0, ctx.canvas.width, ctx.canvas.height);
    },
    [layer],
  );

  const stats = useCallback(
    (depth: DepthMap) => `${depth.width}×${depth.height} map · depthart s`,
    [],
  );

  return <CameraDemo run={run} draw={draw} stats={stats} />;
}

export function Depth() {
  const { state, start } = useModel(loadEstimator);
  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Fetches DepthART s, 13 MB, from the Hugging Face Hub."
    >
      {(estimator) => <Map estimator={estimator} />}
    </WeightsGate>
  );
}
