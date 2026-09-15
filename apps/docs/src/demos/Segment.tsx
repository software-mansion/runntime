import {
  createInstanceSegmenter,
  models,
  type InstanceSegmentation,
  type InstanceSegmenter,
  type LoadOptions,
} from 'runntime/zoo';
import { useCallback, useMemo } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadSegmenter = (opts: LoadOptions) =>
  createInstanceSegmenter(models.instanceSegmentation.YOLO26_SEG.DEFAULT, opts);

function Masks({ segmenter }: { segmenter: InstanceSegmenter }) {
  const run = useCallback(
    (frame: Parameters<InstanceSegmenter['segmentInstances']>[0]) =>
      segmenter.segmentInstances(frame),
    [segmenter],
  );

  // One scratch canvas for every mask: the gray coverage image becomes an
  // ImageData here, then drawImage stretches it over the object's box.
  const layer = useMemo(() => new OffscreenCanvas(1, 1), []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, objects: InstanceSegmentation[]) => {
      const layerCtx = layer.getContext('2d')!;
      const scale = Math.max(1, ctx.canvas.width / 640);
      ctx.font = `${13 * scale}px sans-serif`;
      ctx.textBaseline = 'top';
      for (const { label, classId, mask, box } of objects) {
        const hue = (classId * 47) % 360;
        layer.width = mask.width;
        layer.height = mask.height;
        const pixels = layerCtx.createImageData(mask.width, mask.height);
        const [r, g, b] = hslToRgb(hue, 0.8, 0.55);
        for (let i = 0; i < mask.data.length; i++) {
          pixels.data[i * 4] = r;
          pixels.data[i * 4 + 1] = g;
          pixels.data[i * 4 + 2] = b;
          pixels.data[i * 4 + 3] = mask.data[i]! >> 1;
        }
        layerCtx.putImageData(pixels, 0, 0);
        ctx.drawImage(layer, box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);
        ctx.fillStyle = `hsl(${hue} 80% 40%)`;
        ctx.fillText(label, box.xmin + 4 * scale, box.ymin + 4 * scale);
      }
    },
    [layer],
  );

  const stats = useCallback(
    (objects: InstanceSegmentation[]) =>
      `${objects.length} ${objects.length === 1 ? 'object' : 'objects'} · yolo26n segment`,
    [],
  );

  return <CameraDemo run={run} draw={draw} stats={stats} />;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}

export function Segment() {
  const { state, start } = useModel(loadSegmenter);
  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Fetches YOLO26n segment, 6 MB, from the Hugging Face Hub."
    >
      {(segmenter) => <Masks segmenter={segmenter} />}
    </WeightsGate>
  );
}
