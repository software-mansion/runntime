import {
  createObjectDetector,
  models,
  type LoadOptions,
  type ObjectDetection,
  type ObjectDetector,
} from 'runntime/zoo';
import { useCallback } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadDetector = (opts: LoadOptions) =>
  createObjectDetector(models.objectDetection.YOLO26.DEFAULT, opts);

function Boxes({ detector }: { detector: ObjectDetector }) {
  const run = useCallback(
    (frame: Parameters<ObjectDetector['detectObjects']>[0]) => detector.detectObjects(frame),
    [detector],
  );

  const draw = useCallback((ctx: CanvasRenderingContext2D, objects: ObjectDetection[]) => {
    const scale = Math.max(1, ctx.canvas.width / 640);
    ctx.lineWidth = 2 * scale;
    ctx.font = `${13 * scale}px sans-serif`;
    ctx.textBaseline = 'top';
    for (const { label, classId, confidence, box } of objects) {
      const hue = (classId * 47) % 360;
      ctx.strokeStyle = `hsl(${hue} 80% 50%)`;
      ctx.fillStyle = `hsl(${hue} 80% 50%)`;
      const w = box.xmax - box.xmin;
      const h = box.ymax - box.ymin;
      ctx.strokeRect(box.xmin, box.ymin, w, h);
      const text = `${label} ${Math.round(confidence * 100)}%`;
      const pad = 3 * scale;
      const tw = ctx.measureText(text).width + 2 * pad;
      const th = 16 * scale;
      ctx.fillRect(box.xmin, box.ymin, tw, th);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, box.xmin + pad, box.ymin + pad / 2);
    }
  }, []);

  const stats = useCallback(
    (objects: ObjectDetection[]) =>
      `${objects.length} ${objects.length === 1 ? 'object' : 'objects'} · yolo26n`,
    [],
  );

  return <CameraDemo run={run} draw={draw} stats={stats} />;
}

export function Detect() {
  const { state, start } = useModel(loadDetector);
  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Fetches YOLO26n, 5 MB, from the Hugging Face Hub."
    >
      {(detector) => <Boxes detector={detector} />}
    </WeightsGate>
  );
}
