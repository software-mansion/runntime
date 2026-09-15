import {
  createKeypointDetector,
  models,
  type KeypointDetection,
  type KeypointDetector,
  type LoadOptions,
} from 'runntime/zoo';
import { useCallback } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Landmarks below this are drawn neither as dots nor as limb ends. */
const VISIBLE = 0.5;

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadDetector = (opts: LoadOptions) =>
  createKeypointDetector(models.keypointDetection.YOLO26_POSE.DEFAULT, opts);

function Skeletons({ detector }: { detector: KeypointDetector }) {
  const run = useCallback(
    (frame: Parameters<KeypointDetector['detectKeypoints']>[0]) => detector.detectKeypoints(frame),
    [detector],
  );

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, people: KeypointDetection[]) => {
      const scale = Math.max(1, ctx.canvas.width / 640);
      ctx.lineWidth = 2 * scale;
      ctx.lineCap = 'round';
      people.forEach((person, i) => {
        const hue = (i * 67) % 360;
        ctx.strokeStyle = `hsl(${hue} 80% 55%)`;
        ctx.fillStyle = `hsl(${hue} 80% 55%)`;
        ctx.beginPath();
        for (const [a, b] of detector.skeleton) {
          const from = person.landmarks[a];
          const to = person.landmarks[b];
          if (from.confidence < VISIBLE || to.confidence < VISIBLE) continue;
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
        }
        ctx.stroke();
        for (const mark of Object.values(person.landmarks)) {
          if (mark.confidence < VISIBLE) continue;
          ctx.beginPath();
          ctx.arc(mark.x, mark.y, 3 * scale, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    },
    [detector],
  );

  const stats = useCallback(
    (people: KeypointDetection[]) =>
      `${people.length} ${people.length === 1 ? 'person' : 'people'} · yolo26n pose`,
    [],
  );

  return <CameraDemo run={run} draw={draw} stats={stats} />;
}

export function Keypoints() {
  const { state, start } = useModel(loadDetector);
  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Fetches YOLO26n pose, 6 MB, from the Hugging Face Hub."
    >
      {(detector) => <Skeletons detector={detector} />}
    </WeightsGate>
  );
}
