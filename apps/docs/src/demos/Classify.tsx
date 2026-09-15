import { createImageClassifier, models, type ImageClassifier, type LoadOptions } from 'runntime/zoo';
import { useCallback } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadClassifier = (opts: LoadOptions) =>
  createImageClassifier(models.imageClassification.MOBILENETV4.DEFAULT, opts);

/** How many classes the demo lists under the frame. */
const TOP_K = 5;

type Class = Awaited<ReturnType<ImageClassifier['classify']>>[number];

function Ranked({ classifier }: { classifier: ImageClassifier }) {
  const run = useCallback(
    (frame: Parameters<ImageClassifier['classify']>[0]) =>
      classifier.classify(frame, { topk: TOP_K }),
    [classifier],
  );

  // The classifier crops the centered square, so the sides of the frame
  // never reach the model. The outline shows where to hold things.
  const draw = useCallback((ctx: CanvasRenderingContext2D) => {
    const side = Math.min(ctx.canvas.width, ctx.canvas.height);
    const x = (ctx.canvas.width - side) / 2;
    const y = (ctx.canvas.height - side) / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(x + 1, y + 1, side - 2, side - 2);
  }, []);

  const below = useCallback(
    (classes: Class[]) => (
      <ul className="hits">
        {classes.map(({ label, classId, confidence }) => (
          <li className="hit" key={classId}>
            <span className="hit-text">{label}</span>
            <span className="hit-track">
              <span className="hit-bar" style={{ width: `${confidence * 100}%` }} />
            </span>
            <span className="hit-score">{confidence.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    ),
    [],
  );

  const stats = useCallback(
    (classes: Class[]) => `${classes[0]?.label ?? 'nothing'} · mobilenetv4 conv-s`,
    [],
  );

  return <CameraDemo run={run} draw={draw} below={below} stats={stats} />;
}

export function Classify() {
  const { state, start } = useModel(loadClassifier);
  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Fetches MobileNetV4 Conv-S, 10 MB, from the Hugging Face Hub."
    >
      {(classifier) => <Ranked classifier={classifier} />}
    </WeightsGate>
  );
}
