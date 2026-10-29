import { createDepthEstimator, models, type DepthEstimator, type LoadOptions } from 'runntime/zoo';
import { useCallback } from 'react';
import { CameraDemo } from './CameraDemo.tsx';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadEstimator = (opts: LoadOptions) =>
  createDepthEstimator(models.depthEstimation.DEPTHART.DEFAULT, opts);

function Map({ estimator }: { estimator: DepthEstimator }) {
  // Frame in, map in the canvas, both on the GPU: nearest black, farthest white.
  const encode = useCallback(
    (source: GPUExternalTexture, target: GPUTexture) =>
      estimator.encodeEstimateDepth({ source, target, normalize: 'minmax' }),
    [estimator],
  );

  return (
    <CameraDemo
      encode={encode}
      stats={`${estimator.inputSize}×${estimator.inputSize} map · depthart s`}
    />
  );
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
