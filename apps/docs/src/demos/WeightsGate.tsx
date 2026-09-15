import type { ReactNode } from 'react';
import type { ModelState } from './useModel.ts';

/** The download button, the progress bar and the load error, in one place.
 *  Renders the demo only once the model is on the GPU. */
export function WeightsGate<T>({
  state,
  onStart,
  note,
  children,
}: {
  state: ModelState<T>;
  onStart: () => void;
  /** One line under the button: what is fetched and how big it is. */
  note: string;
  children: (model: T) => ReactNode;
}) {
  if (state.status === 'ready') return children(state.model);

  return (
    <div className="demo demo-gate not-content">
      <button className="demo-button" onClick={onStart} disabled={state.status === 'loading'}>
        {state.status === 'loading' ? 'Downloading…' : 'Download weights'}
      </button>
      {state.status === 'loading' ? (
        <div className="progress">
          <div className="progress-fill" style={{ width: `${Math.round(state.percent * 100)}%` }} />
        </div>
      ) : (
        <p className="demo-note">{note}</p>
      )}
      {state.status === 'error' && <p className="demo-error">{state.message}</p>}
    </div>
  );
}
