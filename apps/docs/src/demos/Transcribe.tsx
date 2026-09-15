import { createSpeechToText, models, type LoadOptions, type SpeechToText } from 'runntime/zoo';
import { useEffect, useState } from 'react';
import { startRecording, type Recording } from './recorder.ts';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Long enough for a sentence or two, short enough that a forgotten
 *  recording does not grow without bound. */
const MAX_SECONDS = 20;

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadSpeechToText = (opts: LoadOptions) =>
  createSpeechToText(models.speechToText.MOONSHINE.TINY.F16, opts);

function Recorder({ stt }: { stt: SpeechToText }) {
  const [recording, setRecording] = useState<Recording>();
  const [text, setText] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [transcribeMs, setTranscribeMs] = useState<number>();
  const [error, setError] = useState<string>();

  const stop = async (active: Recording) => {
    setRecording(undefined);
    setBusy(true);
    try {
      const audio = await active.stop();
      const start = performance.now();
      const spoken = await stt.transcribe(audio);
      setTranscribeMs(performance.now() - start);
      setText(spoken);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    setError(undefined);
    try {
      setRecording(await startRecording());
    } catch {
      setError('No microphone. Check the permission the browser asked for.');
    }
  };

  // Stop on its own, so a recording left running does not fill memory.
  useEffect(() => {
    if (!recording) return;
    const timer = setTimeout(() => void stop(recording), MAX_SECONDS * 1000);
    return () => clearTimeout(timer);
  }, [recording]);

  return (
    <div className="demo not-content">
      <div className="demo-row">
        <button
          className="demo-button"
          onClick={() => (recording ? void stop(recording) : void start())}
          disabled={busy}
        >
          {recording ? '■ Stop' : busy ? 'Transcribing…' : '● Record'}
        </button>
      </div>

      <p className="transcript">
        {recording
          ? `Listening — say something, then stop. Ends on its own after ${MAX_SECONDS} s.`
          : (text ?? 'Record a few words to see them transcribed here.')}
      </p>

      {error && <p className="demo-error">{error}</p>}

      <div className="demo-stats">
        <span>moonshine tiny</span>
        <span>16 kHz mono</span>
        <span>transcribe {transcribeMs === undefined ? '—' : `${transcribeMs.toFixed(0)} ms`}</span>
      </div>
    </div>
  );
}

export function Transcribe() {
  const { state, start } = useModel(loadSpeechToText);

  // The runner owns GPU buffers; release them when the page unmounts.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const stt = state.model;
    return () => stt.dispose();
  }, [state]);

  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Moonshine tiny, 54 MB of safetensors from the Hugging Face Hub, run in f16 on the GPU. Cached in OPFS after the first load."
    >
      {(stt) => <Recorder stt={stt} />}
    </WeightsGate>
  );
}
