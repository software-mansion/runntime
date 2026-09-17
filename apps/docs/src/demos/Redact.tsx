import {
  createPrivacyFilter,
  models,
  type LoadOptions,
  type PrivacyFilter,
  type PrivacySpan,
} from 'runntime/zoo';
import { useEffect, useState, type ReactNode } from 'react';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

const SAMPLE = `Hi, this is Jane Doe. You can reach me at jane.doe@example.com or on +1 415 555 0134.
I moved to 221B Baker Street, London on March 3rd. My account is GB29 NWBK 6016 1331 9268 19,
and the API key for the staging box is sk-live-3f9a2c1d84e7.`;

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadPrivacyFilter = (opts: LoadOptions) =>
  createPrivacyFilter(models.privacyFilter.PRIVACY_FILTER.DEFAULT, opts);

/** The text broken into runs, so the spans can be marked in place: the gaps
 *  between spans are plain, each span is highlighted or swapped for its
 *  placeholder. */
function render(text: string, spans: PrivacySpan[], redact: boolean): ReactNode[] {
  const out: ReactNode[] = [];
  let at = 0;
  for (const [i, span] of spans.entries()) {
    if (span.start > at) out.push(text.slice(at, span.start));
    out.push(
      redact ? (
        <span className="pii pii-redacted" key={i}>
          {span.placeholder}
        </span>
      ) : (
        <mark className="pii" key={i}>
          {span.text}
          <span className="pii-label">{span.label.replace(/^private_/, '')}</span>
        </mark>
      ),
    );
    at = span.end;
  }
  if (at < text.length) out.push(text.slice(at));
  return out;
}

function Scanner({ filter }: { filter: PrivacyFilter }) {
  const [text, setText] = useState(SAMPLE);
  const [scanned, setScanned] = useState<string>();
  const [spans, setSpans] = useState<PrivacySpan[]>([]);
  const [redact, setRedact] = useState(false);
  const [busy, setBusy] = useState(false);
  const [scanMs, setScanMs] = useState<number>();

  const scan = async (input: string) => {
    setBusy(true);
    try {
      const start = performance.now();
      const found = await filter.detect(input);
      setScanMs(performance.now() - start);
      setSpans(found);
      setScanned(input);
    } finally {
      setBusy(false);
    }
  };

  // One pass over the sample, so the panel is not empty.
  useEffect(() => {
    void scan(SAMPLE);
  }, [filter]);

  return (
    <div className="demo not-content">
      <textarea
        className="demo-input demo-textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        aria-label="Text to scan"
      />

      <div className="demo-row demo-row-actions">
        <button
          className="demo-button"
          onClick={() => void scan(text)}
          disabled={busy || !text.trim()}
        >
          {busy ? 'Scanning…' : 'Scan'}
        </button>
        <button
          className="demo-button demo-button-quiet"
          onClick={() => setRedact(!redact)}
          disabled={spans.length === 0}
        >
          {redact ? 'Show matches' : 'Redact'}
        </button>
      </div>

      <p className="transcript pii-text">
        {scanned === undefined
          ? 'Scanning the sample…'
          : spans.length === 0
            ? 'No personal data found in this text.'
            : render(scanned, spans, redact)}
      </p>

      <div className="demo-stats">
        <span>privacy-filter int8</span>
        <span>{spans.length} spans</span>
        <span>scan {scanMs === undefined ? '—' : `${scanMs.toFixed(0)} ms`}</span>
      </div>
    </div>
  );
}

export function Redact() {
  const { state, start } = useModel(loadPrivacyFilter);

  // The filter owns GPU buffers; release them when the page unmounts.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const filter = state.model;
    return () => filter.dispose();
  }, [state]);

  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="privacy-filter, ⚠️ 1.55 GB of safetensors from the Hugging Face Hub, run in int8 on the GPU. Cached in OPFS after the first load."
    >
      {(filter) => <Scanner filter={filter} />}
    </WeightsGate>
  );
}
