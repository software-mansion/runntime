/** Text to speech demo: Reverb speaks the text in one of its preset voices. */
import {
  createTextToSpeech,
  models,
  type LoadOptions,
  type ReverbLanguage,
  type TextToSpeech,
  type Voice,
} from 'runntime/zoo';
import { useEffect, useState } from 'react';
import { Verbalizer } from 'verbalis';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

/** Reverb's preset voices: codec codes of a reference clip with its
 *  transcript, a few KB of JSON each. */
const VOICES_URL = 'https://huggingface.co/software-mansion/Reverb-TTS/resolve/main/voices';

const VOICES: { name: string; label: string; language: ReverbLanguage }[] = [
  { name: 'en_male_narrator_1', label: 'English, male narrator', language: 'en' },
  { name: 'en_female_narrator', label: 'English, female narrator', language: 'en' },
  { name: 'en_female_whisper', label: 'English, female whisper', language: 'en' },
  { name: 'en_male_angry', label: 'English, male angry', language: 'en' },
  { name: 'de_female_standard', label: 'German, female', language: 'de' },
  { name: 'de_male_standard', label: 'German, male', language: 'de' },
  { name: 'fr_female_standard', label: 'French, female', language: 'fr' },
  { name: 'fr_male_standard', label: 'French, male', language: 'fr' },
  { name: 'es_male_standard_1', label: 'Spanish, male', language: 'es' },
  { name: 'it_male_expressive', label: 'Italian, male expressive', language: 'it' },
  { name: 'it_male_standard', label: 'Italian, male', language: 'it' },
  { name: 'pl_male_standard', label: 'Polish, male', language: 'pl' },
  { name: 'pt_female_standard', label: 'Portuguese, female', language: 'pt' },
  { name: 'pt_male_standard', label: 'Portuguese, male', language: 'pt' },
];

/** The text the box starts with, in the voice's language: a number and a
 *  unit, so verbalize has something to read out. */
const SAMPLES: Record<ReverbLanguage, string> = {
  en: 'Good morning! It is 21 °C outside, so take a light jacket.',
  de: 'Guten Morgen! Draußen sind es 21 °C, nimm also eine leichte Jacke mit.',
  fr: 'Bonjour ! Il fait 21 °C dehors, prends donc une veste légère.',
  es: '¡Buenos días! Fuera hace 21 °C, así que lleva una chaqueta ligera.',
  it: 'Buongiorno! Fuori ci sono 21 °C, quindi prendi una giacca leggera.',
  pl: 'Dzień dobry! Na zewnątrz jest 21 °C, więc weź lekką kurtkę.',
  pt: 'Bom dia! Lá fora estão 21 °C, por isso leva um casaco leve.',
};

/** A preset voice file as the Hub stores it. */
interface VoiceFile {
  codes: number[];
  frames: number;
  transcript: string;
  speech_frames: number;
}

async function fetchVoice(name: string, language: ReverbLanguage): Promise<Voice> {
  const res = await fetch(`${VOICES_URL}/${name}.json`);
  if (!res.ok) throw new Error(`Could not fetch the voice ${name}: HTTP ${res.status}.`);
  const file: VoiceFile = await res.json();
  return {
    codes: Int32Array.from(file.codes),
    frames: file.frames,
    transcript: file.transcript,
    language,
    speechFrames: file.speech_frames,
  };
}

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadTextToSpeech = (opts: LoadOptions) =>
  createTextToSpeech(
    {
      ...models.textToSpeech.REVERB.DEFAULT,
      verbalize: (text, language) => new Verbalizer(language).process(text),
    },
    opts,
  );

function play(samples: Float32Array, sampleRate: number) {
  const context = new AudioContext({ sampleRate });
  const buffer = context.createBuffer(1, samples.length, sampleRate);
  buffer.getChannelData(0).set(samples);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.onended = () => void context.close();
  source.start();
}

function Speaker({ tts }: { tts: TextToSpeech }) {
  const [voiceIndex, setVoiceIndex] = useState(0);
  const [text, setText] = useState(SAMPLES.en);
  const [audio, setAudio] = useState<Float32Array>();
  const [busy, setBusy] = useState(false);
  const [synthesizeMs, setSynthesizeMs] = useState<number>();
  const [error, setError] = useState<string>();
  const voice = VOICES[voiceIndex]!;

  const pickVoice = (index: number) => {
    setVoiceIndex(index);
    setText(SAMPLES[VOICES[index]!.language]);
  };

  const speak = async () => {
    setBusy(true);
    setError(undefined);
    try {
      const reference = await fetchVoice(voice.name, voice.language);
      const start = performance.now();
      const spoken = await tts.synthesize(text, reference);
      setSynthesizeMs(performance.now() - start);
      setAudio(spoken);
      play(spoken, tts.sampleRate);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const seconds = audio ? audio.length / tts.sampleRate : 0;

  return (
    <div className="demo not-content">
      <select
        className="demo-select"
        value={voiceIndex}
        onChange={(e) => pickVoice(Number(e.target.value))}
        aria-label="Voice"
      >
        {VOICES.map((v, i) => (
          <option key={v.name} value={i}>
            {v.label}
          </option>
        ))}
      </select>

      <textarea
        className="demo-input demo-textarea demo-row-actions"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        aria-label="Text to speak"
      />

      <div className="demo-row demo-row-actions">
        <button
          className="demo-button"
          onClick={() => void speak()}
          disabled={busy || !text.trim()}
        >
          {busy ? 'Speaking…' : '▶ Speak'}
        </button>
        <button
          className="demo-button demo-button-quiet"
          onClick={() => audio && play(audio, tts.sampleRate)}
          disabled={!audio || busy}
        >
          Play again
        </button>
      </div>

      {error && <p className="demo-error">{error}</p>}

      <div className="demo-stats">
        <span>reverb int4</span>
        <span>24 kHz mono</span>
        <span>audio {audio ? `${seconds.toFixed(1)} s` : '—'}</span>
        <span>synthesize {synthesizeMs === undefined ? '—' : `${synthesizeMs.toFixed(0)} ms`}</span>
        <span>
          {synthesizeMs === undefined ? '—' : `${(seconds / (synthesizeMs / 1000)).toFixed(1)}×`}{' '}
          realtime
        </span>
      </div>
    </div>
  );
}

export function Speak() {
  const { state, start } = useModel(loadTextToSpeech);

  // The runner owns GPU buffers; release them when the page unmounts.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const tts = state.model;
    return () => tts.dispose();
  }, [state]);

  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="Reverb, 217 MB of safetensors from the Hugging Face Hub: int4 weights and the codec decoder, run on the GPU. Cached in OPFS after the first load."
    >
      {(tts) => <Speaker tts={tts} />}
    </WeightsGate>
  );
}
