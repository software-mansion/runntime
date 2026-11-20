/** Text to speech demo: Reverb speaks the text in one of its preset voices,
 *  or in the user's own, cloned from the microphone. */
import {
  createTextToSpeech,
  createVoiceMaker,
  models,
  type LoadOptions,
  type ReverbLanguage,
  type TextToSpeech,
  type Voice,
  type VoiceMaker,
} from 'runntime/zoo';
import { useEffect, useRef, useState } from 'react';
import { Verbalizer } from 'verbalis';
import { startRecording, type Recording } from './recorder.ts';
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

/** The text the box starts with, in the voice's language: about ten seconds
 *  of speech, with a number and a unit so verbalize has something to read
 *  out. */
const SAMPLES: Record<ReverbLanguage, string> = {
  en: 'Good morning! It is 21 °C outside and the sky is clear, so a light jacket will do. The train to the city centre leaves in 15 minutes, just enough time to finish your coffee.',
  de: 'Guten Morgen! Draußen sind es 21 °C und der Himmel ist klar, eine leichte Jacke reicht also völlig aus. Der Zug in die Innenstadt fährt in 15 Minuten, du hast gerade noch Zeit, deinen Kaffee auszutrinken.',
  fr: 'Bonjour ! Il fait 21 °C dehors et le ciel est dégagé, une veste légère suffira donc. Le train pour le centre-ville part dans 15 minutes, juste le temps de finir ton café.',
  es: '¡Buenos días! Fuera hace 21 °C y el cielo está despejado, así que basta con una chaqueta ligera. El tren al centro de la ciudad sale dentro de 15 minutos, el tiempo justo para terminar el café.',
  it: 'Buongiorno! Fuori ci sono 21 °C e il cielo è sereno, quindi basta una giacca leggera. Il treno per il centro città parte tra 15 minuti, giusto il tempo di finire il caffè.',
  pl: 'Dzień dobry! Na zewnątrz jest 21 °C, a niebo jest bezchmurne, więc wystarczy lekka kurtka. Pociąg do centrum miasta odjeżdża za 15 minut, akurat tyle, żeby spokojnie dopić kawę.',
  pt: 'Bom dia! Lá fora estão 21 °C e o céu está limpo, por isso basta um casaco leve. O comboio para o centro da cidade parte daqui a 15 minutos, mesmo a tempo de acabares o café.',
};

const LANGUAGES: { code: ReverbLanguage; label: string }[] = [
  { code: 'en', label: 'English' },
  { code: 'de', label: 'German' },
  { code: 'fr', label: 'French' },
  { code: 'es', label: 'Spanish' },
  { code: 'it', label: 'Italian' },
  { code: 'pl', label: 'Polish' },
  { code: 'pt', label: 'Portuguese' },
];

/** The sentence the user reads into the microphone, by language. It is the
 *  transcript of the recording, so it holds only words, no digits. */
const READ_ALOUD: Record<ReverbLanguage, string> = {
  en: 'The morning train rolls past green fields and quiet villages, while I drink my coffee and watch the sun rise over the hills.',
  de: 'Der Morgenzug fährt an grünen Feldern und stillen Dörfern vorbei, während ich meinen Kaffee trinke und die Sonne über den Hügeln aufgehen sehe.',
  fr: 'Le train du matin longe des champs verts et des villages tranquilles, pendant que je bois mon café et regarde le soleil se lever sur les collines.',
  es: 'El tren de la mañana pasa junto a campos verdes y pueblos tranquilos, mientras tomo mi café y veo salir el sol sobre las colinas.',
  it: 'Il treno del mattino passa accanto a campi verdi e paesi tranquilli, mentre bevo il mio caffè e guardo il sole sorgere sulle colline.',
  pl: 'Poranny pociąg mija zielone pola i ciche wioski, a ja piję kawę i patrzę, jak słońce wschodzi nad wzgórzami.',
  pt: 'O comboio da manhã passa por campos verdes e aldeias tranquilas, enquanto bebo o meu café e vejo o sol nascer sobre as colinas.',
};

/** The rate asked of the microphone, in Hz: the one Reverb's codec runs at. */
const VOICE_SAMPLE_RATE = 24_000;

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

/** The preset voices fetched on this page, by name. */
const voiceCache = new Map<string, Promise<Voice>>();

/** The voice from the cache, fetched on first use. A failed fetch leaves the
 *  cache, so the next call tries again. */
function loadVoice(name: string, language: ReverbLanguage): Promise<Voice> {
  let voice = voiceCache.get(name);
  if (!voice) {
    voice = fetchVoice(name, language);
    voice.catch(() => voiceCache.delete(name));
    voiceCache.set(name, voice);
  }
  return voice;
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

const loadVoiceMaker = (opts: LoadOptions) =>
  createVoiceMaker(models.textToSpeech.REVERB.DEFAULT, opts);

/** Records the user reading a sentence and makes a voice from it. */
function VoiceRecorder({
  maker,
  initialLanguage,
  onVoice,
  onError,
}: {
  maker: VoiceMaker;
  initialLanguage: ReverbLanguage;
  onVoice: (voice: Voice) => void;
  onError: (message: string | undefined) => void;
}) {
  const [language, setLanguage] = useState(initialLanguage);
  const [recording, setRecording] = useState<Recording>();
  const [elapsed, setElapsed] = useState(0);
  const [busy, setBusy] = useState(false);
  // The recording still capturing: the click and the time limit stop it once.
  const live = useRef<Recording>(undefined);

  // Release the microphone if the page goes away mid-recording.
  useEffect(
    () => () => {
      void live.current?.stopRaw();
      live.current = undefined;
    },
    [],
  );

  const stop = async (active: Recording) => {
    if (live.current !== active) return;
    live.current = undefined;
    setRecording(undefined);
    setBusy(true);
    try {
      const { samples, sampleRate } = await active.stopRaw();
      if (samples.length < maker.minSeconds * sampleRate) {
        onError(`Too short. Read the whole sentence, at least ${maker.minSeconds} s.`);
        return;
      }
      // The timer can run a few samples past the longest recording the maker takes.
      const clip = samples.subarray(0, Math.floor(maker.maxSeconds * sampleRate));
      onVoice(
        await maker.fromRecording(clip, { sampleRate, transcript: READ_ALOUD[language], language }),
      );
    } catch (err) {
      onError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    onError(undefined);
    setElapsed(0);
    try {
      const started = await startRecording(VOICE_SAMPLE_RATE);
      live.current = started;
      setRecording(started);
    } catch {
      onError('No microphone. Check the permission the browser asked for.');
    }
  };

  // Count the seconds, and stop at the longest recording the maker takes.
  useEffect(() => {
    if (!recording) return;
    const startedAt = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - startedAt) / 1000), 100);
    const timer = setTimeout(() => void stop(recording), maker.maxSeconds * 1000);
    return () => {
      clearInterval(tick);
      clearTimeout(timer);
    };
  }, [recording]);

  return (
    <div className="demo-row-actions">
      <div className="demo-row">
        <select
          className="demo-select"
          value={language}
          onChange={(e) => {
            const picked = LANGUAGES.find((l) => l.code === e.target.value);
            if (picked) setLanguage(picked.code);
          }}
          disabled={!!recording || busy}
          aria-label="Language of your voice"
        >
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
        <button
          className="demo-button demo-button-quiet"
          onClick={() => (recording ? void stop(recording) : void start())}
          disabled={busy}
        >
          {recording ? '■ Stop' : busy ? 'Making the voice…' : '● Record'}
        </button>
        {recording && (
          <span className="demo-note">
            {elapsed.toFixed(1)} s of {maker.maxSeconds.toFixed(0)} s
          </span>
        )}
      </div>
      <p className="transcript">{READ_ALOUD[language]}</p>
      <p className="demo-note">
        Record yourself reading the sentence above, word for word, to speak in your own voice.
      </p>
    </div>
  );
}

/** The "Your voice" button under the speaker: loads the voice encoder when
 *  clicked, then records. Hovering it tells what the click downloads. */
function OwnVoice(props: {
  initialLanguage: ReverbLanguage;
  onVoice: (voice: Voice) => void;
  onError: (message: string | undefined) => void;
}) {
  const { state, start } = useModel(loadVoiceMaker);

  // The runner owns GPU buffers; release them when the page unmounts.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const maker = state.model;
    return () => maker.dispose();
  }, [state]);

  if (state.status === 'ready') {
    return (
      <div className="demo-own-voice">
        <VoiceRecorder maker={state.model} {...props} />
      </div>
    );
  }

  const loading = state.status === 'loading';
  return (
    <div className="demo-own-voice">
      <div className="demo-row">
        <span className="demo-tip">
          <button
            className="demo-button demo-button-quiet"
            onClick={start}
            disabled={loading}
            aria-describedby="own-voice-tip"
          >
            {loading ? `Downloading… ${Math.round(state.percent * 100)}%` : '● Your voice'}
          </button>
          <span id="own-voice-tip" role="tooltip" className="demo-tip-text">
            Downloads Reverb's voice encoder, about 170 MB, once: after that it is cached in the
            browser. Then record yourself and Reverb speaks in your voice. The recording never
            leaves the browser.
          </span>
        </span>
        {loading && (
          <div className="progress">
            <div
              className="progress-fill"
              style={{ width: `${Math.round(state.percent * 100)}%` }}
            />
          </div>
        )}
      </div>
      {state.status === 'error' && <p className="demo-error">{state.message}</p>}
    </div>
  );
}

function Speaker({ tts }: { tts: TextToSpeech }) {
  const [voiceIndex, setVoiceIndex] = useState(0);
  const [ownVoice, setOwnVoice] = useState<Voice>();
  const [ownPicked, setOwnPicked] = useState(false);
  const [text, setText] = useState(SAMPLES.en);
  const [audio, setAudio] = useState<Float32Array>();
  const [busy, setBusy] = useState(false);
  const [synthesizeMs, setSynthesizeMs] = useState<number>();
  const [error, setError] = useState<string>();
  const playback = useRef<AudioContext>(null);
  const voice = VOICES[voiceIndex]!;

  // Fetch the picked voice ahead of the click.
  useEffect(() => {
    void loadVoice(voice.name, voice.language);
  }, [voice]);

  useEffect(() => () => void playback.current?.close(), []);

  /** Plays the samples, in place of whatever is playing. */
  const play = (samples: Float32Array) => {
    void playback.current?.close();
    const context = new AudioContext({ sampleRate: tts.sampleRate });
    playback.current = context;
    const buffer = context.createBuffer(1, samples.length, tts.sampleRate);
    buffer.getChannelData(0).set(samples);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => {
      if (playback.current !== context) return;
      playback.current = null;
      void context.close();
    };
    source.start();
  };

  /** Picks a preset by its index, or the user's voice by the index after
   *  the last preset. */
  const pickVoice = (index: number) => {
    const preset = VOICES[index];
    if (preset) setVoiceIndex(index);
    setOwnPicked(!preset);
    setText(SAMPLES[(preset ?? ownVoice ?? voice).language]);
  };

  const pickOwnVoice = (made: Voice) => {
    setOwnVoice(made);
    setOwnPicked(true);
    setText(SAMPLES[made.language]);
  };

  const speak = async () => {
    setBusy(true);
    setError(undefined);
    try {
      const own = ownPicked ? ownVoice : undefined;
      const reference = own ?? (await loadVoice(voice.name, voice.language));
      const start = performance.now();
      const spoken = await tts.synthesize(text, reference);
      setSynthesizeMs(performance.now() - start);
      setAudio(spoken);
      play(spoken);
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
        value={ownPicked ? VOICES.length : voiceIndex}
        onChange={(e) => pickVoice(Number(e.target.value))}
        aria-label="Voice"
      >
        {VOICES.map((v, i) => (
          <option key={v.name} value={i}>
            {v.label}
          </option>
        ))}
        {ownVoice && <option value={VOICES.length}>Your voice</option>}
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
          onClick={() => audio && play(audio)}
          disabled={!audio || busy}
        >
          Play again
        </button>
      </div>

      {error && <p className="demo-error">{error}</p>}

      <OwnVoice initialLanguage={voice.language} onVoice={pickOwnVoice} onError={setError} />

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
