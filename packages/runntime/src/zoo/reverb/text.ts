/** Reverb's prompt, its target-length rule, the reference's speaking time,
 *  and the trim of the decoded audio. */

import { FRAME_RATE } from './config.ts';

/** The prompt before the reference frames. The reference transcript and the
 *  text are read as one, joined by a space. */
export function reverbPrompt(language: string, referenceText: string, text: string): string {
  return (
    `<|denoise|><|lang_start|>${language}<|lang_end|><|instruct_start|>None<|instruct_end|>` +
    `<|text_start|>${referenceText} ${text}<|text_end|>`
  );
}

/** A character's spoken weight by its Unicode category: separators 0.2,
 *  marks 0, punctuation and symbols 0.5, numbers 3.5, the rest 1. */
function charWeight(ch: string): number {
  if (/\p{Z}/u.test(ch)) return 0.2;
  if (/\p{M}/u.test(ch)) return 0;
  if (/[\p{P}\p{S}]/u.test(ch)) return 0.5;
  if (/\p{N}/u.test(ch)) return 3.5;
  return 1;
}

const weightOf = (s: string) => [...s].reduce((sum, ch) => sum + charWeight(ch), 0);

/** Targets estimated under this many frames get relatively more time. */
const SHORT_FRAMES = 50;
/** Cut from the decoded clip's end: its last frames can hold a burst of noise. */
const END_CUT_SECONDS = 0.2;
/** The trim measures loudness in windows of 10 ms. */
const WINDOWS_PER_SECOND = 100;
/** A window under this level is silence. */
const SILENCE_DBFS = -50;
/** More silent windows than this, 200 ms, split two sounds. */
const GAP_WINDOWS = 20;
/** A sound of at most this many windows, 120 ms, alone at an edge is a codec burst. */
const BURST_WINDOWS = 12;
/** Silence kept on each side of the trimmed clip. */
const KEEP_SECONDS = 0.1;

/** Target frames for `text`: its spoken weight at the reference's rate of
 *  weight per frame. */
export function estimateFrames(
  text: string,
  referenceText: string,
  referenceFrames: number,
): number {
  let frames = weightOf(text) / (weightOf(referenceText) / referenceFrames);
  if (frames < SHORT_FRAMES) frames = SHORT_FRAMES * (frames / SHORT_FRAMES) ** (1 / 3);
  return Math.max(1, Math.floor(frames));
}

/** Cuts the clip's last END_CUT_SECONDS, then its leading and trailing
 *  silence, keeping KEEP_SECONDS of it on each side. A codec burst alone at
 *  either edge goes too. */
export function trimSilence(audio: Float32Array, sampleRate: number): Float32Array {
  const clip = audio.subarray(
    0,
    Math.max(0, audio.length - Math.floor(END_CUT_SECONDS * sampleRate)),
  );
  const window = Math.floor(sampleRate / WINDOWS_PER_SECOND);
  const count = Math.floor(clip.length / window);
  const loud: number[] = [];
  for (let w = 0; w < count; w++) {
    let sq = 0;
    for (let i = w * window; i < (w + 1) * window; i++) sq += clip[i]! * clip[i]!;
    if (20 * Math.log10(Math.sqrt(sq / window) + 1e-10) >= SILENCE_DBFS) loud.push(w);
  }
  if (loud.length === 0) return clip;
  let sounds: number[][] = [[loud[0]!]];
  for (let i = 1; i < loud.length; i++) {
    if (loud[i]! - loud[i - 1]! > GAP_WINDOWS) sounds.push([]);
    sounds.at(-1)!.push(loud[i]!);
  }
  if (sounds.length > 1 && sounds[0]!.length <= BURST_WINDOWS) sounds = sounds.slice(1);
  if (sounds.length > 1 && sounds.at(-1)!.length <= BURST_WINDOWS) sounds = sounds.slice(0, -1);
  const margin = Math.round(KEEP_SECONDS * WINDOWS_PER_SECOND);
  const start = Math.max(sounds[0]![0]! - margin, 0) * window;
  const end = (sounds.at(-1)!.at(-1)! + 1 + margin) * window;
  return clip.slice(start, Math.min(end, clip.length));
}

/** Python's round(): halves go to the even neighbour, so frame counts match
 *  the reverb-tts package exactly. */
const roundHalfEven = (x: number) => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

/** Seconds of a reference recording that set the speaker's pace: the
 *  recording without its silent edges (keeping `lead` and `trail` seconds)
 *  and with every pause inside the speech counted as at most `maxPause`
 *  seconds. Levels are measured after scaling the clip to an RMS of 0.1, so a
 *  quiet recording isn't all silence. A port of the reverb-tts package's
 *  audio.speech_length (Reverb-TTS PR #6). */
export function speechSeconds(
  audio: Float32Array,
  sampleRate: number,
  maxPause = 0.5,
  lead = 0.1,
  trail = 0.2,
): number {
  const window = Math.floor(sampleRate / WINDOWS_PER_SECOND);
  const count = Math.floor(audio.length / window);
  if (count === 0) return audio.length / sampleRate;
  let total = 0;
  for (let i = 0; i < count * window; i++) total += audio[i]! * audio[i]!;
  const gain = 0.1 / Math.max(Math.sqrt(total / (count * window)), 1e-8);
  const loud: number[] = [];
  for (let w = 0; w < count; w++) {
    let sq = 0;
    for (let i = w * window; i < (w + 1) * window; i++) sq += (audio[i]! * gain) ** 2;
    if (20 * Math.log10(Math.sqrt(sq / window) + 1e-10) >= SILENCE_DBFS) loud.push(w);
  }
  if (loud.length === 0) return audio.length / sampleRate;
  const cap = roundHalfEven(maxPause * WINDOWS_PER_SECOND);
  let kept = loud.length;
  for (let k = 1; k < loud.length; k++) kept += Math.min(loud[k]! - loud[k - 1]! - 1, cap);
  const first = loud[0]!;
  const last = loud[loud.length - 1]!;
  const edges =
    Math.min(first, roundHalfEven(lead * WINDOWS_PER_SECOND)) +
    Math.min(count - 1 - last, roundHalfEven(trail * WINDOWS_PER_SECOND));
  return (kept + edges) / WINDOWS_PER_SECOND;
}

/** speechSeconds as codec frames, rounded the way the package rounds them. */
export const speechFrames = (audio: Float32Array, sampleRate: number) =>
  roundHalfEven(speechSeconds(audio, sampleRate) * FRAME_RATE);
