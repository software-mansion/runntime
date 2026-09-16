/** Live-captioning segment analysis: a lightweight energy-based voice
 *  activity check over one growing utterance buffer. The live loop calls
 *  this every tick to decide when the speaker has gone quiet (finalize the
 *  segment: commit its text and start a new one) and whether the buffer
 *  holds any speech at all (pure silence is never transcribed).
 *
 *  Method: per-hop RMS (32 ms hops), smoothed by a trailing moving average,
 *  compared against a fixed threshold. Deliberately crude next to ML VADs —
 *  no model download, ~1 ms for a 15 s buffer, and misjudgments only shift
 *  Where text is committed, never what the transcriber hears. The threshold
 *  is the tunable; raise it for noisy rooms. */

export interface SegmenterOpts {
  sampleRate: number;
  threshold?: number;
  hopSeconds?: number;
  smoothHops?: number;
  silenceSeconds?: number;
  maxSegmentSeconds?: number;
}

export interface SegmentAnalysis {
  seconds: number;
  hasSpeech: boolean;
  trailingSilenceSeconds: number;
  shouldFinalize: boolean;
}

export function analyzeSegment(audio: Float32Array, opts: SegmenterOpts): SegmentAnalysis {
  const {
    sampleRate,
    threshold = 0.015,
    hopSeconds = 0.032,
    smoothHops = 8,
    silenceSeconds = 0.8,
    maxSegmentSeconds = 15,
  } = opts;
  if (!(sampleRate > 0)) throw new Error(`analyzeSegment: bad sampleRate ${sampleRate}`);
  if (!(threshold > 0)) throw new Error(`analyzeSegment: threshold must be positive`);

  const hop = Math.max(1, Math.round(hopSeconds * sampleRate));
  const seconds = audio.length / sampleRate;
  const hops = Math.ceil(audio.length / hop);
  const rms = new Float64Array(hops);
  for (let h = 0; h < hops; h++) {
    const start = h * hop;
    const end = Math.min(start + hop, audio.length);
    let sum = 0;
    for (let i = start; i < end; i++) sum += audio[i]! * audio[i]!;
    rms[h] = Math.sqrt(sum / (end - start));
  }

  let lastSpeech = -1;
  let windowSum = 0;
  for (let h = 0; h < hops; h++) {
    windowSum += rms[h]!;
    if (h >= smoothHops) windowSum -= rms[h - smoothHops]!;
    const envelope = windowSum / Math.min(h + 1, smoothHops);
    if (envelope > threshold) lastSpeech = h;
  }

  const hasSpeech = lastSpeech >= 0;
  const trailingSilenceSeconds = hasSpeech ? ((hops - 1 - lastSpeech) * hop) / sampleRate : seconds;
  return {
    seconds,
    hasSpeech,
    trailingSilenceSeconds,
    shouldFinalize:
      hasSpeech && (trailingSilenceSeconds >= silenceSeconds || seconds >= maxSegmentSeconds),
  };
}
