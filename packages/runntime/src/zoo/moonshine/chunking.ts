/** Long-audio splitting for the transcriber. The decoder's position budget
 *  (194 tokens ≈ 30 s at ~6.5 tokens/s) bounds a single utterance, so longer
 *  clips are cut into independent segments that are transcribed one after
 *  another. Each cut lands on the quietest 100 ms inside a search window at
 *  the end of the segment's allowed span — cutting on silence instead of at a
 *  fixed offset keeps words intact. Pure CPU over subarray views: no copies,
 *  every returned chunk aliases the input buffer. */

export interface ChunkOpts {
  maxSeconds: number;
  searchSeconds: number;
  sampleRate: number;
}

const QUIET_WINDOW_SECONDS = 0.1;

const TAIL_MERGE_SECONDS = 1;

function quietestCut(audio: Float32Array, from: number, to: number, window: number): number {
  let energy = 0;
  for (let i = from; i < from + window; i++) energy += audio[i]! * audio[i]!;
  let best = energy;
  let bestStart = from;
  for (let start = from + 1; start + window <= to; start++) {
    energy += audio[start + window - 1]! * audio[start + window - 1]!;
    energy -= audio[start - 1]! * audio[start - 1]!;
    if (energy < best) {
      best = energy;
      bestStart = start;
    }
  }
  return bestStart + (window >> 1);
}

export function splitIntoChunks(audio: Float32Array, opts: ChunkOpts): Float32Array[] {
  const max = Math.floor(opts.maxSeconds * opts.sampleRate);
  const search = Math.min(Math.floor(opts.searchSeconds * opts.sampleRate), max - 1);
  const window = Math.min(Math.floor(QUIET_WINDOW_SECONDS * opts.sampleRate), search);
  if (max <= 0 || window <= 0) throw new Error('splitIntoChunks: degenerate chunk config');
  const tailMerge = Math.floor(TAIL_MERGE_SECONDS * opts.sampleRate);

  const chunks: Float32Array[] = [];
  let cursor = 0;
  while (audio.length - cursor > max + tailMerge) {
    const cut = quietestCut(audio, cursor + max - search, cursor + max, window);
    chunks.push(audio.subarray(cursor, cut));
    cursor = cut;
  }
  chunks.push(audio.subarray(cursor));
  return chunks;
}
