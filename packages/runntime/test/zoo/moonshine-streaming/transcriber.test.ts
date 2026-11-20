import { describe, expect, it } from 'vitest';
import { padToFrames } from '../../../src/zoo/moonshine-streaming/transcriber.ts';

const FRAME = 80;
const samples = (length: number) => Float32Array.from({ length }, (_, i) => i + 1);

describe('padToFrames', () => {
  it('returns a clip of whole frames as it is', () => {
    const audio = samples(3 * FRAME);
    expect(padToFrames(audio, FRAME)).toBe(audio);
  });

  it('keeps the whole frames and zeroes a partial last frame', () => {
    const audio = samples(2 * FRAME + 7);
    const padded = padToFrames(audio, FRAME);
    expect(padded).toHaveLength(3 * FRAME);
    expect(padded.subarray(0, 2 * FRAME)).toEqual(audio.subarray(0, 2 * FRAME));
    expect(padded.subarray(2 * FRAME)).toEqual(new Float32Array(FRAME));
  });

  it('gives one frame of zeros for a clip shorter than a frame', () => {
    expect(padToFrames(samples(FRAME - 1), FRAME)).toEqual(new Float32Array(FRAME));
  });

  it('rejects empty audio', () => {
    expect(() => padToFrames(new Float32Array(0), FRAME)).toThrow('transcribe: empty audio');
  });
});
