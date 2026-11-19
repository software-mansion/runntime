import { describe, expect, it } from 'vitest';
import {
  reverbPrompt,
  estimateFrames,
  speechFrames,
  speechSeconds,
  trimSilence,
} from '../../../src/zoo/reverb/text.ts';

describe('reverbPrompt', () => {
  it('wraps the language and both texts, joined by a space, in the markers', () => {
    expect(reverbPrompt('pl', 'Dzień dobry.', 'Cześć!')).toBe(
      '<|denoise|><|lang_start|>pl<|lang_end|><|instruct_start|>None<|instruct_end|>' +
        '<|text_start|>Dzień dobry. Cześć!<|text_end|>',
    );
  });
});

describe('estimateFrames', () => {
  it("scales the text's spoken weight by the reference's frames per weight", () => {
    // 'abcd' weighs 4 over 400 frames: 100 frames per letter.
    expect(estimateFrames('ab', 'abcd', 400)).toBe(200);
  });

  it('weighs separators 0.2, punctuation 0.5, numbers 3.5 and marks 0', () => {
    // 'a b' 2.2, 'a.' 1.5, '7' 3.5, 'e' plus a combining acute 1.
    expect(estimateFrames('a b', 'a', 100)).toBe(220);
    expect(estimateFrames('a.', 'a', 100)).toBe(150);
    expect(estimateFrames('7', 'a', 100)).toBe(350);
    expect(estimateFrames('é', 'a', 100)).toBe(100);
  });

  it('gives a target under 50 frames relatively more time', () => {
    // 25 frames becomes 50·(25 / 50)^⅓ = 39.7.
    expect(estimateFrames('a', 'abcd', 100)).toBe(39);
  });
});

describe('trimSilence', () => {
  // 1 kHz, so a 10 ms window is 10 samples.
  const RATE = 1000;
  /** 3 s of silence with a level of 0.5 over each [start, end) sample range. */
  const clip = (...sounds: [number, number][]) => {
    const audio = new Float32Array(3000);
    for (const [start, end] of sounds) audio.fill(0.5, start, end);
    return audio;
  };

  it('cuts the last 200 ms, then the silence around the sound but 100 ms', () => {
    const out = trimSilence(clip([1000, 2000]), RATE);
    expect(out.length).toBe(1200);
    expect(out[99]).toBe(0);
    expect(out[100]).toBe(0.5);
    expect(out[1099]).toBe(0.5);
    expect(out[1100]).toBe(0);
  });

  it('drops a burst of at most 120 ms alone at an edge', () => {
    const burst = trimSilence(clip([100, 150], [1000, 2000]), RATE);
    expect(burst).toEqual(trimSilence(clip([1000, 2000]), RATE));
  });

  it('keeps a silent clip, less its last 200 ms', () => {
    expect(trimSilence(new Float32Array(3000), RATE).length).toBe(2800);
  });
});

describe('speechSeconds', () => {
  // 1 kHz, so a 10 ms window is 10 samples.
  const RATE = 1000;
  /** Sound (a level of `level`) and silence in turn, each `seconds` long. */
  const clip = (level: number, ...seconds: number[]) => {
    const parts = seconds.map((s, i) =>
      new Float32Array(Math.round(s * RATE)).fill(i % 2 ? level : 0),
    );
    const audio = new Float32Array(parts.reduce((n, part) => n + part.length, 0));
    let at = 0;
    for (const part of parts) {
      audio.set(part, at);
      at += part.length;
    }
    return audio;
  };

  it('drops the silent edges, keeping 100 ms before the speech and 200 ms after it', () => {
    expect(speechSeconds(clip(0.5, 1, 2, 1), RATE)).toBeCloseTo(2.3);
  });

  it('counts a pause inside the speech as at most 500 ms', () => {
    expect(speechSeconds(clip(0.5, 0, 1, 2, 1), RATE)).toBeCloseTo(2.5);
  });

  it('keeps a shorter pause whole', () => {
    expect(speechSeconds(clip(0.5, 0, 1, 0.3, 1), RATE)).toBeCloseTo(2.3);
  });

  it('hears a quiet recording, measured after scaling it to an RMS of 0.1', () => {
    // 0.001 is -60 dBFS, under the -50 dBFS of silence before the scaling.
    expect(speechSeconds(clip(0.001, 1, 2, 1), RATE)).toBeCloseTo(2.3);
  });

  it('keeps a silent clip whole', () => {
    expect(speechSeconds(new Float32Array(3000), RATE)).toBe(3);
  });
});

describe('speechFrames', () => {
  it('rounds half a frame to the even one, as Python does', () => {
    // 2.02 s is 50.5 frames at 25 frames per second.
    expect(speechFrames(new Float32Array(2020).fill(0.5), 1000)).toBe(50);
  });
});
