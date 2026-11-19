import type { TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { ReverbModel } from '../../../src/zoo/reverb/model.ts';
import { createReverbSpeaker, type Voice } from '../../../src/zoo/reverb/speaker.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';

// The default 128 MiB binding limit. Reverb's widest row is its 6152 fine
// logits in f32, so a pass runs at most 5454 rows. Node has no WebGPU, and
// open() checks the lengths before it touches the device.
const root = {
  device: { features: new Set(), limits: { maxStorageBufferBindingSize: 2 ** 27 } },
} as unknown as TgpuRoot;

const speaker = createReverbSpeaker(root, new ReverbModel(), new HiggsDecoder(), {
  encode: () => [1, 2, 3],
});

/** A voice of `frames` frames whose transcript is one letter per frame. */
const voice = (frames: number): Voice => ({
  codes: new Int32Array(8 * frames),
  frames,
  transcript: 'a'.repeat(frames),
  language: 'en',
});

describe('createReverbSpeaker', () => {
  it('refuses a text past the rows a pass can bind', () => {
    // One frame per letter.
    expect(() => speaker.open('a'.repeat(5455), voice(10))).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: expect.stringMatching(/5455 frames/),
      }),
    );
  });

  it('refuses a voice and prompt past the rows a pass can bind', () => {
    // 3 prompt tokens and 5452 reference frames.
    expect(() => speaker.open('a'.repeat(60), voice(5452))).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: expect.stringMatching(/5455 rows/),
      }),
    );
  });

  it("estimates the text's length from the voice's speech frames", () => {
    // 20 letters over 20 frames, 10 of them speech: half a frame per letter, so
    // 10910 letters need 5455 frames, not the 10910 the whole clip gives.
    const paced = { ...voice(20), speechFrames: 10 };
    expect(() => speaker.open('a'.repeat(10910), paced)).toThrow(
      expect.objectContaining({ message: expect.stringMatching(/5455 frames/) }),
    );
  });

  it('caps the speech frames at the whole clip', () => {
    const paced = { ...voice(10), speechFrames: 1000 };
    expect(() => speaker.open('a'.repeat(5455), paced)).toThrow(
      expect.objectContaining({ message: expect.stringMatching(/5455 frames/) }),
    );
  });

  it('refuses a text of fewer than two frames', () => {
    // 400 letters in one frame: 'a' gets 50·(0.0025 / 50)^⅓ = 1.8 frames.
    const dense = { ...voice(1), transcript: 'a'.repeat(400) };
    expect(() => speaker.open('a', dense)).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: expect.stringMatching(/1 frame/),
      }),
    );
  });
});
