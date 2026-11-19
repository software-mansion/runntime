import { afterEach, describe, expect, it, vi } from 'vitest';
import { decodeAudio } from '../../../../src/zoo/tasks/audio/audioInput.ts';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('decodeAudio', () => {
  it.each([0, -16_000, Number.NaN, Infinity])('refuses sampleRate %s', async (rate) => {
    const made = vi.fn();
    vi.stubGlobal('OfflineAudioContext', made);
    await expect(decodeAudio(new ArrayBuffer(8), rate)).rejects.toMatchObject({
      code: 'INVALID_ARGUMENT',
      message: expect.stringMatching(/must be a positive number/),
    });
    expect(made).not.toHaveBeenCalled();
  });

  it('refuses a sample rate the browser does not take', async () => {
    // The browser throws NotSupportedError for a rate outside its range.
    const refusal = new DOMException('rate out of range', 'NotSupportedError');
    vi.stubGlobal(
      'OfflineAudioContext',
      vi.fn(function () {
        throw refusal;
      }),
    );
    const err = await decodeAudio(new ArrayBuffer(8), 1_000).catch((e: unknown) => e);
    expect(err).toMatchObject({
      code: 'INVALID_ARGUMENT',
      message: expect.stringMatching(/sampleRate 1000 is not one this browser takes/),
    });
    expect((err as Error).cause).toBe(refusal);
  });

  it('decodes at the given rate', async () => {
    const samples = Float32Array.of(0.5, -0.5);
    const made = vi.fn(function (this: object) {
      Object.assign(this, {
        decodeAudioData: async () => ({ numberOfChannels: 1, getChannelData: () => samples }),
      });
    });
    vi.stubGlobal('OfflineAudioContext', made);
    expect(await decodeAudio(new ArrayBuffer(8), 24_000)).toBe(samples);
    expect(made).toHaveBeenCalledWith(1, 1, 24_000);
  });
});
