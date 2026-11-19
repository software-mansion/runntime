import { describe, expect, it } from 'vitest';
import { spokenInput } from '../../../../src/zoo/tasks/audio/textToSpeech.ts';
import type { Voice } from '../../../../src/zoo/index.ts';

const voice: Voice = {
  codes: new Int32Array(8),
  frames: 1,
  transcript: 'Mam 3 psy.',
  language: 'pl',
};

describe('spokenInput', () => {
  it('keeps the text and the voice without a verbalizer', () => {
    const input = spokenInput('Mam 2 koty.', voice, undefined);
    expect(input.text).toBe('Mam 2 koty.');
    expect(input.voice).toBe(voice);
  });

  it("verbalizes the text and the voice's transcript in the voice's language", () => {
    const calls: [string, string][] = [];
    const input = spokenInput('Mam 2 koty.', voice, (text, language) => {
      calls.push([text, language]);
      return text.replace('2', 'dwa').replace('3', 'trzy');
    });
    expect(calls).toEqual([
      ['Mam 2 koty.', 'pl'],
      ['Mam 3 psy.', 'pl'],
    ]);
    expect(input.text).toBe('Mam dwa koty.');
    expect(input.voice).toEqual({ ...voice, transcript: 'Mam trzy psy.' });
    expect(voice.transcript).toBe('Mam 3 psy.');
  });

  it.each([
    ['text', (text: string) => (text.startsWith('Mam 2') ? ' ' : text)],
    ['voice transcript', (text: string) => (text.startsWith('Mam 3') ? '' : text)],
  ])('refuses an empty verbalized %s', (what, verbalize) => {
    expect(() => spokenInput('Mam 2 koty.', voice, verbalize)).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: expect.stringContaining(`for the ${what}`),
      }),
    );
  });
});
