import { describe, expect, it } from 'vitest';
import { stopPrefixLength } from '../../../../src/zoo/tasks/nlp/textGeneration.ts';

describe('stopPrefixLength', () => {
  it.each([
    ['no stop strings', 'hello', undefined, 0],
    ['no overlap', 'hello', ['\nUser:'], 0],
    ['one char held', 'hello\n', ['\nUser:'], 1],
    ['most of the stop held', 'hello\nUse', ['\nUser:'], 4],
    ['the longest of several', 'a</t', ['</tool>', '<', '</'], 3],
    ['shorter than the text', 'ab', ['abc'], 2],
    ['an empty text', '', ['abc'], 0],
  ])('%s', (_, text, stop, want) => {
    expect(stopPrefixLength(text, stop)).toBe(want);
  });
});
