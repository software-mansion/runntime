import { describe, expect, it } from 'vitest';
import { MASK } from '../../../src/zoo/reverb/config.ts';
import {
  mainPassUpdate,
  polishFirst,
  polishSecond,
  seededRandom,
} from '../../../src/zoo/reverb/sampler.ts';

const CODEBOOKS = 8;

/** Coarse stats over [2, frames]: code 100 + slot everywhere, confidence
 *  from `confidence`, the held code's log-probability from `held`, by
 *  default the same. */
function stats(
  frames: number,
  confidence: (slot: number) => number,
  held: (slot: number) => number = confidence,
) {
  return {
    best: Int32Array.from({ length: 2 * frames }, (_, i) => 100 + i),
    confidence: Float32Array.from({ length: 2 * frames }, (_, i) => confidence(i)),
    heldConfidence: Float32Array.from({ length: 2 * frames }, (_, i) => held(i)),
  };
}

const masked = (frames: number) => new Int32Array(CODEBOOKS * frames).fill(MASK);
const slotsWhere = (tokens: Int32Array, frames: number, keep: (code: number) => boolean) =>
  [...tokens.subarray(0, 2 * frames)].flatMap((c, i) => (keep(c) ? [i] : []));

describe('seededRandom', () => {
  it('gives the same uniforms in [0, 1) for the same seed', () => {
    const draw = (seed: number) => Array.from({ length: 100 }, seededRandom(seed));
    expect(draw(7)).toEqual(draw(7));
    expect(draw(7)).not.toEqual(draw(8));
    expect(draw(7).every((u) => u >= 0 && u < 1)).toBe(true);
  });
});

describe('mainPassUpdate', () => {
  it('commits the best scored masked slots, codebook 0 before 1', () => {
    const frames = 50;
    const tokens = masked(frames);
    // Equal noise: the score is confidence minus 5 per codebook index.
    mainPassUpdate(
      tokens,
      frames,
      stats(frames, (i) => i / 100),
      0,
      8,
      () => 0.5,
    );
    // The reference commits 2 slots in pass 0 of 8 for 50 frames.
    expect(slotsWhere(tokens, frames, (c) => c !== MASK)).toEqual([48, 49]);
    expect(tokens[49]).toBe(149);
    expect(tokens.subarray(2 * frames).every((c) => c === MASK)).toBe(true);
  });

  it('re-masks the earlier commits whose code is least likely, a quarter as many as it makes', () => {
    const frames = 20;
    const tokens = masked(frames);
    // Slots 0–9 committed earlier, slot i's code at log-probability i and
    // its best code at −i, so the best codes alone would point at slot 9.
    for (let i = 0; i < 10; i++) tokens[i] = 7;
    mainPassUpdate(
      tokens,
      frames,
      stats(
        frames,
        (i) => -i,
        (i) => i,
      ),
      4,
      8,
      seededRandom(1),
    );
    // The reference commits 3 slots in pass 4 of 8 for 20 frames, and
    // round(0.75) is 1.
    const k = 3;
    expect(slotsWhere(tokens, frames, (c) => c === 7)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(slotsWhere(tokens, frames, (c) => c !== MASK && c !== 7)).toHaveLength(k);
  });

  // Masked slots, and so the slots this pass commits, are set lowest here;
  // only earlier commits are re-masked.
  it('re-masks only commits made before the pass', () => {
    const frames = 20;
    const tokens = masked(frames);
    for (let i = 0; i < 10; i++) tokens[i] = 7;
    mainPassUpdate(
      tokens,
      frames,
      stats(
        frames,
        () => 0,
        (i) => (i < 10 ? i : -100),
      ),
      4,
      8,
      seededRandom(1),
    );
    expect(slotsWhere(tokens, frames, (c) => c === 7)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(slotsWhere(tokens, frames, (c) => c !== MASK && c !== 7)).toHaveLength(3);
  });

  it('rounds half a re-mask to even', () => {
    const frames = 50;
    const tokens = masked(frames);
    tokens[0] = 7;
    mainPassUpdate(
      tokens,
      frames,
      stats(frames, () => 0),
      0,
      8,
      seededRandom(1),
    );
    // The reference commits 2 slots in pass 0 of 8 for 50 frames, and
    // round(0.5) is 0.
    expect(tokens[0]).toBe(7);
  });

  it('rounds 1.5 re-masks to 2', () => {
    const frames = 77;
    const tokens = masked(frames);
    // Slots 0–3 committed earlier, slot i's code at log-probability i.
    for (let i = 0; i < 4; i++) tokens[i] = 7;
    mainPassUpdate(
      tokens,
      frames,
      stats(frames, (i) => i),
      3,
      8,
      seededRandom(1),
    );
    // The reference commits 6 slots in pass 3 of 8 for 77 frames.
    expect(slotsWhere(tokens, frames, (c) => c === 7)).toEqual([2, 3]);
  });

  it("commits per pass as the reference's f32 schedule does", () => {
    // The reference's commits for 77 frames over 8 passes.
    const reference = [3, 3, 4, 6, 9, 14, 28];
    const frames = 77;
    const made = reference.map((_, step) => {
      const tokens = masked(frames);
      mainPassUpdate(
        tokens,
        frames,
        stats(frames, () => 0),
        step,
        8,
        () => 0.5,
      );
      return slotsWhere(tokens, frames, (c) => c !== MASK).length;
    });
    expect(made).toEqual(reference);
  });

  it('commits every masked coarse slot in the last pass and re-masks none', () => {
    const frames = 6;
    const tokens = masked(frames);
    tokens[0] = 7;
    mainPassUpdate(
      tokens,
      frames,
      stats(frames, () => 0),
      7,
      8,
      seededRandom(1),
    );
    expect(tokens[0]).toBe(7);
    expect(slotsWhere(tokens, frames, (c) => c === MASK)).toEqual([]);
  });

  it('draws one uniform per coarse slot, masked or not', () => {
    const frames = 5;
    const tokens = masked(frames);
    tokens[3] = 7;
    let draws = 0;
    mainPassUpdate(
      tokens,
      frames,
      stats(frames, () => 0),
      2,
      8,
      () => (draws++, 0.5),
    );
    expect(draws).toBe(2 * frames);
  });
});

describe('polish passes', () => {
  // Fine stats over codebooks 2–7: code 10·book + frame.
  const frames = 3;
  const fine = {
    best: Int32Array.from(
      { length: 6 * frames },
      (_, i) => 10 * Math.floor(i / frames) + (i % frames),
    ),
  };

  it('fill codebooks 2–3 first, then 4–7', () => {
    const tokens = masked(frames);
    polishFirst(tokens, frames, fine);
    expect([...tokens.subarray(2 * frames, 4 * frames)]).toEqual([0, 1, 2, 10, 11, 12]);
    expect(tokens.subarray(4 * frames).every((c) => c === MASK)).toBe(true);
    polishSecond(tokens, frames, fine);
    expect([...tokens.subarray(4 * frames)]).toEqual([
      20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52,
    ]);
    expect(tokens.subarray(0, 2 * frames).every((c) => c === MASK)).toBe(true);
  });
});
