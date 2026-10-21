import type { TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { cachedBindGroup } from '../../../src/core/gpu/dispatchCache.ts';

/** A root whose createBindGroup returns a new object and counts the calls. */
function stubRoot() {
  const made: object[] = [];
  const root = {
    createBindGroup: (layout: object, entries: object) => {
      const bg = { layout, entries };
      made.push(bg);
      return bg;
    },
  } as unknown as TgpuRoot;
  return { root, made };
}

const layoutA = {};
const layoutB = {};
const x = {};
const y = {};
const z = {};

type Call = [layout: object, entries: Record<string, object>];

describe('cachedBindGroup', () => {
  // The cache keys a bind group by its layout and which object fills each
  // named slot.
  it.each<{ name: string; first: Call; second: Call; same: boolean }>([
    {
      name: 'the same layout and entries',
      first: [layoutA, { a: x, b: y }],
      second: [layoutA, { a: x, b: y }],
      same: true,
    },
    {
      name: 'the entries in another key order',
      first: [layoutA, { a: x, b: y }],
      second: [layoutA, { b: y, a: x }],
      same: true,
    },
    {
      name: 'another object in one slot',
      first: [layoutA, { a: x, b: y }],
      second: [layoutA, { a: x, b: z }],
      same: false,
    },
    {
      name: 'the same objects in swapped slots',
      first: [layoutA, { a: x, b: y }],
      second: [layoutA, { a: y, b: x }],
      same: false,
    },
    {
      name: 'the same object under another name',
      first: [layoutA, { a: x }],
      second: [layoutA, { b: x }],
      same: false,
    },
    {
      name: 'one slot more',
      first: [layoutA, { a: x }],
      second: [layoutA, { a: x, b: y }],
      same: false,
    },
    {
      name: 'another layout',
      first: [layoutA, { a: x }],
      second: [layoutB, { a: x }],
      same: false,
    },
  ])('$name: same bind group is $same', ({ first, second, same }) => {
    const { root, made } = stubRoot();
    const one = cachedBindGroup(root, ...first);
    const two = cachedBindGroup(root, ...second);
    expect(one === two).toBe(same);
    expect(made).toHaveLength(same ? 1 : 2);
  });

  it('keeps a cache per root', () => {
    const r1 = stubRoot();
    const r2 = stubRoot();
    const one = cachedBindGroup(r1.root, layoutA, { a: x });
    expect(cachedBindGroup(r2.root, layoutA, { a: x })).not.toBe(one);
    expect(cachedBindGroup(r1.root, layoutA, { a: x })).toBe(one);
  });
});
