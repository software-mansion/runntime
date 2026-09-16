export const V2_SPAN_CLASS_NAMES = [
  'O',
  'account_number',
  'private_address',
  'private_date',
  'private_email',
  'private_person',
  'private_phone',
  'private_url',
  'secret',
] as const;

const BOUNDARY_PREFIXES = ['B', 'I', 'E', 'S'] as const;
export type BoundaryTag = (typeof BOUNDARY_PREFIXES)[number];

export const V2_NER_CLASS_NAMES: readonly string[] = [
  'O',
  ...V2_SPAN_CLASS_NAMES.filter((n) => n !== 'O').flatMap((base) =>
    BOUNDARY_PREFIXES.map((p) => `${p}-${base}`),
  ),
];

export interface LabelInfo {
  spanClassNames: string[];
  tokenToSpanLabel: Map<number, number>;
  tokenBoundaryTags: Map<number, BoundaryTag | null>;
  backgroundTokenLabel: number;
  backgroundSpanLabel: number;
  numClasses: number;
}

export function buildLabelInfo(classNames: readonly string[]): LabelInfo {
  const spanClassNames: string[] = ['O'];
  const spanLabelLookup = new Map<string, number>([['O', 0]]);
  const tokenToSpanLabel = new Map<number, number>();
  const tokenBoundaryTags = new Map<number, BoundaryTag | null>();
  let backgroundIdx: number | null = null;

  classNames.forEach((name, idx) => {
    if (name === 'O') {
      backgroundIdx = idx;
      tokenToSpanLabel.set(idx, 0);
      tokenBoundaryTags.set(idx, null);
      return;
    }
    const dash = name.indexOf('-');
    const boundary = name.slice(0, dash) as BoundaryTag;
    const base = name.slice(dash + 1);
    let spanIdx = spanLabelLookup.get(base);
    if (spanIdx === undefined) {
      spanIdx = spanClassNames.length;
      spanClassNames.push(base);
      spanLabelLookup.set(base, spanIdx);
    }
    tokenToSpanLabel.set(idx, spanIdx);
    tokenBoundaryTags.set(idx, boundary);
  });

  if (backgroundIdx === null) throw new Error("Class names must include 'O'");
  return {
    spanClassNames,
    tokenToSpanLabel,
    tokenBoundaryTags,
    backgroundTokenLabel: backgroundIdx,
    backgroundSpanLabel: 0,
    numClasses: classNames.length,
  };
}

export interface ViterbiBiases {
  backgroundStay: number;
  backgroundToStart: number;
  insideToContinue: number;
  insideToEnd: number;
  endToBackground: number;
  endToStart: number;
}

export const ZERO_BIASES: ViterbiBiases = {
  backgroundStay: 0,
  backgroundToStart: 0,
  insideToContinue: 0,
  insideToEnd: 0,
  endToBackground: 0,
  endToStart: 0,
};
