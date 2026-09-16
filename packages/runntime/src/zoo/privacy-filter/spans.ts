import type { LabelInfo } from './config.ts';
import type { Tokenizer } from './tokenizer.ts';

export interface TokenSpan {
  label: number; // span-class index (1..8)
  tokenStart: number;
  tokenEnd: number; // exclusive
}

export interface DetectedSpan {
  label: string;
  start: number; // UTF-16 index into text
  end: number;
  text: string;
  placeholder: string;
}

export function labelsToSpans(labels: readonly number[], info: LabelInfo): TokenSpan[] {
  const spans: TokenSpan[] = [];
  let currentLabel: number | null = null;
  let startIdx: number | null = null;

  const flush = (endExclusive: number) => {
    if (currentLabel !== null && startIdx !== null) {
      spans.push({ label: currentLabel, tokenStart: startIdx, tokenEnd: endExclusive });
    }
    currentLabel = null;
    startIdx = null;
  };

  labels.forEach((labelId, idx) => {
    const spanLabel = info.tokenToSpanLabel.get(labelId);
    const tag = info.tokenBoundaryTags.get(labelId) ?? null;
    if (spanLabel === undefined) return;
    if (spanLabel === info.backgroundSpanLabel) {
      flush(idx);
      return;
    }
    if (tag === 'S') {
      flush(idx);
      spans.push({ label: spanLabel, tokenStart: idx, tokenEnd: idx + 1 });
    } else if (tag === 'B') {
      flush(idx);
      currentLabel = spanLabel;
      startIdx = idx;
    } else if (tag === 'I') {
      if (currentLabel === null || currentLabel !== spanLabel) {
        flush(idx);
        currentLabel = spanLabel;
        startIdx = idx;
      }
    } else if (tag === 'E') {
      if (currentLabel === null || currentLabel !== spanLabel || startIdx === null) {
        flush(idx);
        spans.push({ label: spanLabel, tokenStart: idx, tokenEnd: idx + 1 });
      } else {
        spans.push({ label: currentLabel, tokenStart: startIdx, tokenEnd: idx + 1 });
        currentLabel = null;
        startIdx = null;
      }
    }
  });
  flush(labels.length);
  return spans;
}

export function labelPlaceholder(label: string): string {
  const normalized = label
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return `<${normalized || 'REDACTED'}>`;
}

interface CharSpan {
  label: number;
  start: number;
  end: number;
}

function trimWhitespace(spans: CharSpan[], text: string): CharSpan[] {
  const out: CharSpan[] = [];
  for (let { label, start, end } of spans) {
    if (!(0 <= start && start < end && end <= text.length)) continue;
    while (start < end && /\s/u.test(text[start]!)) start++;
    while (end > start && /\s/u.test(text[end - 1]!)) end--;
    if (end > start) out.push({ label, start, end });
  }
  return out;
}

function discardOverlappingByLabel(spans: CharSpan[]): CharSpan[] {
  const byLabel = new Map<number, CharSpan[]>();
  for (const s of spans) {
    if (!byLabel.has(s.label)) byLabel.set(s.label, []);
    byLabel.get(s.label)!.push(s);
  }
  const kept: CharSpan[] = [];
  for (const labelSpans of byLabel.values()) {
    labelSpans.sort((a, b) => a.start - b.start || b.end - b.start - (a.end - a.start));
    const keptHere: CharSpan[] = [];
    for (const s of labelSpans) {
      const overlaps = keptHere.some((k) => !(s.end <= k.start || s.start >= k.end));
      if (!overlaps) keptHere.push(s);
    }
    kept.push(...keptHere);
  }
  kept.sort((a, b) => a.start - b.start || a.end - b.end || a.label - b.label);
  return kept;
}

function selectNonOverlapping(spans: DetectedSpan[]): DetectedSpan[] {
  const ordered = [...spans].sort(
    (a, b) =>
      a.start - b.start || b.end - b.start - (a.end - a.start) || a.label.localeCompare(b.label),
  );
  const kept: DetectedSpan[] = [];
  let cursor = 0;
  for (const s of ordered) {
    if (s.start < cursor || s.end <= s.start) continue;
    kept.push(s);
    cursor = s.end;
  }
  return kept;
}

export function detectedSpansFromLabels(
  labels: readonly number[],
  tokenIds: readonly number[],
  tokenizer: Tokenizer,
  info: LabelInfo,
): DetectedSpan[] {
  const tokenSpans = labelsToSpans(labels, info);
  const { text, charStarts, charEnds } = tokenizer.tokenCharOffsets(tokenIds);

  let charSpans: CharSpan[] = [];
  for (const { label, tokenStart, tokenEnd } of tokenSpans) {
    if (!(0 <= tokenStart && tokenStart < tokenEnd && tokenEnd <= charStarts.length)) continue;
    const start = charStarts[tokenStart]!;
    const end = charEnds[tokenEnd - 1]!;
    if (end > start) charSpans.push({ label, start, end });
  }
  charSpans = trimWhitespace(charSpans, text);
  charSpans = discardOverlappingByLabel(charSpans);

  const detected: DetectedSpan[] = [];
  for (const { label, start, end } of charSpans) {
    if (!(0 <= start && start < end && end <= text.length)) continue;
    const name = info.spanClassNames[label] ?? `label_${label}`;
    detected.push({
      label: name,
      start,
      end,
      text: text.slice(start, end),
      placeholder: labelPlaceholder(name),
    });
  }
  return selectNonOverlapping(detected);
}
