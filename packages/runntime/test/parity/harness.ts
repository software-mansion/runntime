/** What every `*.parity.ts` shares in the page: reading the saved
 *  references, breaking a weights file on purpose, and writing the report. */
import { inject } from 'vitest';
import { commands } from 'vitest/browser';
import {
  bufferSource,
  parseSafetensorsHeader,
  type RangeSource,
} from '../../src/core/weights/safetensors.ts';

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** A file in the references directory, by its path under it. */
async function referenceBytes(path: string): Promise<Uint8Array> {
  try {
    return await fetchBytes(`/@fs${inject('parityDir')}/${path}`);
  } catch (err) {
    throw new Error(
      `no reference ${path} in ${inject('parityDir')}: run its tools/parity script, or set RUNNTIME_PARITY_DIR`,
      { cause: err },
    );
  }
}

/** A JSON reference file. `T` is the shape its tools/parity script writes. */
export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await referenceBytes(path))) as T;
}

/** The F32 and I32 tensors of a safetensors reference file, by name. */
export async function readTensors(path: string): Promise<Map<string, Float32Array | Int32Array>> {
  const bytes = await referenceBytes(path);
  const { tensors } = await parseSafetensorsHeader(bufferSource(bytes));
  const out = new Map<string, Float32Array | Int32Array>();
  for (const [name, { dtype, begin, end }] of tensors) {
    // A copy, so the view starts at an aligned offset.
    const data = bytes.slice(begin, end).buffer;
    if (dtype === 'F32') out.set(name, new Float32Array(data));
    else if (dtype === 'I32') out.set(name, new Int32Array(data));
    else throw new Error(`${path}: tensor ${name} is ${dtype}, expected F32 or I32`);
  }
  return out;
}

/** The safetensors file at `url` with the square F32 weight `name`
 *  transposed: a model that loads and runs, and computes the wrong thing. */
export async function transposedWeights(url: string, name: string): Promise<RangeSource> {
  const bytes = await fetchBytes(url);
  const info = (await parseSafetensorsHeader(bufferSource(bytes))).tensors.get(name);
  const [rows, cols] = info?.shape ?? [];
  if (!info || info.dtype !== 'F32' || info.shape.length !== 2 || rows !== cols) {
    throw new Error(`${url}: ${name} is not a square F32 weight`);
  }
  const weight = new Float32Array(bytes.slice(info.begin, info.end).buffer);
  const transposed = Float32Array.from(
    weight,
    (_, i) => weight[(i % rows!) * rows! + Math.floor(i / rows!)]!,
  );
  bytes.set(new Uint8Array(transposed.buffer), info.begin);
  return bufferSource(bytes);
}

/** Writes `rows` as a markdown table to `.parity/results/<family>.md`, the
 *  table a pull request carries. Columns follow the first row's keys. */
export async function writeReport(
  family: string,
  rows: readonly Readonly<Record<string, string>>[],
): Promise<void> {
  const columns = Object.keys(rows[0] ?? {});
  const line = (cells: readonly string[]) => `| ${cells.join(' | ')} |`;
  const table = [
    line(columns),
    line(columns.map(() => '---')),
    ...rows.map((row) => line(columns.map((column) => row[column]!))),
  ];
  await commands.writeFile(`${inject('parityResultsDir')}/${family}.md`, `${table.join('\n')}\n`);
}

declare module 'vitest' {
  export interface ProvidedContext {
    parityDir: string;
    parityResultsDir: string;
  }
}
