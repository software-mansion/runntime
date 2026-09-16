/** CPU-side helpers around the GPU `topk` primitive. Selection MATH happens
 *  on the GPU (core `topk`: strict-`>` top-4 like torch.topk + softmax over
 *  the selected logits); its small packed [T,2k] output is read back and
 *  parsed here, because the expert-compute DAG's structure (which tokens
 *  gather to which expert) is built in JS. `routeTop4` is the CPU reference
 *  mirror of the kernel, kept for tests. */

export interface Selection {
  idx: number[][];
  w: number[][];
}

export function parseTopk(packed: Float32Array, tokens: number, k: number): Selection {
  if (packed.length !== tokens * 2 * k) {
    throw new Error(`parseTopk: packed length ${packed.length} != tokens·2k = ${tokens * 2 * k}`);
  }
  const idx: number[][] = [];
  const w: number[][] = [];
  for (let t = 0; t < tokens; t++) {
    const base = t * 2 * k;
    idx.push(Array.from(packed.subarray(base, base + k)));
    w.push(Array.from(packed.subarray(base + k, base + 2 * k)));
  }
  return { idx, w };
}

export function routeTop4(logits: Float32Array, tokens: number, experts: number): Selection {
  const idx: number[][] = [];
  const w: number[][] = [];
  for (let t = 0; t < tokens; t++) {
    const vals = [-Infinity, -Infinity, -Infinity, -Infinity];
    const ids = [0, 0, 0, 0];
    for (let e = 0; e < experts; e++) {
      const v = logits[t * experts + e]!;
      if (v > vals[3]!) {
        vals[3] = v;
        ids[3] = e;
        for (let s = 3; s > 0 && vals[s]! > vals[s - 1]!; s--) {
          [vals[s], vals[s - 1]] = [vals[s - 1]!, vals[s]!];
          [ids[s], ids[s - 1]] = [ids[s - 1]!, ids[s]!];
        }
      }
    }
    const m = vals[0]!;
    const exps = vals.map((v) => Math.exp(v - m));
    const z = exps.reduce((s, v) => s + v, 0);
    idx.push(ids);
    w.push(exps.map((v) => v / z));
  }
  return { idx, w };
}

export interface ExpertGroup {
  expert: number;
  tokens: number[];
  weights: number[];
}

export interface Grouping {
  groups: ExpertGroup[];
  slotPos: number[][];
}

export function groupByExpert(sel: Selection, experts: number): Grouping {
  const tokens = sel.idx.length;
  const groups: ExpertGroup[] = [];
  const slotPos: number[][] = Array.from({ length: 4 }, () => new Array<number>(tokens).fill(-1));
  let offset = 0;
  for (let e = 0; e < experts; e++) {
    const g: ExpertGroup = { expert: e, tokens: [], weights: [] };
    for (let t = 0; t < tokens; t++) {
      // Top-4 never repeats an expert, so (e, t) matches at most one slot —
      // each token contributes at most one row per expert group.
      for (let s = 0; s < 4; s++) {
        if (sel.idx[t]![s] === e) {
          slotPos[s]![t] = offset + g.tokens.length;
          g.tokens.push(t);
          g.weights.push(sel.w[t]![s]!);
        }
      }
    }
    if (g.tokens.length > 0) {
      groups.push(g);
      offset += g.tokens.length;
    }
  }
  return { groups, slotPos };
}
