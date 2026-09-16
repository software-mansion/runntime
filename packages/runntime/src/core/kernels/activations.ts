/** Fused-epilogue activations: one body per activation, shared by every kernel
 *  that folds one into its epilogue and by the standalone unary kernels. */

import tgpu, { d, std } from 'typegpu';

export const ACT_CODE = { none: 0, tanh: 1, gelu: 2, silu: 3, relu: 4 } as const;
export type ActName = keyof typeof ACT_CODE;
export type SlotAct = Exclude<ActName, 'none'>;

export type Activation = {
  (x: number): number;
  (x: d.v2f): d.v2f;
  (x: d.v4f): d.v4f;
};

export const identityAct = ((x: d.v4f) => {
  'use gpu';
  return x * 1;
}) as Activation;

export const tanhAct = ((x: d.v4f) => {
  'use gpu';
  return std.tanh(x);
}) as Activation;

const INV_SQRT2 = Math.SQRT1_2; // 1/√2
const ERF_P = 0.3275911;
const ERF_A1 = 0.254829592;
const ERF_A2 = -0.284496736;
const ERF_A3 = 1.421413741;
const ERF_A4 = -1.453152027;
const ERF_A5 = 1.061405429;

export const gelu = ((x: d.v4f) => {
  'use gpu';
  const z = x * INV_SQRT2;
  const az = std.abs(z);
  const t = 1 / (1 + az * ERF_P);
  // a1·t + … + a5·t⁵ in Horner form.
  const poly = t * (ERF_A1 + t * (ERF_A2 + t * (ERF_A3 + t * (ERF_A4 + t * ERF_A5))));
  const erf = std.sign(z) * (1 - poly * std.exp(-az * az));
  return 0.5 * x * (1 + erf);
}) as Activation;

export const siluAct = ((x: d.v4f) => {
  'use gpu';
  return x / (1 + std.exp(-x));
}) as Activation;

export const reluAct = ((x: d.v4f) => {
  'use gpu';
  return std.max(x, d.vec4f(0));
}) as Activation;

export const ACTIVATIONS: readonly Activation[] = [identityAct, tanhAct, gelu, siluAct, reluAct];

const ACT_NAMES = ['none', 'tanh', 'gelu', 'silu', 'relu'] as const;
for (const [i, name] of ACT_NAMES.entries()) {
  if (ACT_CODE[name] !== i || ACTIVATIONS[i] === undefined) {
    throw new Error(
      `activations: ACT_CODE.${name} is ${ACT_CODE[name]} but ACTIVATIONS has its body at ${i} — the two tables drifted`,
    );
  }
}
if (ACTIVATIONS.length !== Object.keys(ACT_CODE).length) {
  throw new Error('activations: ACT_CODE and ACTIVATIONS have different lengths');
}

export function activationFor(code: number | undefined): Activation {
  const body = ACTIVATIONS[code ?? 0];
  if (!body) {
    throw new Error(
      `unknown activation code ${code} — valid codes are 0..${ACTIVATIONS.length - 1}`,
    );
  }
  return body;
}

export const actSlot = tgpu.slot<Activation>(identityAct);
