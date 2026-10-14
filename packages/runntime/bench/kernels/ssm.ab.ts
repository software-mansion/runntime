/** The three SSM kernels at DepthART's 14×14 grid. Each has one route; the
 *  cases cover its stages (c, rank). */
import { abSuite, type AbCase } from '../ab.ts';
import { chw, mat, randMat } from './inputs.ts';

const HW = 14;
const P = HW * HW;

function stage(c: number, rank: number): AbCase[] {
  const dirs = 4;
  const state = 8;
  return [
    {
      name: `ssmScanProject, DepthART-B c ${c} rank ${rank}`,
      build: (s) => {
        const x = s.up(s.rand(c * P), chw(c, HW, HW));
        const xw = randMat(s, 1, dirs * (rank + 2 * state) * c);
        const dw = randMat(s, 1, dirs * c * rank);
        return () => s.lib.ssmScanProject(x, xw, dw, { rank });
      },
    },
    {
      name: `ssmSelectiveScan, DepthART-B c ${c} rank ${rank}`,
      build: (s) => {
        const x = s.up(s.rand(c * P), chw(c, HW, HW));
        const proj = randMat(s, dirs * (c + 2 * state + rank), P);
        // A is negative in the model: exp(Δ·A) decays.
        const a = s.up(
          s.rand(dirs * c * state).map((v) => -Math.abs(v) - 0.1),
          mat(1, dirs * c * state),
        );
        const dSkip = randMat(s, 1, dirs * c);
        const deltaBias = randMat(s, 1, dirs * c);
        return () => s.lib.ssmSelectiveScan(x, proj, a, dSkip, deltaBias);
      },
    },
    {
      name: `ssmScanMerge, DepthART-B c ${c}`,
      build: (s) => {
        const dir = randMat(s, dirs * c, P);
        return () => s.lib.ssmScanMerge(dir, { c, h: HW, w: HW });
      },
    },
  ];
}

abSuite('ssm', [...stage(12, 3), ...stage(96, 12), ...stage(288, 24)]);
