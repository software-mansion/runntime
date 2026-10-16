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
      build: (version) => {
        const x = version.upload(version.rand(c * P), chw(c, HW, HW));
        const xw = randMat(version, 1, dirs * (rank + 2 * state) * c);
        const dw = randMat(version, 1, dirs * c * rank);
        return () => version.lib.ssmScanProject(x, xw, dw, { rank });
      },
    },
    {
      name: `ssmSelectiveScan, DepthART-B c ${c} rank ${rank}`,
      build: (version) => {
        const x = version.upload(version.rand(c * P), chw(c, HW, HW));
        const proj = randMat(version, dirs * (c + 2 * state + rank), P);
        // A is negative in the model: exp(Δ·A) decays.
        const a = version.upload(
          version.rand(dirs * c * state).map((v) => -Math.abs(v) - 0.1),
          mat(1, dirs * c * state),
        );
        const dSkip = randMat(version, 1, dirs * c);
        const deltaBias = randMat(version, 1, dirs * c);
        return () => version.lib.ssmSelectiveScan(x, proj, a, dSkip, deltaBias);
      },
    },
    {
      name: `ssmScanMerge, DepthART-B c ${c}`,
      build: (version) => {
        const dir = randMat(version, dirs * c, P);
        return () => version.lib.ssmScanMerge(dir, { c, h: HW, w: HW });
      },
    },
  ];
}

abSuite('ssm', [...stage(12, 3), ...stage(96, 12), ...stage(288, 24)]);
