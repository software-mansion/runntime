/** `pnpm bench:ab kernels [-b <ref>] [-k matmul,softmax]`: copies the base
 *  ref's `src` to `.bench/base`, runs the kernel A/B in headless Chromium and
 *  prints the report, also saved to `.bench/results/report.md`. */
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { formatTable, type CaseResult } from './stats.ts';

const pkg = fileURLToPath(new URL('..', import.meta.url));
const base = `${pkg}.bench/base`;
const results = `${pkg}.bench/results`;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    base: { type: 'string', short: 'b', default: 'main' },
    kernels: { type: 'string', short: 'k' },
  },
});
if (positionals[0] !== 'kernels') {
  console.error('usage: pnpm bench:ab kernels [-b <ref>] [-k matmul,softmax]');
  process.exit(1);
}

const git = (...args: string[]) => execFileSync('git', args, { cwd: pkg, encoding: 'utf8' }).trim();
const sha = git('rev-parse', '--short', values.base);
const head = git('rev-parse', '--short', 'HEAD');
const dirty = git('status', '--porcelain', 'src') ? ' + uncommitted changes' : '';

rmSync(`${pkg}.bench`, { recursive: true, force: true });
mkdirSync(base, { recursive: true });
mkdirSync(results, { recursive: true });
execFileSync('sh', ['-c', `git archive ${sha} src | tar -x -C "${base}"`], { cwd: pkg });

const files = values.kernels?.split(',').map((k) => `bench/kernels/${k}.ab.ts`) ?? [];
const run = spawnSync(
  'pnpm',
  ['exec', 'vitest', 'run', '--config', 'bench/vitest.config.ts', ...files],
  { cwd: pkg, stdio: 'inherit' },
);

const rows = readdirSync(results)
  .filter((f) => f.endsWith('.json'))
  .flatMap((f) => JSON.parse(readFileSync(`${results}/${f}`, 'utf8')) as CaseResult[]);
const report = [
  `A: \`${values.base}\` (${sha}), B: \`HEAD\` (${head})${dirty}`,
  '',
  formatTable(rows),
  '',
].join('\n');
writeFileSync(`${results}/report.md`, report);
console.log(`\n${report}`);
process.exit(run.status ?? 1);
