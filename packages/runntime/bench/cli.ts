/** `pnpm bench:ab kernels [-b <ref>] [-k matmul,reduce]`: copies the base
 *  ref's `src` to `.bench/base`, runs the kernel A/B in headless Chromium and
 *  prints the report as a table, with a markdown copy for the pull request in
 *  `.bench/results/report.md`. */
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs, styleText } from 'node:util';
import { formatTable, formatTerminal, type CaseResult, type Verdict } from './stats.ts';

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
  console.error('usage: pnpm bench:ab kernels [-b <ref>] [-k matmul,reduce]');
  process.exit(1);
}

const known = readdirSync(`${pkg}bench/kernels`)
  .filter((f) => f.endsWith('.ab.ts'))
  .map((f) => f.slice(0, -'.ab.ts'.length));
const unknown = values.kernels?.split(',').filter((k) => !known.includes(k)) ?? [];
if (unknown.length > 0) {
  console.error(`no bench/kernels/<name>.ab.ts for: ${unknown.join(', ')}`);
  console.error(`-k takes: ${known.join(', ')}`);
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
const title = `A: \`${values.base}\` (${sha}), B: \`HEAD\` (${head})${dirty}`;
writeFileSync(`${results}/report.md`, `${title}\n\n${formatTable(rows)}\n`);

const colors: Record<Verdict, 'green' | 'red' | 'dim'> = {
  faster: 'green',
  slower: 'red',
  same: 'dim',
};
const paint = (v: Verdict, text: string) => styleText(colors[v], text);
console.log(`\n${title.replaceAll('`', '')}\n\n${formatTerminal(rows, paint)}\n`);
console.log(`Markdown for the pull request: ${results}/report.md`);
process.exit(run.status ?? 1);
