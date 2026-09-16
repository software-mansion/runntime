// Copies the repo README into packages/runntime so npm ships the same page as
// GitHub. The package copy is generated, like dist: gitignored, written by
// prepack. Pass --check to fail instead of writing, for CI.
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const source = new URL('README.md', root);
const target = new URL('packages/runntime/README.md', root);

const contents = await readFile(source, 'utf8');

if (process.argv.includes('--check')) {
  const current = await readFile(target, 'utf8').catch(() => null);
  if (current !== contents) {
    console.error(`${fileURLToPath(target)} is out of date. Run \`pnpm sync:readme\`.`);
    process.exit(1);
  }
} else {
  await writeFile(target, contents);
}
