/** The parity project: zoo models against saved outputs of their PyTorch
 *  references, on the real GPU. Kept out of `vitest.config.ts` so `pnpm test`
 *  never runs it. */
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { playwright } from '@vitest/browser-playwright';
import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';

const at = (path: string) => fileURLToPath(new URL(path, import.meta.url));

/** Where the references are: one folder per model family, written by
 *  `tools/parity`. */
const REFERENCES = process.env.RUNNTIME_PARITY_DIR ?? join(homedir(), '.cache', 'runntime-parity');

export default defineConfig({
  plugins: [typegpu()],
  optimizeDeps: { include: ['typegpu'] },
  // The page reads the references over the dev server.
  server: { fs: { allow: [at('../..'), REFERENCES] } },
  test: {
    name: 'parity',
    include: ['test/**/*.parity.ts'],
    provide: { parityDir: REFERENCES, parityResultsDir: at('.parity/results') },
    // One file at a time: each loads whole models onto the one GPU.
    fileParallelism: false,
    testTimeout: 300_000,
    hookTimeout: 300_000,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: { channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
