/** The kernel A/B project, run by `bench/cli.ts`. Kept out of
 *  `vitest.config.ts` so `pnpm test` never runs it. */
import { fileURLToPath } from 'node:url';
import { playwright } from '@vitest/browser-playwright';
import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';

const at = (path: string) => fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  root: at('..'),
  plugins: [typegpu()],
  optimizeDeps: { include: ['typegpu'] },
  resolve: {
    alias: {
      '@ab/a': at('../.bench/base/src'),
      '@ab/b': at('../src'),
    },
  },
  test: {
    name: 'bench',
    include: ['bench/kernels/*.ab.ts'],
    provide: { abResultsDir: at('../.bench/results') },
    // One file at a time: two pages sharing the GPU would time each other.
    fileParallelism: false,
    testTimeout: 120_000,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          channel: 'chromium',
          args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'],
        },
      }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
