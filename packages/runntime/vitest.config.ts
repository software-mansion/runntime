import { playwright } from '@vitest/browser-playwright';
import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          environment: 'node',
          include: ['test/**/*.test.ts'],
          exclude: ['test/**/*.gpu.test.ts'],
        },
      },
      {
        // Kernels carry a 'use gpu' directive, compiled to WGSL by the plugin.
        plugins: [typegpu()],
        // Pre-bundled up front: a mid-run re-bundle reloads the page and drops the suite.
        optimizeDeps: { include: ['typegpu'] },
        test: {
          name: 'gpu',
          include: ['test/**/*.gpu.test.ts'],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({
              launchOptions: { channel: 'chromium', args: ['--enable-unsafe-webgpu'] },
            }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
