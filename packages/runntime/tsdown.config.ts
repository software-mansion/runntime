import { defineConfig } from 'tsdown';
import typegpu from 'unplugin-typegpu/rollup';

export default defineConfig({
  // Keyed so the output layout matches the published subpaths.
  entry: {
    'zoo/index': 'src/zoo/index.ts',
    'zoo/transformers/index': 'src/zoo/transformers/index.ts',
  },
  format: 'esm',
  dts: true,
  // Kernels carry a 'use gpu' directive; the plugin embeds their WGSL metadata
  // into the published JS so consumers don't need a build plugin of their own.
  plugins: [typegpu()],
  external: ['typegpu', '@huggingface/transformers', 'js-tiktoken'],
});
