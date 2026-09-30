/** Expressive Code plugins, merged with the options in astro.config.mjs.
 *  Twoslash type-checks every ```ts twoslash block against the built
 *  `runntime` package and shows the type on hover. A type error fails the build. */
import ecTwoSlash from 'expressive-code-twoslash';
import ts from 'typescript';

export default {
  plugins: [
    ecTwoSlash({
      // The popup shows only the type. The page prose explains it.
      includeJsDoc: false,
      twoslashOptions: {
        compilerOptions: {
          strict: true,
          moduleResolution: ts.ModuleResolutionKind.Bundler,
        },
        extraFiles: {
          'global.d.ts': '/// <reference lib="dom" />\n/// <reference types="@webgpu/types" />',
        },
      },
    }),
  ],
};
