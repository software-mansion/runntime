## Working on the docs

- `pnpm dev:docs` / `pnpm build:docs`. The docs and their live demos import
  `runntime/zoo` from the library's `dist`, so build the library first or run
  `pnpm --filter docs... run build`. The dev server does not watch the
  library: after editing `packages/runntime/src`, rebuild it.
- Code samples are ` ```ts twoslash `: the docs build type-checks them
  against the built library and fails on an error. Setup the reader does not
  need (`declare const canvas: HTMLCanvasElement;`) goes above
  `// ---cut---`.
- Never edit the API reference under `/api/`. `starlight-typedoc` generates
  it from the exports and their doc comments.
- Benchmark numbers come from measured runs. Never edit or derive one by
  hand.

## Publishing the site

The site is published from the `docs` branch, not from `main`. Work lands on
`main`; merging `main` into `docs` deploys it to Pages, usually right after an
npm release. A fix for the live site goes in a pull request against `docs`,
then `docs` is merged back into `main`. Both merges use a merge commit, never
squash, or the two branches drift apart. The npm publish is a manual
workflow.

## Astro dev server

When starting the dev server, use background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
