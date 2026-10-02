## Development

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

## Versions

The site is published twice from one source. `src/content/docs` is the next
version, live at `/runntime/next/`. `stable/` holds the pages of the released
version, live at `/runntime/`. Both build on every push to main.

- Fix the live docs: edit the page in `stable/`, and the same page in
  `src/content/docs` when the fix applies there too.
- Release: copy the next pages over the stable ones, from `apps/docs`.

```
rm -rf stable && cp -R src/content/docs stable
```

The sidebar in `astro.config.mjs` is shared. An entry whose page is missing
from the build is dropped.
