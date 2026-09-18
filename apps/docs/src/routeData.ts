import { defineRouteMiddleware } from '@astrojs/starlight/route-data';

/** On the next build, every page carries a banner that points at the release. */
export const onRequest = defineRouteMiddleware((context) => {
  const base = import.meta.env.BASE_URL;
  if (!base.endsWith('/next/')) return;
  const root = base.slice(0, -'next/'.length);
  const release = process.env.DOCS_VERSION;
  const label = release ? `the latest version (${release})` : 'the latest version';
  context.locals.starlightRoute.entry.data.banner ??= {
    content:
      `This is unreleased documentation for the ruNNtime next version. ` +
      `For up-to-date documentation, see <a href="${root}">${label}</a>.`,
  };
});
