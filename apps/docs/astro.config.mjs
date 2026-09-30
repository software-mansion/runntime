// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import { createStarlightTypeDocPlugin } from 'starlight-typedoc';
import typegpuPlugin from 'unplugin-typegpu/vite';
import fs from 'node:fs';
import { ExpressiveCodeTheme } from '@astrojs/starlight/expressive-code';

const monoDark = ExpressiveCodeTheme.fromJSONString(
  fs.readFileSync(new URL('./src/styles/code-theme-dark.jsonc', import.meta.url), 'utf-8'),
);
const monoLight = ExpressiveCodeTheme.fromJSONString(
  fs.readFileSync(new URL('./src/styles/code-theme-light.jsonc', import.meta.url), 'utf-8'),
);

// One TypeDoc run per published entry point, so each gets its own
// sidebar group and output folder under src/content/docs/api.
const [zooTypeDoc, zooTypeDocGroup] = createStarlightTypeDocPlugin();
const [transformersTypeDoc, transformersTypeDocGroup] = createStarlightTypeDocPlugin();

/** TypeDoc options shared by both runs. */
const typeDocOptions = {
  tsconfig: '../../packages/runntime/tsconfig.json',
  typeDoc: {
    // Members tagged @internal in their doc comment stay out of the reference.
    excludeInternal: true,
    readme: 'none',
    entryFileName: 'index',
  },
};

const base = `${(process.env.BASE_PATH ?? '/runntime').replace(/\/$/, '')}/`;

/** Astro prefixes component hrefs with `base`, but not links written in
 *  Markdown prose. Without this they all 404 under /runntime/. */
function rehypeBaseLinks() {
  const prefix = (url) =>
    url.startsWith('/') && !url.startsWith('//') && !url.startsWith(base)
      ? base + url.slice(1)
      : url;

  return (tree) => {
    const walk = (node) => {
      if (node.tagName === 'a' && node.properties?.href) {
        node.properties.href = prefix(node.properties.href);
      }
      if (node.tagName === 'img' && node.properties?.src) {
        node.properties.src = prefix(node.properties.src);
      }
      for (const child of node.children ?? []) walk(child);
    };
    walk(tree);
  };
}

// https://astro.build/config
export default defineConfig({
  site: process.env.SITE_URL ?? 'https://docs.swmansion.com',
  base,
  trailingSlash: 'always',
  redirects: {
    '/': `${base}zoo/getting-started/`,
    '/zoo/': `${base}zoo/getting-started/`,
  },
  integrations: [
    starlight({
      title: 'ruNNtime',
      components: {
        ThemeSelect: './src/components/ThemeSelect.astro',
        SiteTitle: './src/components/SiteTitle.astro',
        TableOfContents: './src/components/TableOfContents.astro',
        SocialIcons: './src/components/SocialIcons.astro',
      },
      customCss: [
        './src/styles/fonts.css',
        './src/styles/tokens.css',
        './src/styles/theme.css',
        './src/styles/components.css',
        './src/styles/demos.css',
      ],
      expressiveCode: {
        themes: [monoDark, monoLight],
        // When true this overwrites styleOverrides.frames wholesale. We set our own.
        useStarlightUiThemeColors: false,
        styleOverrides: {
          borderColor: 'var(--border-strong)',
          borderWidth: '1px',
          codeBackground: 'var(--surface-raised)',
          codeLineHeight: '1.45',
          codePaddingBlock: '0.875rem',
          codePaddingInline: '1rem',
          frames: {
            frameBoxShadowCssValue: 'none',
            shadowColor: 'transparent',
            editorBackground: 'var(--surface-raised)',
            editorActiveTabBackground: 'var(--surface-raised)',
            editorActiveTabForeground: 'var(--text-primary)',
            editorActiveTabBorderColor: 'transparent',
            editorActiveTabIndicatorTopColor: 'transparent',
            editorActiveTabIndicatorBottomColor: 'var(--text-primary)',
            editorActiveTabIndicatorHeight: '3px',
            editorTabBorderRadius: '0',
            editorTabBarBackground: 'var(--surface-base)',
            editorTabBarBorderColor: 'transparent',
            editorTabBarBorderBottomColor: 'var(--border-strong)',
            terminalBackground: 'var(--surface-raised)',
            terminalTitlebarBackground: 'var(--surface-base)',
            terminalTitlebarForeground: 'var(--text-muted)',
            terminalTitlebarBorderBottomColor: 'var(--border-strong)',
            inlineButtonBackground: 'transparent',
            inlineButtonForeground: 'var(--text-faint)',
            inlineButtonBorder: 'var(--border-strong)',
            tooltipSuccessBackground: 'var(--text-primary)',
            tooltipSuccessForeground: 'var(--surface-base)',
          },
          textMarkers: {
            markBackground: 'var(--code-line-highlight)',
            markBorderColor: 'var(--border-strong)',
          },
        },
      },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/software-mansion/runntime' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/ZGqqY55qkP' },
      ],
      plugins: [
        starlightSidebarTopics(
          [
            {
              id: 'zoo',
              label: 'Zoo',
              link: '/zoo/getting-started/',
              // Any Starlight icon works here: components.css hides it and
              // paints the squirrel, since Starlight ships no animals.
              icon: 'rocket',
              items: [
                { label: 'Getting started', slug: 'zoo/getting-started' },
                { label: 'Text embedding', slug: 'zoo/text-embedding' },
                { label: 'Speech to text', slug: 'zoo/speech-to-text' },
                { label: 'Privacy filter', slug: 'zoo/privacy-filter' },
                { label: 'Object detection', slug: 'zoo/object-detection' },
                { label: 'Instance segmentation', slug: 'zoo/instance-segmentation' },
                { label: 'Pose & keypoints', slug: 'zoo/pose-and-keypoints' },
                { label: 'Depth estimation', slug: 'zoo/depth-estimation' },
                { label: 'Image classification', slug: 'zoo/image-classification' },
                {
                  label: 'Migrating from transformers.js',
                  slug: 'zoo/migrating-from-transformers-js',
                },
                { label: 'Benchmarks', slug: 'zoo/benchmarks' },
                {
                  label: 'API reference',
                  collapsed: true,
                  items: [zooTypeDocGroup, transformersTypeDocGroup],
                },
              ],
            },
          ],
          {
            // The splash landing page belongs to no topic.
            exclude: ['/'],
            // The generated index pages sit outside the sidebar groups.
            topics: { zoo: ['/api/**'] },
          },
        ),
        // Run after the topics: they fill the placeholder groups under
        // Zoo > API reference.
        zooTypeDoc({
          ...typeDocOptions,
          entryPoints: ['../../packages/runntime/src/zoo/index.ts'],
          output: 'api/zoo',
          sidebar: { label: 'runntime/zoo', collapsed: true },
        }),
        transformersTypeDoc({
          ...typeDocOptions,
          entryPoints: ['../../packages/runntime/src/zoo/transformers/index.ts'],
          output: 'api/transformers',
          sidebar: { label: 'runntime/zoo/transformers', collapsed: true },
        }),
      ],
    }),
    react(),
  ],
  markdown: {
    rehypePlugins: [rehypeBaseLinks],
  },
  vite: {
    // The live demos import runntime/zoo from source, and its kernels are written
    // as TypeGPU 'use gpu' functions the plugin transpiles to WGSL.
    plugins: [typegpuPlugin()],
    server: { fs: { allow: ['../..'] } },
  },
});
