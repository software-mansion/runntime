// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import typegpuPlugin from 'unplugin-typegpu/vite';
import fs from 'node:fs';
import { ExpressiveCodeTheme } from '@astrojs/starlight/expressive-code';

const monoDark = ExpressiveCodeTheme.fromJSONString(
  fs.readFileSync(new URL('./src/styles/code-theme-dark.jsonc', import.meta.url), 'utf-8'),
);
const monoLight = ExpressiveCodeTheme.fromJSONString(
  fs.readFileSync(new URL('./src/styles/code-theme-light.jsonc', import.meta.url), 'utf-8'),
);

/** GitHub Pages serves this repo's site under /runntime/, so every asset and
 *  link needs that prefix. Overridable for a custom domain, where it is '/'.
 *  Normalized with a trailing slash so the redirects below can concatenate. */
const base = `${(process.env.BASE_PATH ?? '/runntime').replace(/\/$/, '')}/`;

/** Markdown and MDX prose links are written root-absolute (`/zoo/...`), the
 *  way they read in the source repo. Astro prefixes component hrefs with
 *  `base` but leaves content links alone, so under a project Pages path they
 *  would all 404. This rewrites them at build time, which keeps the content
 *  files identical to upstream. */
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
  site: process.env.SITE_URL ?? 'https://software-mansion.github.io',
  base,
  trailingSlash: 'always',
  redirects: {
    // Astro prefixes hrefs with `base` but not redirect destinations, so
    // these spell it out. Without it the landing page 404s on Pages.
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
        { icon: 'github', label: 'GitHub', href: 'https://github.com/software-mansion' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/ZGqqY55qkP' },
      ],
      plugins: [
        starlightSidebarTopics(
          [
            {
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
                { label: 'Migrating from transformers.js', slug: 'zoo/migrating-from-transformers-js' },
                { label: 'Benchmarks', slug: 'zoo/benchmarks' },
              ],
            },
          ],
          {
            // The splash landing page belongs to no topic.
            exclude: ['/'],
          },
        ),
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
