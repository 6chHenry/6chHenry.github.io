import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';

// Production site is served at root. Use SITE_BASE=/next only for hybrid preview deploys.
const base = process.env.SITE_BASE ?? '/';

export default defineConfig({
  site: 'https://6chHenry.github.io',
  base,
  trailingSlash: 'always',
  integrations: [
    mdx(),
    sitemap({
      filter: (page) => !page.includes('/tags/'),
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeSlug, rehypeKatex],
    shikiConfig: {
      theme: 'github-dark-dimmed',
      wrap: true,
    },
  },
  vite: {
    css: {
      devSourcemap: true,
    },
  },
});
