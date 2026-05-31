import { defineCollection, z } from 'astro:content';

const baseSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  date: z.coerce.date().optional(),
  updatedAt: z.coerce.date().optional(),
  tags: z.preprocess((value) => (Array.isArray(value) ? value : []), z.array(z.string())),
  draft: z.boolean().default(false),
  legacyPath: z.string().optional(),
});

const notes = defineCollection({
  type: 'content',
  schema: baseSchema,
});

const essay = defineCollection({
  type: 'content',
  schema: baseSchema,
});

const projects = defineCollection({
  type: 'content',
  schema: baseSchema.extend({
    repo: z.string().url().optional(),
    featured: z.boolean().default(false),
    status: z.string().optional(),
    period: z.string().optional(),
    role: z.string().optional(),
    techStack: z.array(z.string()).default([]),
    links: z
      .array(
        z.object({
          label: z.string(),
          href: z.string(),
          type: z.string().optional(),
        }),
      )
      .default([]),
    accent: z.string().optional(),
    summary: z.string().optional(),
  }),
});

const gallery = defineCollection({
  type: 'content',
  schema: baseSchema.extend({
    cover: z.string(),
    coverDesc: z.string().optional(),
    images: z.array(z.string()).default([]),
    imageDescs: z.array(z.string()).default([]),
    category: z.enum(['illustration', 'photography', 'design', 'ui-ux', 'other']).default('other'),
  }),
});

export const collections = { notes, essay, projects, gallery };
