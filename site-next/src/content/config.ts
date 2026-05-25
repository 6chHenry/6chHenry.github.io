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
  }),
});

const gallery = defineCollection({
  type: 'content',
  schema: baseSchema.extend({
    cover: z.string(),
    images: z.array(z.string()).default([]),
    category: z.enum(['illustration', 'photography', 'design', 'ui-ux', 'other']).default('other'),
  }),
});

export const collections = { notes, essay, projects, gallery };
