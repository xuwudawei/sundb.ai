import { type IndexProgress, indexSchema, type IndexTotalStats, totalSchema } from '@/api/rag';
import { authenticationHeaders, handleErrors, handleResponse, type Page, type PageParams, requestUrl, zodPage } from '@/lib/request';
import { zodJsonDate } from '@/lib/zod';
import { z, type ZodType } from 'zod';

interface DatasourceBase {
  id: number;
  name: string;
  description: string;
  created_at: Date;
  updated_at: Date;
  user_id: string;
  build_kg_index: boolean;
  llm_id: number | null;
}

export type Datasource = DatasourceBase & ({
  data_source_type: 'file'
  config: { file_id: number, file_name: string }[]
} | {
  data_source_type: 'web_sitemap'
  config: { url: string }
} | {
  data_source_type: 'web_single_page'
  config: { urls: string[] }
})

export type DataSourceIndexProgress = {
  vector_index: IndexProgress
  documents: IndexTotalStats
  chunks: IndexTotalStats
  kg_index?: IndexProgress
  relationships?: IndexTotalStats
}

export interface BaseCreateDatasourceParams {
  name: string;
  description: string;
  build_kg_index: boolean;
  llm_id: number | null;
}

export type CreateDatasourceParams = BaseCreateDatasourceParams & ({
  data_source_type: 'file'
  config: { file_id: number, file_name: string }[]
} | {
  data_source_type: 'web_single_page'
  config: { urls: string[] }
} | {
  data_source_type: 'web_sitemap'
  config: { url: string }
})

export interface Upload {
  created_at?: Date;
  updated_at?: Date;
  id: number;
  name: string;
  size: number;
  path: string;
  mime_type: string;
  user_id: string;
}

export type DatasourceVectorIndexError = {
  document_id: number
  document_name: string
  source_uri: string
  error: string | null
}

export type DatasourceKgIndexError = {
  chunk_id: string
  source_uri: string
  error: string | null
}

const baseDatasourceSchema = z.object({
  id: z.number(),
  name: z.string(),
  description: z.string(),
  created_at: zodJsonDate(),
  updated_at: zodJsonDate(),
  user_id: z.string(),
  build_kg_index: z.boolean(),
  llm_id: z.number().nullable(),
});

const datasourceSchema = baseDatasourceSchema
  .and(z.discriminatedUnion('data_source_type', [
    z.object({
      data_source_type: z.literal('file'),
      config: z.array(z.object({ file_id: z.number(), file_name: z.string() })),
    }),
    z.object({
      data_source_type: z.enum(['web_single_page']),
      config: z.object({ urls: z.string().array() }).or(z.object({ url: z.string() })).transform(obj => {
        if ('url' in obj) {
          return { urls: [obj.url] };
        } else {
          return obj;
        }
      }),
    }),
    z.object({
      data_source_type: z.enum(['web_sitemap']),
      config: z.object({ url: z.string() }),
    })],
  )) satisfies ZodType<Datasource, any, any>;

const uploadSchema = z.object({
  id: z.number(),
  name: z.string(),
  size: z.number(),
  path: z.string(),
  mime_type: z.string(),
  user_id: z.string(),
  created_at: zodJsonDate().optional(),
  updated_at: zodJsonDate().optional(),
}) satisfies ZodType<Upload, any, any>;

const datasourceOverviewSchema = z.object({
  vector_index: indexSchema,
  documents: totalSchema,
  chunks: totalSchema,
  kg_index: indexSchema.optional(),
  relationships: totalSchema.optional(),
}) satisfies ZodType<DataSourceIndexProgress>;

const vectorIndexErrorSchema = z.object({
  document_id: z.number(),
  document_name: z.string(),
  source_uri: z.string(),
  error: z.string().nullable(),
}) satisfies ZodType<DatasourceVectorIndexError, any, any>;

const kgIndexErrorSchema = z.object({
  chunk_id: z.string(),
  source_uri: z.string(),
  error: z.string().nullable(),
}) satisfies ZodType<DatasourceKgIndexError, any, any>;

export async function listDataSources ({ page = 1, size = 10 }: PageParams = {}): Promise<Page<Datasource>> {
  return fetch(requestUrl('/api/v1/admin/datasources', { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(datasourceSchema)));
}

export async function getDatasource (id: number): Promise<Datasource> {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(datasourceSchema));
}

export async function deleteDatasource (id: number): Promise<void> {
  await fetch(requestUrl(`/api/v1/admin/datasources/${id}`), {
    method: 'DELETE',
    headers: await authenticationHeaders(),
  }).then(handleErrors);
}

export async function getDatasourceOverview (id: number): Promise<DataSourceIndexProgress> {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}/overview`), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(datasourceOverviewSchema));
}

export async function createDatasource (params: CreateDatasourceParams) {
  return fetch(requestUrl(`/api/v1/admin/datasources`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  }).then(handleResponse(datasourceSchema));
}

export async function uploadFiles (files: File[]) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  return fetch(requestUrl(`/api/v1/admin/uploads`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
    },
    body: formData,
  }).then(handleResponse(uploadSchema.array()));
}

export async function listDatasourceVectorIndexErrors (id: number, { page = 1, size = 10 }: PageParams = {}) {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}/vector-index-errors`, { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(vectorIndexErrorSchema)));
}

export async function listDatasourceKgIndexErrors (id: number, { page = 1, size = 10 }: PageParams = {}) {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}/kg-index-errors`, { page, size }), {
    headers: await authenticationHeaders(),
  }).then(handleResponse(zodPage(kgIndexErrorSchema)));
}

export async function retryDatasourceAllFailedTasks (id: number) {
  return fetch(requestUrl(`/api/v1/admin/datasources/${id}/retry-failed-tasks`), {
    method: 'POST',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
  }).then(handleErrors);
}