import type { TaskResult } from '@/core/db/task';
import { rag } from '@/core/interface';
import { handleErrors } from '@/lib/fetch';
import { XMLParser } from 'fast-xml-parser';
import { z } from 'zod';

interface Entry {
  loc: string;
}

interface UrlEntry extends Entry {
}

interface SitemapIndex {
  sitemapindex: { sitemap: Entry[] };
}

interface UrlSet {
  urlset: {
    url: UrlEntry[]
  };
}

type Sitemap = SitemapIndex | UrlSet

export class SitemapXmlTaskProcessor extends rag.ImportSourceTaskProcessor<{}> {
  static identifier = 'rag.import-source-task.sitemap';
  static displayName = 'Sitemap.xml processor';
  static optionsSchema = z.object({});

  private readonly parser = new XMLParser({
    isArray: tagName => ['sitemap', 'url'].includes(tagName),
  });

  support (taskType: string): boolean {
    return taskType === 'sitemap';
  }

  async process (task: { url: string }): Promise<TaskResult> {
    const urls = await this.fetchSitemap(task.url);

    return {
      enqueue: urls.map(url => ({
        type: 'html',
        url,
      })),
    };
  }

  private async fetchSitemap (url: string): Promise<string[]> {
    const response = await fetch(url).then(handleErrors);

    const sitemap: Sitemap = this.parser.parse(Buffer.from(await response.arrayBuffer()));

    if ('sitemapindex' in sitemap) {
      return Promise.all(sitemap.sitemapindex.sitemap.map(sitemap => this.fetchSitemap(sitemap.loc))).then(urls => urls.flatMap(url => url));
    } else {
      return sitemap.urlset.url?.map(url => url.loc) ?? [];
    }
  }
}
