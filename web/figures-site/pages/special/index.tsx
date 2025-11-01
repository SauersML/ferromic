import fs from 'node:fs/promises';
import path from 'node:path';
import type { GetStaticProps } from 'next';
import Head from 'next/head';

import {
  SpecialGallery,
  type SpecialFigureGroup,
} from '../../components/SpecialGallery';

interface SpecialManifest {
  generatedAt: string | null;
  groups: SpecialFigureGroup[];
}

interface SpecialPageProps {
  manifest: SpecialManifest;
  manifestError: string | null;
  supplementaryHref: string | null;
}

const MANIFEST_PATH = path.join(process.cwd(), 'data', 'special-figures.json');
const SUPPLEMENTARY_RELATIVE_PATH = path.join(
  process.cwd(),
  'public',
  'downloads',
  'supplementary_tables.xlsx',
);

export const getStaticProps: GetStaticProps<SpecialPageProps> = async () => {
  let supplementaryHref: string | null = null;
  try {
    await fs.access(SUPPLEMENTARY_RELATIVE_PATH);
    supplementaryHref = '/downloads/supplementary_tables.xlsx';
  } catch (error) {
    supplementaryHref = null;
  }

  try {
    const raw = await fs.readFile(MANIFEST_PATH, 'utf-8');
    const manifest = JSON.parse(raw) as SpecialManifest;
    return {
      props: {
        manifest,
        manifestError: null,
        supplementaryHref,
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return {
      props: {
        manifest: { generatedAt: null, groups: [] },
        manifestError: `Unable to read special manifest: ${message}`,
        supplementaryHref,
      },
    };
  }
};

export default function SpecialPage({ manifest, manifestError, supplementaryHref }: SpecialPageProps) {
  return (
    <>
      <Head>
        <title>Special Figure Collection • Ferromic</title>
        <meta
          name="description"
          content="Dedicated gallery of requested Ferromic PDF figures."
        />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if (typeof window !== 'undefined' && window['pdfjs-dist/build/pdf']) {
                window['pdfjs-dist/build/pdf'].GlobalWorkerOptions.workerSrc = 
                  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
              }
            `,
          }}
        />
      </Head>
      {manifestError ? (
        <div style={{ maxWidth: '720px', margin: '4rem auto', padding: '0 1rem' }}>
          <h1>Special Figure Collection</h1>
          <p>
            Unable to load the special figure manifest. Ensure that
            <code style={{ margin: '0 0.35rem' }}>data/special-figures.json</code>
            exists and contains valid JSON.
          </p>
          <pre
            style={{
              background: 'rgba(220, 38, 38, 0.1)',
              borderRadius: '8px',
              padding: '1rem',
              overflowX: 'auto',
              color: '#991b1b',
            }}
          >
            {manifestError}
          </pre>
        </div>
      ) : (
        <SpecialGallery
          groups={manifest.groups}
          generatedAt={manifest.generatedAt}
          supplementaryHref={supplementaryHref}
        />
      )}
    </>
  );
}
