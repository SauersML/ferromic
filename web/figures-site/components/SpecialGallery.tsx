import { useEffect, useRef, useState } from 'react';
import styles from '../styles/SpecialGallery.module.css';

export interface SpecialFigureItem {
  title: string;
  filename: string;
  description?: string;
}

export interface SpecialFigureGroup {
  title: string;
  slug: string;
  figures: SpecialFigureItem[];
}

export interface SpecialGalleryProps {
  groups: SpecialFigureGroup[];
  generatedAt: string | null;
  supplementaryHref?: string | null;
}

const BASE_PATH = '/ferromic/figures';

function PDFPreview({ assetPath, title }: { assetPath: string; title: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    // Use PDF.js to get actual page dimensions
    const loadPDF = async () => {
      try {
        // @ts-ignore - pdfjs-dist loaded via CDN
        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        if (!pdfjsLib) return;

        const loadingTask = pdfjsLib.getDocument(assetPath);
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 1 });
        
        setDimensions({
          width: viewport.width,
          height: viewport.height,
        });
      } catch (e) {
        console.error('Failed to load PDF dimensions:', e);
      }
    };

    loadPDF();
  }, [assetPath]);

  const containerStyle = dimensions
    ? {
        aspectRatio: `${dimensions.width} / ${dimensions.height}`,
        height: 'auto',
        width: '100%',
      }
    : {};

  return (
    <div ref={containerRef} className={styles.preview} style={containerStyle}>
      <object
        data={assetPath}
        type="application/pdf"
        aria-label={`Preview of ${title}`}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

function formatGeneratedAt(timestamp: string | null): string {
  if (!timestamp) {
    return 'Unknown time';
  }
  return new Date(timestamp).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

export function SpecialGallery({ groups, generatedAt, supplementaryHref }: SpecialGalleryProps) {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Figure Collection</h1>
        <p>
          Auto-generated figures for each analysis.
        </p>
        <p className={styles.timestamp}>Last updated: {formatGeneratedAt(generatedAt)}</p>
        {supplementaryHref ? (
          <p className={styles.downloadLink}>
            <a href={supplementaryHref} download>
              Download supplementary tables (XLSX)
            </a>
          </p>
        ) : (
          <p className={styles.downloadLinkMissing}>
            Supplementary tables will appear here after the next data refresh.
          </p>
        )}
      </header>

      <nav className={styles.nav} aria-label="Figure sections">
        <ul>
          {groups.map((group) => (
            <li key={group.slug}>
              <a href={`#${group.slug}`}>{group.title}</a>
            </li>
          ))}
        </ul>
      </nav>

      <main>
        {groups.map((group) => (
          <section key={group.slug} id={group.slug} className={styles.section}>
            <h2>{group.title}</h2>
            <div className={styles.grid}>
              {group.figures.map((figure) => {
                const assetPath = `${BASE_PATH}/${figure.filename}`;
                return (
                  <figure key={`${group.slug}-${figure.filename}`} className={styles.figure}>
                    <figcaption>{figure.title}</figcaption>
                    <PDFPreview assetPath={assetPath} title={figure.title} />
                    {figure.description ? (
                      <p className={styles.description}>{figure.description}</p>
                    ) : null}
                    <p className={styles.links}>
                      <a href={assetPath}>Download PDF</a>
                    </p>
                  </figure>
                );
              })}
            </div>
          </section>
        ))}
      </main>

      <footer className={styles.footer}>
        <p>
          Source available on{' '}
          <a href="https://github.com/ferromic/ferromic">github.com/ferromic/ferromic</a>.
        </p>
      </footer>
    </div>
  );
}
