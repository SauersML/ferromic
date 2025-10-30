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
}

const BASE_PATH = 'figures/special';

function formatGeneratedAt(timestamp: string | null): string {
  if (!timestamp) {
    return 'Unknown time';
  }
  return new Date(timestamp).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

export function SpecialGallery({ groups, generatedAt }: SpecialGalleryProps) {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Special Figure Collection</h1>
        <p>
          Curated Ferromic figures requested for the specialised gallery. All previews
          render directly from the source PDF files.
        </p>
        <p className={styles.timestamp}>Last updated: {formatGeneratedAt(generatedAt)}</p>
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
                    <a href={assetPath} className={styles.preview}>
                      <object
                        data={assetPath}
                        type="application/pdf"
                        aria-label={`Preview of ${figure.title}`}
                      />
                    </a>
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
