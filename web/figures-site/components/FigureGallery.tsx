import styles from '../styles/Gallery.module.css';

type FigureType = 'image' | 'pdf';

export interface FigureItem {
  name: string;
  href: string;
  preview: string;
  type: FigureType;
}

export interface FigureGroup {
  title: string;
  slug: string;
  items: FigureItem[];
}

export interface FigureGalleryProps {
  groups: FigureGroup[];
  generatedAt: string | null;
}

function formatGeneratedAt(generatedAt: string | null): string {
  if (!generatedAt) {
    return 'Unknown time';
  }
  return new Date(generatedAt).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
}

export function FigureGallery({ groups, generatedAt }: FigureGalleryProps) {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Ferromic Analysis Figures</h1>
        <p>
          Automatically generated visualisations from the Ferromic analysis pipeline.
        </p>
        <p className={styles.timestamp}>
          Last updated: {formatGeneratedAt(generatedAt)}
        </p>
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
        {groups.length === 0 ? (
          <section className={styles.emptyState}>
            <h2>No figures yet</h2>
            <p>
              Run the <code>run-analysis</code> workflow to populate this gallery with the
              latest outputs.
            </p>
          </section>
        ) : (
          groups.map((group) => (
            <section key={group.slug} id={group.slug} className={styles.section}>
              <h2>{group.title}</h2>
              <div className={styles.grid}>
                {group.items.map((item) => (
                  <figure key={item.href} className={styles.figure}>
                    <figcaption>{item.name}</figcaption>
                    <a href={item.href} className={styles.preview}>
                      {item.type === 'pdf' ? (
                        <object
                          data={item.preview}
                          type="application/pdf"
                          aria-label={`Preview of ${item.name}`}
                        />
                      ) : (
                        <img src={item.preview} alt={`Preview of ${item.name}`} />
                      )}
                    </a>
                    <p className={styles.links}>
                      <a href={item.href}>Download</a>
                    </p>
                  </figure>
                ))}
              </div>
            </section>
          ))
        )}
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
