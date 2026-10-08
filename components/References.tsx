import type { Reference } from "@/lib/types";

/** Fuentes oficiales citadas: enlaces a la ficha técnica en CIMA. */
export function References({ references }: { references: Reference[] }) {
  const safe = references.filter((r) => /^https:\/\/cima\.aemps\.es\//.test(r.url));
  if (!safe.length) return null;
  return (
    <div className="mt-4">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Fuentes (CIMA · AEMPS)</p>
      <ul className="flex flex-wrap gap-2">
        {safe.map((ref) => (
          <li key={ref.url}>
            <a href={ref.url} target="_blank" rel="noopener noreferrer"
               className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1 text-xs hover:border-accent hover:text-accent-strong">
              <span className="truncate">{ref.title}</span>
              {ref.nregistro && <span className="text-muted">· {ref.nregistro}</span>}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
