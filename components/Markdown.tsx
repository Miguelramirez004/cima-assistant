"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// [Ref 1: NOMBRE (Nº Registro: 12345)] sin enlace -> enlace a la ficha técnica en CIMA
const REF_PATTERN = /\[Ref (\d+): ([^\]()]+?) \(Nº Registro: (\d+)\)\](?!\()/g;

export function linkReferences(text: string): string {
  return text.replace(REF_PATTERN, (_, n: string, name: string, nregistro: string) =>
    `[Ref ${n}: ${name} (Nº Registro: ${nregistro})](https://cima.aemps.es/cima/dochtml/ft/${nregistro}/FichaTecnica.html)`);
}

/** Markdown del modelo, sin HTML crudo; enlaces solo http(s), en pestaña nueva. */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-cima">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: label }) =>
            href && /^https?:\/\//.test(href)
              ? <a href={href} target="_blank" rel="noopener noreferrer">{label}</a>
              : <span>{label}</span>,
        }}
      >
        {linkReferences(children)}
      </ReactMarkdown>
    </div>
  );
}
