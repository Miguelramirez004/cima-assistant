"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { Aparecer, EscanerDocumento, EstructuraMolecular, ProcesoEnCurso, Spinner, usePlaceholderRotativo } from "@/components/motion";
import { References } from "@/components/References";
import { useActiveOrg } from "@/components/SessionProvider";
import { Alert, Button, inputClass } from "@/components/ui";
import { streamConsulta } from "@/lib/client-actions";
import type { ChatMessage, Conversation, Reference } from "@/lib/types";

// Sugerencias de la portada, con la sección de la ficha técnica que suele responderlas
const EXAMPLES = [
  { q: "¿Cuáles son las contraindicaciones del ibuprofeno?", cat: "Contraindicaciones", sec: "4.3", color: "#b42318",
    icon: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM5.6 5.6l12.8 12.8" },
  { q: "¿Qué medicamentos están indicados para la hipertensión?", cat: "Indicaciones", sec: "4.1", color: "#1d4ed8",
    icon: "M3 12h4l3-7 4 14 3-7h4" },
  { q: "¿Puedo tomar omeprazol durante el embarazo?", cat: "Embarazo y lactancia", sec: "4.6", color: "#a3195b",
    icon: "M12 21s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.6-7 10-7 10z" },
  { q: "Interacciones del clopidogrel con omeprazol", cat: "Interacciones", sec: "4.5", color: "#9a4a06",
    icon: "M7 7h10M7 7l3-3M7 7l3 3M17 17H7M17 17l-3-3M17 17l-3 3" },
];

const GARANTIAS = [
  { text: "Fichas técnicas y prospectos oficiales", icon: "M7 3h7l5 5v13H7zM14 3v5h5" },
  { text: "Cada respuesta cita su fuente", icon: "M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" },
  { text: "Solo información de CIMA (AEMPS)", icon: "M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6zM9 12l2 2 4-4" },
];

const PLACEHOLDERS = [
  "Escriba su consulta sobre medicamentos…",
  "¿Dosis de amoxicilina en niños?",
  "¿Interacciones de la warfarina?",
  "¿Puedo partir los comprimidos de enalapril?",
];

interface Pending {
  question: string;
  trace: string[];
  text: string;
  references: Reference[];
}

export function ChatWorkspace({ conversations: initialConversations, conversationId: initialId, messages: initialMessages }: {
  conversations: Conversation[];
  conversationId: string | null;
  messages: ChatMessage[];
}) {
  const { org } = useActiveOrg();
  const [conversations, setConversations] = useState(initialConversations);
  const [conversationId, setConversationId] = useState(initialId);
  const [messages, setMessages] = useState(initialMessages);
  const [pending, setPending] = useState<Pending | null>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const placeholder = usePlaceholderRotativo(PLACEHOLDERS);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.length, pending?.text, pending?.trace.length]);

  const send = async (text: string) => {
    const question = text.trim();
    if (question.length < 3 || pending) return;
    setInput("");
    setError(null);
    const now = new Date().toISOString();
    setMessages((m) => [...m, { id: `local-${now}`, role: "user", content: question, reasoning: null, references: [], created_at: now }]);
    setPending({ question, trace: [], text: "", references: [] });

    let currentId = conversationId;
    try {
      await streamConsulta(org.id, question, conversationId, (event) => {
        switch (event.type) {
          case "conversation":
            if (!currentId) {
              currentId = event.conversation_id;
              setConversationId(currentId);
              setConversations((list) => [{ id: currentId!, title: question.slice(0, 80), updated_at: now }, ...list]);
              window.history.replaceState(null, "", `/consultas/${currentId}`);
            }
            break;
          case "trace":
            setPending((p) => p && { ...p, trace: [...p.trace, event.message] });
            break;
          case "references":
            setPending((p) => p && { ...p, references: event.references });
            break;
          case "token":
            setPending((p) => p && { ...p, text: p.text + event.text });
            break;
          case "done":
            setMessages((m) => [...m, {
              id: event.message_id, role: "assistant", content: event.answer, reasoning: event.reasoning,
              references: event.references, created_at: new Date().toISOString(),
            }]);
            setPending(null);
            break;
          case "error":
            setError(event.detail);
            break;
        }
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo completar la consulta");
    } finally {
      setPending(null);
    }
  };

  const isEmpty = !messages.length && !pending;

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-8 lg:self-start">
        <Link href="/consultas" className="mb-3 flex min-h-11 items-center justify-center gap-2 rounded-lg bg-accent px-3 text-sm font-semibold text-white transition-colors hover:bg-accent-strong">
          <svg aria-hidden="true" viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
          Nueva consulta
        </Link>
        <ul className="max-h-[60vh] space-y-0.5 overflow-y-auto" aria-label="Conversaciones">
          {conversations.map((c) => (
            <li key={c.id}>
              <Link href={`/consultas/${c.id}`}
                    aria-current={c.id === conversationId ? "page" : undefined}
                    className={`block truncate rounded-md px-2.5 py-1.5 text-sm ${c.id === conversationId
                      ? "bg-accent-soft font-medium text-accent-strong" : "text-slate-700 hover:bg-surface"}`}>
                {c.title ?? "Sin título"}
              </Link>
            </li>
          ))}
        </ul>
      </aside>

      <section className="flex min-h-[70vh] min-w-0 flex-col">
        <div className="flex-1 space-y-6 pb-4">
          {isEmpty && (
            <Portada input={input} setInput={setInput} placeholder={placeholder}
                     onSend={send} />
          )}

          {messages.map((message) => message.role === "user"
            ? <UserBubble key={message.id} text={message.content} />
            : <AssistantMessage key={message.id} text={message.content} reasoning={message.reasoning} references={message.references} />)}

          {pending && !pending.text && (
            <ProcesoEnCurso ilustracion={<EscanerDocumento />} barra={false}
                            pasos={[...pending.trace, "Consultando CIMA (AEMPS)…"]} actual={pending.trace.length} />
          )}
          {pending?.text && (
            <div className="cima-funde cima-escribiendo rounded-xl border border-line bg-white p-5">
              <Markdown>{pending.text}</Markdown>
            </div>
          )}
          {error && <Alert>{error}</Alert>}
          <div ref={bottom} />
        </div>

        {!isEmpty && (
        <form onSubmit={(e) => { e.preventDefault(); send(input); }}
              className="sticky bottom-0 border-t border-line bg-background pb-2 pt-3">
          <div className="flex items-end gap-2">
            <label htmlFor="question" className="sr-only">Consulta</label>
            <textarea id="question" rows={2} maxLength={2000} value={input} disabled={Boolean(pending)}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                      placeholder={placeholder} className={`${inputClass} resize-none`} />
            <Button type="submit" disabled={Boolean(pending) || input.trim().length < 3}>
              {pending ? <><Spinner tono="claro" />Consultando…</> : "Enviar"}
            </Button>
          </div>
          <p className="mt-1.5 text-xs text-muted">Información de la AEMPS con fines informativos; no sustituye el criterio profesional.</p>
        </form>
        )}
      </section>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="cima-entra flex justify-end">
      <p className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-white px-4 py-2.5 text-sm font-medium ring-1 ring-line">{text}</p>
    </div>
  );
}

function AssistantMessage({ text, reasoning, references }: { text: string; reasoning: string | null; references: Reference[] }) {
  return (
    <div className="cima-funde rounded-xl border border-line bg-white p-5">
      <Markdown>{text}</Markdown>
      <References references={references} />
      {reasoning && (
        <details className="mt-3 text-xs text-muted">
          <summary className="cursor-pointer">Proceso de búsqueda</summary>
          <p className="mt-1 whitespace-pre-wrap border-l-2 border-line pl-3">{reasoning}</p>
        </details>
      )}
    </div>
  );
}

/** Portada de Consultas: buscador principal y sugerencias sobre una cuadrícula fina. */
function Portada({ input, setInput, placeholder, onSend }: {
  input: string; setInput: (v: string) => void; placeholder: string; onSend: (text: string) => void;
}) {
  return (
    <div className="relative isolate pb-6 pt-8 sm:pt-12">
      <div aria-hidden="true" className="cima-reticula absolute inset-x-0 -top-8 -z-10 h-[460px]" />
      <EstructuraMolecular className="absolute -right-12 -top-6 -z-10 hidden size-48 opacity-20 lg:block" />

      <Aparecer paso={0.06} className="relative mx-auto flex max-w-2xl flex-col items-center gap-6 text-center [&>*]:w-full">
        <div className="flex flex-col items-center gap-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-white px-3 py-1 text-xs font-bold tracking-wide text-accent-strong">
            <span aria-hidden="true" className="size-1.5 rounded-full bg-accent" />
            Fuente oficial · CIMA · AEMPS
          </span>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">¿Qué medicamento quiere consultar hoy?</h1>
          <p className="max-w-lg text-[15px] leading-relaxed text-muted">
            Indicaciones, posología, contraindicaciones o interacciones, a partir de las fichas técnicas oficiales y con sus fuentes citadas.
          </p>
        </div>

        <form onSubmit={(e) => { e.preventDefault(); onSend(input); }}
              className="flex w-full items-center gap-3 rounded-2xl border border-line bg-white p-1.5 pl-4 shadow-[0_12px_28px_-20px_rgba(20,60,48,0.35)] transition focus-within:border-accent focus-within:ring-4 focus-within:ring-accent/15">
          <svg aria-hidden="true" viewBox="0 0 24 24" className="size-[18px] shrink-0 text-muted" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
          <label htmlFor="question" className="sr-only">Consulta</label>
          <input id="question" maxLength={2000} value={input} onChange={(e) => setInput(e.target.value)}
                 placeholder={placeholder} autoComplete="off"
                 className="min-h-12 min-w-0 flex-1 bg-transparent text-[15px] outline-none placeholder:text-slate-400" />
          <Button type="submit" disabled={input.trim().length < 3} className="min-h-11 rounded-xl px-5">
            Consultar
            <svg aria-hidden="true" viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </Button>
        </form>

        <ul className="-mt-2 flex flex-wrap justify-center gap-x-6 gap-y-2 text-[13px] text-muted">
          {GARANTIAS.map((g) => (
            <li key={g.text} className="flex items-center gap-2">
              <svg aria-hidden="true" viewBox="0 0 24 24" className="size-[15px] text-accent" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d={g.icon} /></svg>
              {g.text}
            </li>
          ))}
        </ul>

        <div className="grid w-full gap-3 sm:grid-cols-2">
          {EXAMPLES.map((ex) => (
            <button key={ex.q} type="button" onClick={() => onSend(ex.q)}
                    className="cima-elevable group flex h-full flex-col gap-2.5 rounded-xl border border-line bg-white px-4 py-3.5 text-left">
              <span className="flex w-full items-center gap-2.5">
                <span aria-hidden="true" className="flex size-7 items-center justify-center rounded-lg bg-background" style={{ color: ex.color }}>
                  <svg viewBox="0 0 24 24" className="size-[15px]" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d={ex.icon} /></svg>
                </span>
                <span className="text-xs font-bold text-slate-700">{ex.cat}</span>
                <span className="relative ml-auto flex min-w-16 items-center justify-end">
                  <span className="rounded-md bg-background px-2 py-0.5 text-[11px] font-semibold text-muted transition-opacity group-hover:opacity-0">FT {ex.sec}</span>
                  <svg aria-hidden="true" viewBox="0 0 24 24" className="absolute right-0 size-4 -translate-x-1 text-accent opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
                </span>
              </span>
              <span className="text-[15px] font-semibold leading-snug">{ex.q}</span>
            </button>
          ))}
        </div>

        <p className="text-xs text-muted">Información de la AEMPS con fines informativos; no sustituye el criterio profesional.</p>
      </Aparecer>
    </div>
  );
}
