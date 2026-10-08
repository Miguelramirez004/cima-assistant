"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { References } from "@/components/References";
import { useActiveOrg } from "@/components/SessionProvider";
import { Alert, Button, inputClass } from "@/components/ui";
import { streamConsulta } from "@/lib/client-actions";
import type { ChatMessage, Conversation, Reference } from "@/lib/types";

const EXAMPLES = [
  "¿Cuáles son las contraindicaciones del ibuprofeno?",
  "¿Qué medicamentos están indicados para la hipertensión?",
  "¿Puedo tomar omeprazol durante el embarazo?",
  "Interacciones del clopidogrel con omeprazol",
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

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-8 lg:self-start">
        <Link href="/consultas" className="mb-3 block rounded-lg border border-line bg-white px-3 py-2 text-center text-sm font-medium hover:border-accent hover:text-accent-strong">
          + Nueva conversación
        </Link>
        <ul className="max-h-[60vh] space-y-0.5 overflow-y-auto" aria-label="Conversaciones">
          {conversations.map((c) => (
            <li key={c.id}>
              <Link href={`/consultas/${c.id}`}
                    aria-current={c.id === conversationId ? "page" : undefined}
                    className={`block truncate rounded-md px-2.5 py-1.5 text-sm ${c.id === conversationId
                      ? "bg-accent-soft font-medium text-accent-strong" : "text-slate-600 hover:bg-surface"}`}>
                {c.title ?? "Sin título"}
              </Link>
            </li>
          ))}
        </ul>
      </aside>

      <section className="flex min-h-[70vh] flex-col">
        <div className="flex-1 space-y-6 pb-4">
          {!messages.length && !pending && (
            <div className="py-10 text-center">
              <p className="font-medium">Pregunte sobre cualquier medicamento registrado en CIMA</p>
              <p className="mt-1 text-sm text-muted">Las respuestas se basan solo en las fichas técnicas oficiales y citan sus fuentes.</p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((example) => (
                  <button key={example} onClick={() => send(example)}
                          className="rounded-full border border-line px-3 py-1.5 text-xs text-slate-600 hover:border-accent hover:text-accent-strong">
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => message.role === "user"
            ? <UserBubble key={message.id} text={message.content} />
            : <AssistantMessage key={message.id} text={message.content} reasoning={message.reasoning} references={message.references} />)}

          {pending && (
            <div>
              <ol className="mb-3 space-y-1 border-l-2 border-accent/40 pl-3 text-xs text-muted" aria-live="polite">
                {pending.trace.map((step, i) => <li key={i}>{step}</li>)}
                {!pending.text && <li className="animate-pulse">Consultando CIMA (AEMPS)…</li>}
              </ol>
              {pending.text && <Markdown>{pending.text}</Markdown>}
            </div>
          )}
          {error && <Alert>{error}</Alert>}
          <div ref={bottom} />
        </div>

        <form onSubmit={(e) => { e.preventDefault(); send(input); }}
              className="sticky bottom-0 border-t border-line bg-white pb-2 pt-3">
          <div className="flex items-end gap-2">
            <label htmlFor="question" className="sr-only">Consulta</label>
            <textarea id="question" rows={2} maxLength={2000} value={input} disabled={Boolean(pending)}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
                      placeholder="Escriba su consulta sobre medicamentos…" className={`${inputClass} resize-none`} />
            <Button type="submit" disabled={Boolean(pending) || input.trim().length < 3}>Enviar</Button>
          </div>
          <p className="mt-1.5 text-xs text-muted">Información de la AEMPS con fines informativos; no sustituye el criterio profesional.</p>
        </form>
      </section>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-surface px-4 py-2.5 text-sm ring-1 ring-line">{text}</p>
    </div>
  );
}

function AssistantMessage({ text, reasoning, references }: { text: string; reasoning: string | null; references: Reference[] }) {
  return (
    <div>
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
