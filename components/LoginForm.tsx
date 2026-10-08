"use client";

import { useState } from "react";

import { Alert, Button, inputClass } from "@/components/ui";
import { sendLoginLink } from "@/lib/client-actions";

export function LoginForm({ next, initialError }: { next: string; initialError?: string }) {
  const [email, setEmail] = useState("");
  const [sending, setSending] = useState(false);
  const [sentTo, setSentTo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(initialError ?? null);

  if (sentTo) {
    return (
      <Alert tone="success">
        Le hemos enviado un enlace de acceso a <strong>{sentTo}</strong>. Ábralo en este navegador para entrar.
      </Alert>
    );
  }

  return (
    <form className="space-y-4" onSubmit={async (e) => {
      e.preventDefault();
      setSending(true);
      setError(null);
      try {
        await sendLoginLink(email.trim(), next);
        setSentTo(email.trim());
      } catch (err) {
        setError(err instanceof Error ? err.message : "No se pudo enviar el enlace");
      } finally {
        setSending(false);
      }
    }}>
      <div>
        <label htmlFor="email" className="mb-1.5 block text-sm font-medium">Email</label>
        <input id="email" type="email" autoComplete="email" required value={email}
               onChange={(e) => setEmail(e.target.value)} placeholder="nombre@farmacia.es" className={inputClass} />
      </div>
      {error && <Alert>{error}</Alert>}
      <Button type="submit" disabled={sending} className="w-full">
        {sending ? "Enviando…" : "Enviar enlace de acceso"}
      </Button>
    </form>
  );
}
