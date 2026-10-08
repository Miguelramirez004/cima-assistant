"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { LoginForm } from "@/components/LoginForm";
import { Alert, Button, Spinner } from "@/components/ui";
import { acceptInvitation, signOut } from "@/lib/client-actions";
import { IS_PREVIEW } from "@/lib/env";
import * as fx from "@/lib/preview/fixtures";
import { getSupabaseBrowserClient } from "@/lib/supabase/client";

type State = { status: "loading" } | { status: "signed-out" } | { status: "signed-in"; email: string };

export function InviteAccept({ token }: { token: string }) {
  const router = useRouter();
  const [state, setState] = useState<State>({ status: "loading" });
  const [accepting, setAccepting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      if (IS_PREVIEW) {
        setState({ status: "signed-in", email: fx.PREVIEW_USER.email });
        return;
      }
      const supabase = getSupabaseBrowserClient();
      // Los emails de invitación de Supabase traen la sesión en el fragmento (#access_token=...)
      const hash = new URLSearchParams(window.location.hash.slice(1));
      const accessToken = hash.get("access_token");
      const refreshToken = hash.get("refresh_token");
      if (accessToken && refreshToken) {
        const { error: sessionError } = await supabase.auth.setSession({ access_token: accessToken, refresh_token: refreshToken });
        window.history.replaceState(null, "", window.location.pathname);
        if (sessionError) setError("El enlace del email no es válido o ha caducado.");
      } else if (hash.get("error_description")) {
        setError(hash.get("error_description"));
        window.history.replaceState(null, "", window.location.pathname);
      }
      const { data } = await supabase.auth.getUser();
      setState(data.user ? { status: "signed-in", email: data.user.email ?? "" } : { status: "signed-out" });
    };
    void init();
  }, []);

  const accept = async () => {
    setAccepting(true);
    setError(null);
    try {
      await acceptInvitation(token);
      router.push("/");
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo aceptar la invitación");
      setAccepting(false);
    }
  };

  if (state.status === "loading") return <Spinner label="Comprobando la invitación…" />;

  if (state.status === "signed-out") {
    return (
      <div className="space-y-4">
        {error && <Alert>{error}</Alert>}
        <p className="text-sm text-muted">Inicie sesión con el email al que se envió la invitación.</p>
        <LoginForm next={`/invite/${token}`} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm">Ha iniciado sesión como <strong>{state.email}</strong>.</p>
      {error && <Alert>{error}</Alert>}
      <Button onClick={accept} disabled={accepting} className="w-full">
        {accepting ? "Uniéndose…" : "Aceptar invitación"}
      </Button>
      <button className="w-full text-center text-xs text-muted underline-offset-2 hover:underline"
              onClick={async () => { await signOut(); setState({ status: "signed-out" }); }}>
        No soy yo: usar otro email
      </button>
    </div>
  );
}
