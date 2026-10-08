"use client";

import { useRouter } from "next/navigation";

import { Button } from "@/components/ui";
import { signOut } from "@/lib/client-actions";

export function SignOutButton() {
  const router = useRouter();
  return (
    <Button variant="secondary" className="w-full" onClick={async () => {
      await signOut();
      router.push("/login");
      router.refresh();
    }}>
      Cerrar sesión
    </Button>
  );
}
