import { redirect } from "next/navigation";

import { SessionProvider } from "@/components/SessionProvider";
import { Sidebar } from "@/components/Sidebar";
import { getSessionContext } from "@/lib/data";

export default async function AppLayout({ children }: LayoutProps<"/">) {
  const session = await getSessionContext();
  if (!session.memberships.length && !session.user.isPlatformAdmin) {
    redirect("/sin-organizacion");
  }

  return (
    <SessionProvider value={session}>
      <div className="flex min-h-screen flex-col md:flex-row">
        <Sidebar />
        <main className="min-w-0 flex-1 px-4 py-6 md:px-10 md:py-8">
          <div className="mx-auto w-full max-w-4xl">{children}</div>
        </main>
      </div>
    </SessionProvider>
  );
}
