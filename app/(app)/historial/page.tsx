import Link from "next/link";

import { HistoryList } from "@/components/HistoryList";
import { NoActiveOrg } from "@/components/NoActiveOrg";
import { PageHeader } from "@/components/ui";
import { getSessionContext, listConversations, listFormulations, listProspectos } from "@/lib/data";

export const metadata = { title: "Historial · CIMA Assistant" };

const TABS = [
  { key: "formulaciones", label: "Formulaciones" },
  { key: "prospectos", label: "Prospectos" },
  { key: "consultas", label: "Consultas" },
] as const;

export default async function HistorialPage({ searchParams }: PageProps<"/historial">) {
  const { activeOrg } = await getSessionContext();
  if (!activeOrg) return <NoActiveOrg />;
  const { tab: rawTab } = await searchParams;
  const tab = TABS.find((t) => t.key === rawTab)?.key ?? "formulaciones";

  return (
    <>
      <PageHeader title="Historial"
                  description={`Sus resultados en ${activeOrg.name}. Solo usted puede verlos.`} />
      <nav className="mb-5 flex gap-1 border-b border-line" aria-label="Tipo de historial">
        {TABS.map((t) => (
          <Link key={t.key} href={`/historial?tab=${t.key}`} aria-current={t.key === tab ? "page" : undefined}
                className={`-mb-px border-b-2 px-4 py-2 text-sm ${t.key === tab
                  ? "border-accent font-medium text-accent-strong" : "border-transparent text-muted hover:text-foreground"}`}>
            {t.label}
          </Link>
        ))}
      </nav>
      {tab === "formulaciones" && <HistoryList key={tab} kind="formulations" items={await listFormulations(activeOrg.id)} />}
      {tab === "prospectos" && <HistoryList key={tab} kind="prospectos" items={await listProspectos(activeOrg.id)} />}
      {tab === "consultas" && <HistoryList key={tab} kind="conversations" items={await listConversations(activeOrg.id)} />}
    </>
  );
}
