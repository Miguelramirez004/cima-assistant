import { NoActiveOrg } from "@/components/NoActiveOrg";
import { ProspectoView } from "@/components/ProspectoView";
import { PageHeader } from "@/components/ui";
import { getSessionContext } from "@/lib/data";

export const metadata = { title: "Prospectos · CIMA Assistant" };

export default async function ProspectosPage({ searchParams }: PageProps<"/prospectos">) {
  const { activeOrg } = await getSessionContext();
  const { q } = await searchParams;
  return (
    <>
      <PageHeader title="Prospectos"
                  description="Prospectos en el formato oficial de la AEMPS, a partir del prospecto registrado en CIMA." />
      {activeOrg ? <ProspectoView initialQuery={typeof q === "string" ? q : ""} /> : <NoActiveOrg />}
    </>
  );
}
