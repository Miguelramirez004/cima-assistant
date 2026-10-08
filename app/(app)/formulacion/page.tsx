import { FormulacionView } from "@/components/FormulacionView";
import { NoActiveOrg } from "@/components/NoActiveOrg";
import { PageHeader } from "@/components/ui";
import { getSessionContext } from "@/lib/data";

export const metadata = { title: "Formulación magistral · CIMA Assistant" };

export default async function FormulacionPage() {
  const { activeOrg } = await getSessionContext();
  return (
    <>
      <PageHeader title="Formulación magistral"
                  description="Formulaciones detalladas a partir de los medicamentos registrados en CIMA, con referencias a sus fichas técnicas." />
      {activeOrg ? <FormulacionView /> : <NoActiveOrg />}
    </>
  );
}
