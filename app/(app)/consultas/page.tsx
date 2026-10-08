import { ChatWorkspace } from "@/components/ChatWorkspace";
import { NoActiveOrg } from "@/components/NoActiveOrg";
import { PageHeader } from "@/components/ui";
import { getSessionContext, listConversations } from "@/lib/data";

export const metadata = { title: "Consultas CIMA · CIMA Assistant" };

export default async function ConsultasPage() {
  const { activeOrg } = await getSessionContext();
  if (!activeOrg) return <NoActiveOrg />;
  const conversations = await listConversations(activeOrg.id);
  return (
    <>
      <PageHeader title="Consultas CIMA" />
      <ChatWorkspace key="new" conversations={conversations} conversationId={null} messages={[]} />
    </>
  );
}
