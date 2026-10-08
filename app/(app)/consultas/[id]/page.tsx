import { notFound } from "next/navigation";

import { ChatWorkspace } from "@/components/ChatWorkspace";
import { NoActiveOrg } from "@/components/NoActiveOrg";
import { PageHeader } from "@/components/ui";
import { getConversationMessages, getSessionContext, listConversations } from "@/lib/data";

export const metadata = { title: "Consultas CIMA · CIMA Assistant" };

export default async function ConversationPage({ params }: PageProps<"/consultas/[id]">) {
  const { id } = await params;
  const { activeOrg } = await getSessionContext();
  if (!activeOrg) return <NoActiveOrg />;
  const [conversations, messages] = await Promise.all([
    listConversations(activeOrg.id),
    getConversationMessages(activeOrg.id, id),
  ]);
  if (messages === null) notFound();
  return (
    <>
      <PageHeader title="Consultas CIMA" />
      <ChatWorkspace key={id} conversations={conversations} conversationId={id} messages={messages} />
    </>
  );
}
