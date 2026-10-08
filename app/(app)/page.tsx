import { redirect } from "next/navigation";

import { getSessionContext } from "@/lib/data";

export default async function Home() {
  const { activeOrg } = await getSessionContext();
  redirect(activeOrg ? "/formulacion" : "/admin");
}
