import { NextResponse, type NextRequest } from "next/server";

import { IS_PREVIEW } from "@/lib/env";
import { updateSession } from "@/lib/supabase/proxy";

const PUBLIC_PREFIXES = ["/login", "/auth/", "/invite/"];

export async function proxy(request: NextRequest) {
  if (IS_PREVIEW) {
    return NextResponse.next();
  }

  const { response, isAuthenticated } = await updateSession(request);
  const { pathname, search } = request.nextUrl;
  const isPublic = PUBLIC_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(prefix));

  if (!isAuthenticated && !isPublic) {
    const login = request.nextUrl.clone();
    login.pathname = "/login";
    login.search = pathname === "/" ? "" : `?next=${encodeURIComponent(pathname + search)}`;
    return NextResponse.redirect(login);
  }
  return response;
}

export const config = {
  // Todo salvo la API (tiene su propia autenticación) y los recursos estáticos
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)"],
};
