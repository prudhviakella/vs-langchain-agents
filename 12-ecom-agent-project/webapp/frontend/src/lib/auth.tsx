// Who is signed in. GET /me answers 401 when the session cookie is missing or expired —
// that is "signed out", not an error.
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useMe } from "../api/hooks";
import type { Analyst } from "../api/types";

export function useCurrentUser(): { user?: Analyst; loading: boolean } {
  const me = useMe();
  return { user: me.data, loading: me.isLoading };
}

export function useSignOut() {
  const qc = useQueryClient();
  return async () => {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    qc.clear();                        // the next user must see nothing of this one
    window.location.assign("/");
  };
}
