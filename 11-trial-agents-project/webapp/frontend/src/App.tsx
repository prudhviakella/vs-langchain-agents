// The app: signed out -> LoginScreen; signed in -> sidebar + routes.
//
//   GET /api/auth/me ── 401 ──► LoginScreen ── signed in ──► cache the user
//                    └─ 200 ──► AppShell
//   Sign out: POST /api/auth/logout, then every cached query is dropped, so the
//   next user never sees the previous user's conversations.
import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./api/client";
import { keys, useMe } from "./api/hooks";
import type { User } from "./api/types";
import { Sidebar } from "./components/Sidebar";
import { Spinner } from "./components/ui";
import { ChatProvider } from "./lib/chat";
import { AgentOpsScreen } from "./screens/AgentOpsScreen";
import { ConversationScreen } from "./screens/ConversationScreen";
import { LoginScreen } from "./screens/LoginScreen";
import { MyFeedbackScreen } from "./screens/MyFeedbackScreen";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 15_000, retry: 1, refetchOnWindowFocus: false } },
});

function Gate() {
  const qc = useQueryClient();
  const me = useMe();
  if (me.isLoading) return <div className="flex h-screen items-center justify-center"><Spinner /></div>;
  if (!me.data) return <LoginScreen onSignedIn={(u: User) => qc.setQueryData<User | null>(keys.me, u)} />;

  const signOut = async () => {
    await api<void>("/api/auth/logout", { method: "POST" }).catch(() => undefined);
    qc.clear();
    qc.setQueryData<User | null>(keys.me, null);
  };
  return (
    <ChatProvider>
      <div className="flex">
        <Sidebar user={me.data} onSignOut={signOut} />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<ConversationScreen />} />
            <Route path="/c/:conversationId" element={<ConversationScreen />} />
            <Route path="/feedback" element={<MyFeedbackScreen />} />
            <Route path="/agentops" element={<AgentOpsScreen />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </ChatProvider>
  );
}

export function App({ client = queryClient }: { client?: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <BrowserRouter><Gate /></BrowserRouter>
    </QueryClientProvider>
  );
}
