// React Query hooks — one per backend call the screens make.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type { Analyst, ConversationPage, LatestInteraction, Message, Page, Vote } from "./types";

export const useMe = () =>
  useQuery({ queryKey: ["me"], queryFn: () => api<Analyst>("/me"), retry: false });

export const useConversations = (q: string) =>
  useQuery({
    queryKey: ["conversations", q],
    queryFn: () =>
      q ? api<ConversationPage>("/conversations/search", { query: { q } }) : api<ConversationPage>("/conversations"),
  });

export const useMessages = (convId?: string) =>
  useQuery({
    queryKey: ["messages", convId],
    queryFn: () => api<Page<Message>>(`/conversations/${convId}/messages`),
    enabled: !!convId,
  });

export const useLatest = (convId?: string) =>
  useQuery({
    queryKey: ["latest", convId],
    queryFn: () => api<LatestInteraction>(`/conversations/${convId}/latest-interaction`),
    enabled: !!convId,
    staleTime: 0,
  });

export function useRename() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: { id: string; title: string }) =>
      api(`/conversations/${b.id}`, { method: "PATCH", body: { title: b.title } }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}

export function useDelete() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api(`/conversations/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });
}

export function useFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (b: { conversation_id: string; message_id: string; vote?: Vote; comment?: string }) =>
      api("/feedback", { method: "POST", body: b }),
    onSuccess: (_d, v) => qc.invalidateQueries({ queryKey: ["messages", v.conversation_id] }),
  });
}
