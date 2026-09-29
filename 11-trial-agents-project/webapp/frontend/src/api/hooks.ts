// Server state through React Query: one hook per backend route.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type { Conversation, Feedback, Interaction, Message, UsageSummary, User } from "./types";

export const keys = {
  me: ["me"] as const,
  conversations: (q: string) => ["conversations", q] as const,
  messages: (id: string) => ["messages", id] as const,
  myFeedback: ["feedback", "mine"] as const,
};

export const useMe = () =>
  useQuery({ queryKey: keys.me, queryFn: () => api<User>("/api/auth/me"), retry: false });

export const useConversations = (q: string) =>
  useQuery({ queryKey: keys.conversations(q),
             queryFn: () => api<Conversation[]>(`/api/conversations?q=${encodeURIComponent(q)}`) });

export const useMessages = (id: string | undefined) =>
  useQuery({ queryKey: keys.messages(id ?? ""), enabled: !!id,
             queryFn: () => api<Message[]>(`/api/conversations/${id}/messages`) });

function useInvalidate() {
  const qc = useQueryClient();
  return (prefix: string) => qc.invalidateQueries({ queryKey: [prefix] });
}

export function useRenameConversation() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) =>
      api<Conversation>(`/api/conversations/${id}`, { method: "PATCH", body: { title } }),
    onSuccess: () => invalidate("conversations"),
  });
}

export function useDeleteConversation() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (id: string) => api<void>(`/api/conversations/${id}`, { method: "DELETE" }),
    onSuccess: () => invalidate("conversations"),
  });
}

export function useGiveFeedback() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (body: { conversation_id: string; message_id: string; rating: "up" | "down"; comment: string }) =>
      api<Feedback>("/api/feedback", { method: "POST", body }),
    onSuccess: () => { invalidate("messages"); invalidate("feedback"); invalidate("agentops"); },
  });
}

export const useMyFeedback = () =>
  useQuery({ queryKey: keys.myFeedback, queryFn: () => api<Feedback[]>("/api/feedback") });

export const useUsage = (days: number) =>
  useQuery({ queryKey: ["agentops", "summary", days],
             queryFn: () => api<UsageSummary>(`/api/agentops/summary?days=${days}`) });

export const useInteractions = (days: number) =>
  useQuery({ queryKey: ["agentops", "interactions", days],
             queryFn: () => api<Interaction[]>(`/api/agentops/interactions?days=${days}`) });

export const useAllFeedback = (days: number) =>
  useQuery({ queryKey: ["agentops", "feedback", days],
             queryFn: () => api<Feedback[]>(`/api/agentops/feedback?days=${days}`) });
