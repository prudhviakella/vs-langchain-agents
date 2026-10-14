// Sign in — or create an account with a new username (the trial app's flow):
//   known username  -> password checked
//   new username    -> 404 new_user -> ask for first + last name -> account created
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { Button, Field, inputCls } from "../components/ui/primitives";

export function LoginScreen() {
  const qc = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [first, setFirst] = useState("");
  const [last, setLast] = useState("");
  const [isNew, setIsNew] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const body = isNew ? { username, password, first_name: first, last_name: last } : { username, password };
      await api("/auth/login", { method: "POST", body });
      await qc.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) setIsNew(true);
      else setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-canvas px-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border border-line bg-card p-7 shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-ink">Store Analytics</h1>
          <p className="mt-1 text-sm text-muted">Ask questions about orders, customers, products and delivery.</p>
        </div>
        <Field label="Username">
          <input className={inputCls} value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
        </Field>
        <Field label="Password">
          <input className={inputCls} type="password" value={password} minLength={8}
                 onChange={(e) => setPassword(e.target.value)} required />
        </Field>
        {isNew && (
          <>
            <p className="text-sm text-body">New username — add your name to create the account.</p>
            <Field label="First name">
              <input className={inputCls} value={first} onChange={(e) => setFirst(e.target.value)} required />
            </Field>
            <Field label="Last name">
              <input className={inputCls} value={last} onChange={(e) => setLast(e.target.value)} required />
            </Field>
          </>
        )}
        {error && <p className="text-sm text-flag-crit">{error}</p>}
        <Button type="submit" variant="primary" disabled={busy} className="w-full">
          {busy ? "Signing in…" : isNew ? "Create account" : "Sign in"}
        </Button>
      </form>
    </div>
  );
}
