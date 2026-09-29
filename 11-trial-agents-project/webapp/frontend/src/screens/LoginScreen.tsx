// Sign in with username + password. An unknown username gets a second step:
// first and last name, then the account is created and signed in.
//
//   submit ──► POST /api/auth/login
//                200            signed in (cookie set by the server)
//                404 new_user   show first/last name; submit again creates the user
//                401            wrong username or password
import { useState } from "react";
import { ApiError, api } from "../api/client";
import type { User } from "../api/types";
import { Button, Card, Input } from "../components/ui";

export function LoginScreen({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [form, setForm] = useState({ username: "", password: "", first_name: "", last_name: "" });
  const [isNew, setIsNew] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: e.target.value });

  async function submit(e: React.FormEvent) {
    e.preventDefault(); setError(""); setBusy(true);
    try {
      const body = isNew ? form : { username: form.username, password: form.password };
      onSignedIn(await api<User>("/api/auth/login", { method: "POST", body }));
    } catch (err) {
      const detail = err instanceof ApiError ? err.detail as { code?: string } : null;
      if (err instanceof ApiError && err.status === 404 && detail?.code === "new_user") setIsNew(true);
      else setError(err instanceof ApiError && err.status === 401 ? "Wrong username or password."
        : err instanceof Error ? err.message : "Sign-in failed.");
    } finally { setBusy(false); }
  }

  return (
    <div className="flex h-screen items-center justify-center">
      <Card className="w-full max-w-sm p-7">
        <h1 className="text-lg font-semibold text-ink">Trial Agents</h1>
        <p className="mb-4 text-sm text-muted">{isNew ? "New here — tell us your name." : "Sign in to continue."}</p>
        <form className="space-y-3" onSubmit={submit}>
          <Input placeholder="Username" aria-label="Username" value={form.username} onChange={set("username")}
            autoComplete="username" disabled={isNew} required />
          <Input placeholder="Password" aria-label="Password" type="password" value={form.password}
            onChange={set("password")} autoComplete={isNew ? "new-password" : "current-password"} minLength={8} required />
          {isNew && (<>
            <Input placeholder="First name" aria-label="First name" value={form.first_name} onChange={set("first_name")} required />
            <Input placeholder="Last name" aria-label="Last name" value={form.last_name} onChange={set("last_name")} required />
          </>)}
          {error && <p className="text-sm text-bad" role="alert">{error}</p>}
          <Button type="submit" className="w-full justify-center" disabled={busy}>
            {isNew ? "Create account" : "Sign in"}</Button>
          {isNew && <Button type="button" variant="ghost" className="w-full justify-center"
            onClick={() => setIsNew(false)}>Use a different username</Button>}
        </form>
      </Card>
    </div>
  );
}
