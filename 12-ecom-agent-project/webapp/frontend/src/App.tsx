// The shell: signed out -> LoginScreen; signed in -> the conversation screen.
//   /                      a new conversation
//   /conversation/:convId  an existing one
//   /admin                 the admin console (admins only)
import { Navigate, Route, Routes } from "react-router-dom";
import { Loading } from "./components/ui/primitives";
import { useCurrentUser } from "./lib/auth";
import { AdminScreen } from "./screens/AdminScreen";
import { ConversationScreen } from "./screens/ConversationScreen";
import { LoginScreen } from "./screens/LoginScreen";

export default function App() {
  const { user, loading } = useCurrentUser();
  if (loading) return <Loading label="Loading" />;
  if (!user) return <LoginScreen />;
  return (
    <Routes>
      <Route path="/" element={<ConversationScreen />} />
      <Route path="/conversation/:convId" element={<ConversationScreen />} />
      {user.role === "Admin" && <Route path="/admin" element={<AdminScreen />} />}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
