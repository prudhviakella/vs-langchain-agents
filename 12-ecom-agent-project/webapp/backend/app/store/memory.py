"""In-process store with DynamoStore's interface — for tests. Nothing survives a restart."""
from __future__ import annotations

import copy

from ..models import Message, MessageFeedback


class MemoryStore:
    def __init__(self):
        self.users, self.convs, self.msgs, self.fb = {}, {}, {}, {}

    # users
    def get_user(self, user_id: str) -> dict | None:
        return copy.deepcopy(self.users.get(user_id))

    def create_user(self, user: dict) -> bool:
        if user["user_id"] in self.users:
            return False
        self.users[user["user_id"]] = copy.deepcopy(user)
        return True

    # conversations
    def put_conversation(self, conv: dict) -> None:
        self.convs[conv["conversation_id"]] = copy.deepcopy(conv)

    def get_conversation(self, conversation_id: str) -> dict | None:
        return copy.deepcopy(self.convs.get(conversation_id))

    def list_conversations(self, owner_id: str, limit: int = 100) -> list[dict]:
        mine = [c for c in self.convs.values() if c["owner_id"] == owner_id]
        return copy.deepcopy(sorted(mine, key=lambda c: c["last_activity_ns"], reverse=True)[:limit])

    def delete_conversation(self, conversation_id: str) -> None:
        self.convs.pop(conversation_id, None)
        self.msgs = {k: v for k, v in self.msgs.items() if v.conversation_id != conversation_id}
        self.fb = {k: v for k, v in self.fb.items() if k[0] != conversation_id}

    # messages
    def put_message(self, m: Message) -> None:
        self.msgs[m.message_id] = m.model_copy(deep=True)

    def get_message(self, conversation_id: str, message_id: str) -> Message | None:
        m = self.msgs.get(message_id)
        return m.model_copy(deep=True) if m and m.conversation_id == conversation_id else None

    def list_messages(self, conversation_id: str) -> list[Message]:
        return [m.model_copy(deep=True) for m in sorted(self.msgs.values(), key=lambda m: m.created_at_ns)
                if m.conversation_id == conversation_id]

    def answers_between(self, start_ns: int, end_ns: int) -> list[Message]:
        """Finished answers (complete / failed / stopped) created in [start, end)."""
        return [m.model_copy(deep=True) for m in self.msgs.values() if m.role == "assistant"
                and m.status in ("complete", "failed", "stopped") and start_ns <= m.created_at_ns < end_ns]

    # feedback
    def put_feedback(self, conversation_id: str, message_id: str, user_id: str, vote: str | None, comment: str | None,
                     at_ns: int, context: dict | None = None) -> None:
        self.fb[(conversation_id, message_id, user_id)] = {"vote": vote, "comment": comment, "at_ns": at_ns,
                                                           **(context or {})}

    def feedback_between(self, start_ns: int, end_ns: int) -> list[dict]:
        return [{"conversation_id": c, "message_id": m, "user_id": u, **f} for (c, m, u), f in self.fb.items()
                if start_ns <= f["at_ns"] < end_ns]

    def feedback(self, conversation_id: str, user_id: str) -> dict[str, MessageFeedback]:
        out: dict[str, MessageFeedback] = {}
        for (cid, mid, uid), f in self.fb.items():
            if cid != conversation_id:
                continue
            fb = out.setdefault(mid, MessageFeedback(message_id=mid))
            fb.likes += f["vote"] == "up"
            fb.dislikes += f["vote"] == "down"
            if f["comment"]:
                fb.comments.append({"user_id": uid, "comment": f["comment"], "created_at_ns": f["at_ns"]})
            if uid == user_id:
                fb.my_vote, fb.my_comment = f["vote"], f["comment"]
        return out
