"""In-process store: tests and zero-setup local runs. Nothing is durable.

WHAT THIS DOES NOT DO

    It does not survive a restart or share data between processes — never
    run it behind the load balancer.
"""
import copy

from .base import Store, new_id, now_iso


class MemoryStore(Store):
    def __init__(self):
        self.users, self.conversations, self.messages = {}, {}, {}
        self.feedback = {}                 # (conversation_id, message_id, username) -> dict

    def get_user(self, username):
        return copy.deepcopy(self.users.get(username))

    def create_user(self, user):
        if user["username"] in self.users:
            return False
        self.users[user["username"]] = copy.deepcopy(user)
        return True

    def create_conversation(self, username, title):
        now = now_iso()
        conv = {"conversation_id": new_id(), "username": username, "title": title,
                "created_at": now, "updated_at": now, "message_count": 0, "search_text": ""}
        self.conversations[conv["conversation_id"]] = conv
        self.messages[conv["conversation_id"]] = []
        return copy.deepcopy(conv)

    def get_conversation(self, conversation_id):
        return copy.deepcopy(self.conversations.get(conversation_id))

    def list_conversations(self, username):
        mine = [c for c in self.conversations.values() if c["username"] == username]
        return copy.deepcopy(sorted(mine, key=lambda c: c["updated_at"], reverse=True))

    def rename_conversation(self, conversation_id, title):
        self.conversations[conversation_id]["title"] = title

    def delete_conversation(self, conversation_id):
        self.conversations.pop(conversation_id, None)
        self.messages.pop(conversation_id, None)
        for key in [k for k in self.feedback if k[0] == conversation_id]:
            del self.feedback[key]

    def add_message(self, message):
        cid = message["conversation_id"]
        self.messages[cid].append(copy.deepcopy(message))
        conv = self.conversations[cid]
        conv["updated_at"] = message["created_at"]
        conv["message_count"] += 1
        if message["role"] == "user":
            conv["search_text"] = (conv["search_text"] + " " + message["text"].lower())[:2000]

    def list_messages(self, conversation_id):
        return copy.deepcopy(self.messages.get(conversation_id, []))

    def get_message(self, conversation_id, message_id):
        return next((copy.deepcopy(m) for m in self.messages.get(conversation_id, [])
                     if m["message_id"] == message_id), None)

    def recent_messages(self, conversation_id, limit):
        return [{"role": m["role"], "text": m["text"], "status": m.get("status", "complete")}
                for m in self.messages.get(conversation_id, [])[-limit:]]

    def put_feedback(self, feedback):
        key = (feedback["conversation_id"], feedback["message_id"], feedback["username"])
        self.feedback[key] = copy.deepcopy(feedback)

    def feedback_by_user(self, username):
        return sorted((copy.deepcopy(f) for f in self.feedback.values()
                       if f["username"] == username),
                      key=lambda f: f["created_at"], reverse=True)

    def feedback_for_conversation(self, conversation_id):
        return [copy.deepcopy(f) for k, f in self.feedback.items() if k[0] == conversation_id]

    def feedback_since(self, since):
        return [copy.deepcopy(f) for f in self.feedback.values() if f["created_at"] >= since]

    def interactions_since(self, since, username=None):
        out = []
        for msgs in self.messages.values():
            for m in msgs:
                if m["role"] == "assistant" and m.get("interaction") \
                        and m["created_at"] >= since \
                        and (username is None or m["interaction"]["username"] == username):
                    out.append(copy.deepcopy(m["interaction"]))
        return sorted(out, key=lambda i: i["created_at"], reverse=True)


__all__ = ["MemoryStore", "now_iso", "new_id"]
