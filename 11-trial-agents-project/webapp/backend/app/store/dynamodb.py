"""DynamoDB backend — one table, two indexes (the reference's single-table design).

    PK               SK                          what                 GSI1 (user views)            GSI2 (AgentOps)
    ───────────────  ──────────────────────────  ───────────────────  ───────────────────────────  ─────────────────────────
    USER#<u>         PROFILE                     name, password hash
    CONV#<c>         META                        title, counts        USER#<u>     / updated_at
    CONV#<c>         MSG#<created_at>#<m>        question or answer                                INTERACTION / created_at
                                                                                                   (answers only)
    CONV#<c>         FB#<m>#<u>                  feedback on answer   FEEDBACK#<u> / created_at    FEEDBACK    / created_at

    history         CONV#<c>, MSG# newest first, Limit, role/text/status only
    sidebar         GSI1  USER#<u>, newest first
    my feedback     GSI1  FEEDBACK#<u>
    AgentOps        GSI2  INTERACTION since <time>;  FEEDBACK since <time>
    delete          every item under CONV#<c> — messages and feedback go with it

WHY answer details are a JSON string

An answer's details (agents, queries, citations, artifact) are nested and
only ever read whole. One string attribute avoids DynamoDB's number and
nesting rules and keeps the item's size predictable; answers.py caps it
well under the 400 KB item limit.

THE TABLE — ONE SCHEMA, TWO CREATORS

    AWS run     the CloudFormation stack creates it (webapp/deploy/template.yaml)
    local run   ensure_ready() creates it on start, from TABLE_SCHEMA below
    Both describe the same keys and indexes; a test compares them, so a local
    table can never drift from the deployed one.

WHAT THIS DOES NOT DO

    Search is by title and the conversation's own questions (search_text,
    kept on META), filtered in the query — enough per user; a full-text
    index across all messages would need OpenSearch.
    It never creates a table it was pointed at with APP_TABLE: that one is
    the stack's, and a missing one is reported, not silently replaced.
"""
import json
from decimal import Decimal

import boto3
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

from .. import settings
from .base import Store, new_id, now_iso

_ser, _de = TypeSerializer(), TypeDeserializer()
GSI1, GSI2 = "GSI1", "GSI2"

# Must match the Table resource in webapp/deploy/template.yaml — checked by
# tests/test_backend.py::test_local_table_schema_matches_the_stack.
TABLE_SCHEMA = {
    "BillingMode": "PAY_PER_REQUEST",
    "AttributeDefinitions": [{"AttributeName": n, "AttributeType": "S"}
                             for n in ("PK", "SK", "GSI1PK", "GSI1SK", "GSI2PK", "GSI2SK")],
    "KeySchema": [{"AttributeName": "PK", "KeyType": "HASH"},
                  {"AttributeName": "SK", "KeyType": "RANGE"}],
    "GlobalSecondaryIndexes": [
        {"IndexName": index,
         "KeySchema": [{"AttributeName": f"{index}PK", "KeyType": "HASH"},
                       {"AttributeName": f"{index}SK", "KeyType": "RANGE"}],
         "Projection": {"ProjectionType": "ALL"}}
        for index in (GSI1, GSI2)],
}


def _item(values: dict) -> dict:
    """Plain dict -> DynamoDB attribute map. Floats become Decimals."""
    clean = json.loads(json.dumps(values), parse_float=Decimal)
    return {k: _ser.serialize(v) for k, v in clean.items() if v is not None}


def _plain(item: dict) -> dict:
    out = {k: _de.deserialize(v) for k, v in item.items()}
    return {k: (int(v) if isinstance(v, Decimal) and v == v.to_integral_value() else
                float(v) if isinstance(v, Decimal) else v) for k, v in out.items()}


_KEYS = ("PK", "SK", "GSI1PK", "GSI1SK", "GSI2PK", "GSI2SK")


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _KEYS}


class DynamoStore(Store):
    def __init__(self, client=None, table: str | None = None):
        self.table = table or settings.TABLE
        self.ddb = client or boto3.client(
            "dynamodb", region_name=settings.REGION,
            **({"endpoint_url": settings.DDB_ENDPOINT} if settings.DDB_ENDPOINT else {}))

    # ── the table ────────────────────────────────────────────────────────
    def ensure_ready(self, create: bool) -> str:
        """Make sure the table exists and is ACTIVE, before the first request.

        STEP 1  describe it — present: wait until ACTIVE, "ready"
        STEP 2  missing and this run owns it (a local run): create it from
                TABLE_SCHEMA, wait until ACTIVE, "created"
        STEP 3  missing and it was named with APP_TABLE: stop, and say so

        Called once at startup (main.py). Doing it on the first request
        instead left that request — the first sign-in — waiting about ten
        seconds while the table was created.
        """
        waiter = self.ddb.get_waiter("table_exists")
        wait = {"TableName": self.table, "WaiterConfig": {"Delay": 2, "MaxAttempts": 90}}
        try:
            self.ddb.describe_table(TableName=self.table)
        except self.ddb.exceptions.ResourceNotFoundException:
            if not create:
                raise RuntimeError(
                    f"DynamoDB table {self.table!r} does not exist in {settings.REGION}. "
                    "It is created by the web app's CloudFormation stack "
                    "(webapp/deploy/deploy.py); or unset APP_TABLE to use the local "
                    f"table {settings.LOCAL_TABLE!r}, which is created automatically.")
            self.ddb.create_table(TableName=self.table, **TABLE_SCHEMA)
            waiter.wait(**wait)
            return "created"
        waiter.wait(**wait)
        return "ready"

    # ── helpers ──────────────────────────────────────────────────────────
    def _query(self, **kwargs) -> list[dict]:
        items, start = [], None
        while True:
            page = self.ddb.query(TableName=self.table, **kwargs,
                                  **({"ExclusiveStartKey": start} if start else {}))
            items += [_plain(i) for i in page.get("Items", [])]
            start = page.get("LastEvaluatedKey")
            if not start:
                return items

    def _get(self, pk: str, sk: str) -> dict | None:
        item = self.ddb.get_item(TableName=self.table,
                                 Key=_item({"PK": pk, "SK": sk})).get("Item")
        return _plain(item) if item else None

    # ── users ────────────────────────────────────────────────────────────
    def get_user(self, username):
        found = self._get(f"USER#{username}", "PROFILE")
        return _strip(found) if found else None

    def create_user(self, user):
        try:
            self.ddb.put_item(TableName=self.table,
                              Item=_item({"PK": f"USER#{user['username']}", "SK": "PROFILE", **user}),
                              ConditionExpression="attribute_not_exists(PK)")
            return True
        except self.ddb.exceptions.ConditionalCheckFailedException:
            return False

    # ── conversations ────────────────────────────────────────────────────
    def create_conversation(self, username, title):
        now = now_iso()
        conv = {"conversation_id": new_id(), "username": username, "title": title,
                "created_at": now, "updated_at": now, "message_count": 0, "search_text": ""}
        self.ddb.put_item(TableName=self.table, Item=_item({
            "PK": f"CONV#{conv['conversation_id']}", "SK": "META",
            "GSI1PK": f"USER#{username}", "GSI1SK": now, **conv}))
        return conv

    def get_conversation(self, conversation_id):
        found = self._get(f"CONV#{conversation_id}", "META")
        return _strip(found) if found else None

    def list_conversations(self, username):
        items = self._query(IndexName=GSI1, ScanIndexForward=False,
                            KeyConditionExpression="GSI1PK = :u",
                            ExpressionAttributeValues=_item({":u": f"USER#{username}"}))
        return [_strip(i) for i in items]

    def rename_conversation(self, conversation_id, title):
        self.ddb.update_item(TableName=self.table,
                             Key=_item({"PK": f"CONV#{conversation_id}", "SK": "META"}),
                             UpdateExpression="SET title = :t",
                             ExpressionAttributeValues=_item({":t": title}))

    def delete_conversation(self, conversation_id):
        # STEP 1 every item under the conversation: META, messages, feedback
        keys = [{"PK": i["PK"], "SK": i["SK"]} for i in self._query(
            KeyConditionExpression="PK = :p",
            ExpressionAttributeValues=_item({":p": f"CONV#{conversation_id}"}))]
        # STEP 2 delete in batches of 25 (the BatchWriteItem limit), retrying leftovers
        for start in range(0, len(keys), 25):
            requests = [{"DeleteRequest": {"Key": _item(k)}} for k in keys[start:start + 25]]
            for _ in range(5):
                left = self.ddb.batch_write_item(
                    RequestItems={self.table: requests}).get("UnprocessedItems", {})
                requests = left.get(self.table, [])
                if not requests:
                    break

    # ── messages ─────────────────────────────────────────────────────────
    def add_message(self, message):
        cid, created = message["conversation_id"], message["created_at"]
        record = {"PK": f"CONV#{cid}", "SK": f"MSG#{created}#{message['message_id']}",
                  **{k: v for k, v in message.items() if k not in ("details", "interaction")}}
        if message.get("details") is not None:
            record["details_json"] = json.dumps(message["details"])
        if message.get("interaction"):
            record.update({"GSI2PK": "INTERACTION", "GSI2SK": created,
                           "interaction_json": json.dumps(message["interaction"])})
        self.ddb.put_item(TableName=self.table, Item=_item(record))

        # the conversation: activity time (sidebar order), count, searchable questions
        conv = self.get_conversation(cid) or {}
        search = conv.get("search_text", "")
        if message["role"] == "user":
            search = (search + " " + message["text"].lower())[:2000]
        self.ddb.update_item(
            TableName=self.table, Key=_item({"PK": f"CONV#{cid}", "SK": "META"}),
            UpdateExpression="SET updated_at = :t, GSI1SK = :t, search_text = :s "
                             "ADD message_count :one",
            ExpressionAttributeValues=_item({":t": created, ":s": search, ":one": 1}))

    def _message(self, item: dict) -> dict:
        record = _strip(item)
        if "details_json" in record:
            record["details"] = json.loads(record.pop("details_json"))
        if "interaction_json" in record:
            record["interaction"] = json.loads(record.pop("interaction_json"))
        return record

    def list_messages(self, conversation_id):
        items = self._query(KeyConditionExpression="PK = :p AND begins_with(SK, :m)",
                            ExpressionAttributeValues=_item({":p": f"CONV#{conversation_id}",
                                                             ":m": "MSG#"}))
        return [self._message(i) for i in items]

    def recent_messages(self, conversation_id, limit):
        """One Query, newest first, stopping at `limit`, projecting three small
        attributes. Its cost no longer grows with the conversation: reading
        every message with its details cost 16 read units at turn 10 and ~86
        at turn 50. `role`, `text` and `status` are DynamoDB reserved words,
        hence the #names."""
        page = self.ddb.query(
            TableName=self.table, ScanIndexForward=False, Limit=limit,
            KeyConditionExpression="PK = :p AND begins_with(SK, :m)",
            ProjectionExpression="#r, #t, #s",
            ExpressionAttributeNames={"#r": "role", "#t": "text", "#s": "status"},
            ExpressionAttributeValues=_item({":p": f"CONV#{conversation_id}", ":m": "MSG#"}))
        return list(reversed([_plain(i) for i in page.get("Items", [])]))

    def get_message(self, conversation_id, message_id):
        return next((m for m in self.list_messages(conversation_id)
                     if m["message_id"] == message_id), None)

    # ── feedback ─────────────────────────────────────────────────────────
    def put_feedback(self, feedback):
        self.ddb.put_item(TableName=self.table, Item=_item({
            "PK": f"CONV#{feedback['conversation_id']}",
            "SK": f"FB#{feedback['message_id']}#{feedback['username']}",
            "GSI1PK": f"FEEDBACK#{feedback['username']}", "GSI1SK": feedback["created_at"],
            "GSI2PK": "FEEDBACK", "GSI2SK": feedback["created_at"], **feedback}))

    def feedback_by_user(self, username):
        return [_strip(i) for i in self._query(
            IndexName=GSI1, ScanIndexForward=False, KeyConditionExpression="GSI1PK = :u",
            ExpressionAttributeValues=_item({":u": f"FEEDBACK#{username}"}))]

    def feedback_for_conversation(self, conversation_id):
        return [_strip(i) for i in self._query(
            KeyConditionExpression="PK = :p AND begins_with(SK, :f)",
            ExpressionAttributeValues=_item({":p": f"CONV#{conversation_id}", ":f": "FB#"}))]

    def feedback_since(self, since):
        return [_strip(i) for i in self._query(
            IndexName=GSI2, KeyConditionExpression="GSI2PK = :k AND GSI2SK >= :s",
            ExpressionAttributeValues=_item({":k": "FEEDBACK", ":s": since}))]

    # ── interactions (AgentOps) ──────────────────────────────────────────
    def interactions_since(self, since, username=None):
        items = self._query(IndexName=GSI2, ScanIndexForward=False,
                            KeyConditionExpression="GSI2PK = :k AND GSI2SK >= :s",
                            ExpressionAttributeValues=_item({":k": "INTERACTION", ":s": since}))
        out = [json.loads(i["interaction_json"]) for i in items if "interaction_json" in i]
        return [i for i in out if username is None or i["username"] == username]
