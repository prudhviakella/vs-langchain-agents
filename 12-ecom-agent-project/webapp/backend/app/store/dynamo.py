"""DynamoDB store — ACT's single-table design, trimmed to this app.

    PK            SK                     item
    USER#<u>      PROFILE                name · password_hash
    CONV#<c>      META                   title · owner · last_activity_ns   GSI1: USER#<owner> / <activity> (sidebar)
    CONV#<c>      MSG#<ns 20d>#<msg>     text · status · details_json {reasoning, entities, sources, artifacts,
                                         artifact_summary, options, usage_by_agent, …}
    CONV#<c>      FB#<msg>#<user>        vote · comment · question · answer excerpt
                                         GSI1: FEEDBACK#<yyyy-mm> / <ns>          (Feedback hub)
    the MSG item of a FINISHED answer   GSI1: USAGE#<yyyy-mm-dd> / <ns>#<msg>    (Usage Analytics)
    — the same item, findable a second way: no second write, no scan

FROM ACT
    - Nested data is ONE JSON string (details_json): DynamoDB rejects floats, and
      the Plotly figures and table rows round-trip exactly as a string.
    - The 400 KB item limit: a PutItem rejected for size is retried with the TABLE
      cut to 200, 100, then 50 rows (charts are a few KB; the table is what grows),
      and as a last resort with NO artifacts — the answer and reasoning are never
      lost for a chart the user can re-request. Found in ACT from a production
      failure that was losing whole messages.

WHAT THIS DOES NOT DO
    It creates the table only when missing (on-demand); it never changes or deletes one.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from ..models import Message, MessageFeedback

log = logging.getLogger("store.dynamo")


def _day(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc).date().isoformat()
_SHRINK = (None, 200, 100, 50)
_DETAIL_FIELDS = ("reasoning", "artifacts", "artifact_summary", "entities", "options", "sources",
                  "usage_by_agent", "agents_used", "truncation_note")


class DynamoStore:
    def __init__(self, table: str, region: str):
        self.ddb = boto3.resource("dynamodb", region_name=region)
        self.name = table
        self._ensure_table()
        self.t = self.ddb.Table(table)

    def _ensure_table(self) -> None:
        client = self.ddb.meta.client
        try:
            client.describe_table(TableName=self.name)
            return
        except client.exceptions.ResourceNotFoundException:
            pass
        log.warning("creating DynamoDB table %s (on-demand)", self.name)
        client.create_table(
            TableName=self.name, BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": a, "AttributeType": "S"} for a in ("PK", "SK", "GSI1PK", "GSI1SK")],
            KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
            GlobalSecondaryIndexes=[{"IndexName": "GSI1", "Projection": {"ProjectionType": "ALL"},
                                     "KeySchema": [{"AttributeName": "GSI1PK", "KeyType": "HASH"},
                                                   {"AttributeName": "GSI1SK", "KeyType": "RANGE"}]}])
        client.get_waiter("table_exists").wait(TableName=self.name)

    # ── users ────────────────────────────────────────────────────────────
    def get_user(self, user_id: str) -> dict | None:
        item = self.t.get_item(Key={"PK": f"USER#{user_id}", "SK": "PROFILE"}).get("Item")
        return json.loads(item["body"]) if item else None

    def create_user(self, user: dict) -> bool:
        """False when the username is taken (a conditional put: two sign-ups cannot both win)."""
        try:
            self.t.put_item(Item={"PK": f"USER#{user['user_id']}", "SK": "PROFILE", "body": json.dumps(user)},
                            ConditionExpression="attribute_not_exists(PK)")
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    # ── conversations ────────────────────────────────────────────────────
    def put_conversation(self, conv: dict) -> None:
        self.t.put_item(Item={"PK": f"CONV#{conv['conversation_id']}", "SK": "META",
                              "GSI1PK": f"USER#{conv['owner_id']}", "GSI1SK": f"{conv['last_activity_ns']:020d}",
                              "body": json.dumps(conv)})

    def get_conversation(self, conversation_id: str) -> dict | None:
        item = self.t.get_item(Key={"PK": f"CONV#{conversation_id}", "SK": "META"}).get("Item")
        return json.loads(item["body"]) if item else None

    def list_conversations(self, owner_id: str, limit: int = 100) -> list[dict]:
        items = self.t.query(IndexName="GSI1", KeyConditionExpression=Key("GSI1PK").eq(f"USER#{owner_id}"),
                             ScanIndexForward=False, Limit=limit)["Items"]
        return [json.loads(i["body"]) for i in items]

    def delete_conversation(self, conversation_id: str) -> None:
        kwargs = {"KeyConditionExpression": Key("PK").eq(f"CONV#{conversation_id}"), "ProjectionExpression": "PK, SK"}
        with self.t.batch_writer() as batch:
            while True:
                page = self.t.query(**kwargs)
                for i in page["Items"]:
                    batch.delete_item(Key={"PK": i["PK"], "SK": i["SK"]})
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    # ── messages ─────────────────────────────────────────────────────────
    def _item(self, m: Message, details: dict) -> dict:
        top = m.model_dump(exclude=set(_DETAIL_FIELDS) | {"feedback"})
        item = {"PK": f"CONV#{m.conversation_id}", "SK": f"MSG#{m.created_at_ns:020d}#{m.message_id}",
                "message_id": m.message_id, "body": json.dumps(top), "details_json": json.dumps(details)}
        if m.role == "assistant" and m.status in ("complete", "failed", "stopped"):
            item["GSI1PK"] = f"USAGE#{_day(m.created_at_ns)}"
            item["GSI1SK"] = f"{m.created_at_ns:020d}#{m.message_id}"
        return item

    def put_message(self, m: Message) -> None:
        """STEP 1 full item · STEP 2-4 table cut to 200 / 100 / 50 rows · STEP 5 no artifacts."""
        base = {f: getattr(m, f) for f in _DETAIL_FIELDS}
        for cap in _SHRINK:
            details = dict(base)
            table = (base.get("artifacts") or {}).get("table")
            if cap is not None and table and len(table.get("rows") or []) > cap:
                art = dict(base["artifacts"])
                art["table"] = {**table, "rows": table["rows"][:cap], "truncated": True}
                details["artifacts"] = art
                details["truncation_note"] = f"Table stored as its first {cap} of {len(table['rows'])} rows (size limit)."
            try:
                self.t.put_item(Item=self._item(m, details))
                if cap is not None:
                    log.warning("[store] message %s stored with the table cut to %d rows", m.message_id, cap)
                return
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "ValidationException" or "Item size" not in str(exc):
                    raise                       # a different problem — never mask it
        log.warning("[store] message %s too large even truncated; storing without artifacts", m.message_id)
        details = {**base, "artifacts": {}, "truncation_note": "Results were too large to store; re-run the question to see them."}
        self.t.put_item(Item=self._item(m, details))

    @staticmethod
    def _message(item: dict) -> Message:
        return Message(**json.loads(item["body"]), **json.loads(item.get("details_json") or "{}"))

    def list_messages(self, conversation_id: str) -> list[Message]:
        out, kwargs = [], {"KeyConditionExpression": Key("PK").eq(f"CONV#{conversation_id}") & Key("SK").begins_with("MSG#")}
        while True:
            page = self.t.query(**kwargs)
            out += [self._message(i) for i in page["Items"]]
            if "LastEvaluatedKey" not in page:
                return out
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def get_message(self, conversation_id: str, message_id: str) -> Message | None:
        return next((m for m in self.list_messages(conversation_id) if m.message_id == message_id), None)

    def answers_between(self, start_ns: int, end_ns: int) -> list[Message]:
        """Finished answers in [start, end): one GSI1 query per UTC day — never a scan."""
        out, day = [], dt.datetime.fromtimestamp(start_ns / 1e9, dt.timezone.utc).date()
        last = dt.datetime.fromtimestamp((end_ns - 1) / 1e9, dt.timezone.utc).date()
        while day <= last:
            kwargs = {"IndexName": "GSI1", "KeyConditionExpression": Key("GSI1PK").eq(f"USAGE#{day.isoformat()}")
                      & Key("GSI1SK").between(f"{start_ns:020d}", f"{end_ns:020d}")}
            while True:
                page = self.t.query(**kwargs)
                out += [self._message(i) for i in page["Items"]]
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
            day += dt.timedelta(days=1)
        return out

    # ── feedback ─────────────────────────────────────────────────────────
    def put_feedback(self, conversation_id: str, message_id: str, user_id: str, vote: str | None, comment: str | None,
                     at_ns: int, context: dict | None = None) -> None:
        self.t.put_item(Item={"PK": f"CONV#{conversation_id}", "SK": f"FB#{message_id}#{user_id}",
                              "GSI1PK": f"FEEDBACK#{_day(at_ns)[:7]}", "GSI1SK": f"{at_ns:020d}",
                              "body": json.dumps({"vote": vote, "comment": comment, "at_ns": at_ns, **(context or {})})})

    def feedback_between(self, start_ns: int, end_ns: int) -> list[dict]:
        out, month = [], dt.datetime.fromtimestamp(start_ns / 1e9, dt.timezone.utc).date().replace(day=1)
        last = dt.datetime.fromtimestamp((end_ns - 1) / 1e9, dt.timezone.utc).date()
        while month <= last:
            items = self.t.query(IndexName="GSI1", KeyConditionExpression=Key("GSI1PK").eq(f"FEEDBACK#{month.isoformat()[:7]}")
                                 & Key("GSI1SK").between(f"{start_ns:020d}", f"{end_ns:020d}"))["Items"]
            for i in items:
                _, mid, uid = i["SK"].split("#", 2)
                out.append({"conversation_id": i["PK"][5:], "message_id": mid, "user_id": uid, **json.loads(i["body"])})
            month = (month + dt.timedelta(days=32)).replace(day=1)
        return out

    def feedback(self, conversation_id: str, user_id: str) -> dict[str, MessageFeedback]:
        items = self.t.query(KeyConditionExpression=Key("PK").eq(f"CONV#{conversation_id}") & Key("SK").begins_with("FB#"))["Items"]
        out: dict[str, MessageFeedback] = {}
        for i in items:
            _, mid, uid = i["SK"].split("#", 2)
            f = json.loads(i["body"])
            fb = out.setdefault(mid, MessageFeedback(message_id=mid))
            fb.likes += f["vote"] == "up"
            fb.dislikes += f["vote"] == "down"
            if f.get("comment"):
                fb.comments.append({"user_id": uid, "comment": f["comment"], "created_at_ns": f["at_ns"]})
            if uid == user_id:
                fb.my_vote, fb.my_comment = f["vote"], f.get("comment")
        return out
