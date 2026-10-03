"""Offline tests for kb_to_pinecone.py — no Pinecone, no Bedrock.

    FakeIndex   mirrors the pinecone 10 client: same call signatures and
                response shapes (fetch().vectors, list() pages of .vectors,
                describe_index_stats().namespaces, query().matches), and it
                REJECTS what Pinecone rejects: null or nested metadata,
                non-string lists, metadata over 40 KB, wrong dimension
    embed()     deterministic: the same text always gives the same unit vector

    first full publish · unchanged re-run · edited row (incremental) · DELETE row ·
    row removed from a full feed · bad action · sheet without a builder

    python -m pytest test_kb_to_pinecone.py -q      (FEED env var = the workbook)
"""
import hashlib
import json
import math
import os
import shutil
from types import SimpleNamespace as NS

import openpyxl
import pytest

import kb_to_pinecone as K

FEED = os.environ.get("FEED", "../kb_feed_ecom.xlsx")


class FakeIndex:
    def __init__(self):
        self.data, self.upserts = {}, 0

    def upsert(self, *, vectors, namespace=""):
        for v in vectors:
            assert len(v["values"]) == K.DIMENSION, "wrong dimension"
            for k, val in v["metadata"].items():
                ok = isinstance(val, (str, int, float, bool)) or (isinstance(val, list) and all(isinstance(x, str) for x in val))
                assert ok, f"Pinecone would reject metadata {k}={val!r}"
            assert len(json.dumps(v["metadata"]).encode()) <= 40_960, "metadata over 40 KB"
            self.data.setdefault(namespace, {})[v["id"]] = (v["values"], dict(v["metadata"]))
            self.upserts += 1

    def fetch(self, *, ids, namespace=""):
        ns = self.data.get(namespace, {})
        return NS(vectors={i: NS(id=i, metadata=ns[i][1]) for i in ids if i in ns})

    def list(self, *, namespace="", prefix=None, limit=None):
        ids = sorted(self.data.get(namespace, {}))
        for i in range(0, len(ids), 100):
            yield NS(vectors=[NS(id=x) for x in ids[i:i + 100]])

    def delete(self, *, ids=None, namespace=""):
        for i in ids or []:
            self.data.get(namespace, {}).pop(i, None)

    def describe_index_stats(self):
        return NS(namespaces={ns: NS(vector_count=len(v)) for ns, v in self.data.items() if v})

    def query(self, *, vector, top_k, namespace="", include_metadata=False):
        scored = sorted(((sum(a * b for a, b in zip(vector, vals)), vid, meta)
                         for vid, (vals, meta) in self.data.get(namespace, {}).items()), reverse=True)
        return NS(matches=[NS(id=vid, score=s, metadata=meta if include_metadata else None) for s, vid, meta in scored[:top_k]])


class Embedder:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        v = [0.0] * K.DIMENSION
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % K.DIMENSION] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def run(paths, index, mode="full"):
    embed = Embedder()
    records, owned = K.records_from(paths)
    stats = K.publish(index, embed, records, owned, mode, full_reembed=False)
    return records, owned, stats, embed


def edited_copy(tmp_path, name, edit):
    """A copy of the feed with `edit(workbook)` applied."""
    path = tmp_path / name
    shutil.copy(FEED, path)
    wb = openpyxl.load_workbook(path)
    edit(wb)
    wb.save(path)
    return str(path)


def keep_rows(ws, keep):
    """Keep the header and the rows for which keep(row_dict) is True."""
    head = [c.value for c in ws[1]]
    rows = [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
    ws.delete_rows(2, ws.max_row)
    for r in rows:
        if keep(r):
            ws.append([r[h] for h in head])


def only_sheet(wb, sheet, keep):
    for s in wb.sheetnames[1:]:
        keep_rows(wb[s], keep if s == sheet else (lambda r: False))


# ── the tests ────────────────────────────────────────────────────────────
def test_first_full_publish_stores_every_live_record():
    index = FakeIndex()
    records, owned, stats, embed = run([FEED], index)
    live = [r for r in records if not r.exempt and r.action != "DELETE"]
    assert owned == {"nlq-schema", "nlq-examples", "common"}
    assert stats["embedded"] == len(live) == embed.calls
    assert stats["exempt"] == 8                                        # the lat/lon columns
    counts = {ns: len(v) for ns, v in index.data.items()}
    assert counts == {"nlq-schema": 14 + 185 - 8 + 16, "nlq-examples": 61, "common": 66 + 3}
    assert K.verify(index, Embedder(), records, owned, stats["first"], wait_s=0)


def test_examples_embed_the_question_and_carry_the_sql():
    index = FakeIndex()
    records, *_ = run([FEED], index)
    ex = next(r for r in records if r.metadata["type"] == "example")
    assert "SELECT" not in ex.embed_text and ex.metadata["question"] in ex.embed_text
    assert index.data["nlq-examples"][ex.id][1]["query"].startswith("SELECT")
    col = next(r for r in records if r.id == "nlq_column:ecom.orders.order_status")
    assert col.metadata["enum_values"] == ["Cancelled", "Delivered", "Processing", "Returned", "Shipped"]


def test_unchanged_rerun_embeds_nothing():
    index = FakeIndex()
    run([FEED], index)
    _, _, stats, embed = run([FEED], index)
    assert stats["embedded"] == 0 == embed.calls and stats["deleted"] == 0


def test_an_edited_row_reembeds_only_that_row(tmp_path):
    index = FakeIndex()
    run([FEED], index)
    def edit(wb):
        only_sheet(wb, "nlq_column", lambda r: (r["table"], r["column"]) == ("ecom.orders", "is_gift"))
        ws = wb["nlq_column"]; head = [c.value for c in ws[1]]
        ws.cell(2, head.index("action") + 1, "UPDATE")
        ws.cell(2, head.index("effective_date") + 1, "2026-11-01")
        ws.cell(2, head.index("description") + 1, "TRUE when the customer ticked 'this is a gift' at checkout.")
    change = edited_copy(tmp_path, "change.xlsx", edit)
    _, owned, stats, embed = run([FEED, change], index, mode="incremental")
    assert stats["embedded"] == 1 == embed.calls
    assert "ticked" in index.data["nlq-schema"]["nlq_column:ecom.orders.is_gift"][1]["description"]


def test_a_delete_row_removes_the_vector(tmp_path):
    index = FakeIndex()
    run([FEED], index)
    def edit(wb):
        only_sheet(wb, "glossary", lambda r: r["term"] == "Net 15")
        ws = wb["glossary"]; head = [c.value for c in ws[1]]
        ws.cell(2, head.index("action") + 1, "DELETE")
        ws.cell(2, head.index("effective_date") + 1, "2026-11-01")
    change = edited_copy(tmp_path, "delete.xlsx", edit)
    _, _, stats, _ = run([FEED, change], index, mode="incremental")
    assert "glossary:net_15" not in index.data["common"] and stats["deleted"] == 1
    assert len(index.data["common"]) == 66 + 3 - 1


def test_full_mode_removes_what_left_the_feed(tmp_path):
    index = FakeIndex()
    run([FEED], index)
    smaller = edited_copy(tmp_path, "smaller.xlsx",
                          lambda wb: keep_rows(wb["nlq_join"], lambda r: r["left_table"] != "ecom.reviews"))
    _, _, stats, embed = run([smaller], index)
    assert stats["deleted"] == 2 and embed.calls == 0                  # the 2 reviews joins
    assert not any(i.startswith("nlq_join:ecom.reviews") for i in index.data["nlq-schema"])


def test_a_bad_action_publishes_nothing(tmp_path):
    bad = edited_copy(tmp_path, "bad.xlsx", lambda wb: wb["glossary"].cell(2, 1, "UPSERT"))
    index = FakeIndex()
    with pytest.raises(SystemExit, match="bad action 'UPSERT'"):
        run([bad], index)
    assert index.upserts == 0


def test_a_sheet_without_a_builder_is_refused_not_dropped(tmp_path):
    def edit(wb):
        wb["nlc_node"].append(["INSERT", "2026-10-02", "test", "Customer", "A customer node"])
    feed = edited_copy(tmp_path, "nlc.xlsx", edit)
    with pytest.raises(SystemExit, match=r"\[nlc_node\] has 1 rows but no builder"):
        K.records_from([feed])
