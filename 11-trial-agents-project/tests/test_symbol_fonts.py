"""Symbol-font private codes are translated before any agent reads a passage.

    corpus     966 of 5,764 chunks carry U+F0xx codes; after normalize(),
               none remain — each is its character or a visible [?]
    reading    the IMbrave150 hypertension criterion reads "≥ 150 … > 100",
               the two DIFFERENT signs the protocol prints
    fonts      U+F06C is a Wingdings bullet here, not Symbol's λ
    wiring     every tool's passages pass through _passage(), headings too,
               and the module is packaged into the Lambda zip

Corpus tests run on the real Pinecone export (skipped without it).
"""
import json
import re

import pytest

import fakes
from fakes import DATA

HANDLER = fakes.load_lambda("trial_search")          # puts lambda_tools on sys.path
import symbol_fonts                                    # noqa: E402

PRIVATE = re.compile(r"[\uf020-\uf0ff]")
HAS_DATA = (DATA / "dump.json").exists()


@pytest.fixture(scope="module")
def corpus():
    return json.loads((DATA / "dump.json").read_text())["vectors"]


@pytest.mark.skipif(not HAS_DATA, reason="no real data")
def test_no_symbol_font_code_survives_in_any_passage(corpus):
    affected = [v for v in corpus if PRIVATE.search(v["metadata"].get("text", ""))]
    assert len(affected) == 966
    unreadable = 0
    for v in affected:
        p = HANDLER._passage(v["id"], v["metadata"], "search")
        assert not PRIVATE.search(p["text"]), v["id"]
        assert not any(PRIVATE.search(h) for h in p["headings"]), v["id"]
        unreadable += p["text"].count(symbol_fonts.UNREADABLE)
    assert unreadable == 78          # only U+F0E0, alone in table cells, has no known meaning


@pytest.mark.skipif(not HAS_DATA, reason="no real data")
def test_imbrave150_hypertension_reads_both_of_its_different_signs(corpus):
    v = next(v for v in corpus if v["id"].endswith("47cf5c238f992845:0"))
    text = HANDLER._passage(v["id"], v["metadata"], "search")["text"]
    assert "systolic blood pressure [BP] ≥ 150 mmHg and/or diastolic BP > 100 mmHg" in text
    assert "average of ≥ 3 BP readings on ≥ 2 sessions" in text


@pytest.mark.skipif(not HAS_DATA, reason="no real data")
def test_wingdings_bullets_do_not_become_greek_letters(corpus):
    v = next(v for v in corpus if "\uf06c Event or laboratory" in v["metadata"].get("text", ""))
    text = symbol_fonts.normalize(v["metadata"]["text"])
    assert "• Event or laboratory" in text and "λ" not in text


def test_known_thresholds_and_units():
    raw = ("Grade \uf0b3 3; \uf0a3 10 mg/day; albumin \uf03c 2.5; ALT \uf03e 10 \uf0b4 ULN; "
           "Day 1 \uf0b1 1 week; 181,000/\uf06dL; 38.5 \uf0b0C; TNF-\uf061")
    assert symbol_fonts.normalize(raw) == ("Grade ≥ 3; ≤ 10 mg/day; albumin < 2.5; ALT > 10 × ULN; "
                                           "Day 1 ± 1 week; 181,000/µL; 38.5 °C; TNF-α")


def test_an_unknown_code_is_marked_never_guessed():
    assert symbol_fonts.normalize("| \uf0e0") == "| [?]"
    assert symbol_fonts.normalize("\ue123 outside the symbol range") == "\ue123 outside the symbol range"
    assert symbol_fonts.normalize("") == "" and symbol_fonts.normalize(None) is None


def test_the_module_is_packaged_into_the_lambda_zip(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ts_lambda_deploy_sf", fakes.ROOT / "trial_search" / "infra" / "lambda_deploy.py")
    ld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld)
    assert "symbol_fonts.py" in {p.name for p in ld._source_files()}
