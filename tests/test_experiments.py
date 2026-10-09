"""The experiment registry (docs/experiments): every entry follows the rules in its README, and
the site shows it to signed-in members only (Steve, 2026-10-09)."""

import re

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import ISSUER, PUBLIC_KEY, SECRET, sign_in

from mycomap_vision import experiments
from mycomap_vision.api import create_app
from mycomap_vision.signin import SigninConfig

ENTRIES = experiments.load_all()
IDS = [e.slug for e in ENTRIES]


def test_the_registry_has_entries_and_every_one_parses():
    assert len(ENTRIES) >= 18
    assert len(set(IDS)) == len(IDS)


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_every_entry_has_the_required_fields_and_a_known_status(e):
    for field in experiments.REQUIRED:
        assert str(e.meta[field]).strip(), field
    assert e.meta["status"] in experiments.STATUSES


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_every_entry_has_the_template_sections(e):
    for section in ("Question", "Why it matters", "Setup", "What we tried", "Results",
                    "Verdict", "Decision", "Next"):
        assert f"## {section}\n" in e.body, section


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_related_entries_exist(e):
    assert set(e.meta.get("related") or []) <= set(IDS)


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_no_record_ids_coordinates_or_photo_paths_in_an_entry(e):
    text = e.path.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\d.,-])\d{9,10}(?![\d.,-])", text), "looks like a record id"
    assert not re.search(r"-?\d{1,3}\.\d{4,},\s*-?\d{1,3}\.\d{4,}", text), "looks like coordinates"
    assert "observation_id" not in text and "\\photos\\" not in text and "/photos/" not in text


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_a_decided_status_names_its_decision(e):
    if e.meta["status"] in ("adopted", "dropped"):
        assert e.meta["decision"] != "pending"


def write(tmp_path, name, text):
    d = tmp_path / "exp"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")
    return d


GOOD = """---
title: T
slug: good-one
date: 2026-10-09
status: done
reproducibility: exploratory-pre-freeze
question: Q?
headline: H
verdict: V
decision: pending
---
## Question
"""


@pytest.mark.parametrize("e", ENTRIES, ids=IDS)
def test_every_entry_says_how_reproducible_it_is_and_pre_freeze_ones_are_provisional(e):
    repro = e.meta["reproducibility"]
    assert repro == experiments.EXPLORATORY or experiments.REPRODUCED.match(repro)
    decided = not str(e.meta["decision"]).strip().lower().startswith("pending")
    if repro == experiments.EXPLORATORY and decided:
        assert "Provisional" in e.body and "dataset release v1" in e.body


REPRODUCED = GOOD.replace("reproducibility: exploratory-pre-freeze",
                          "reproducibility: reproduced-on-v1\ndataset_release: v1\n"
                          "reference_hash: abc123\ncode_commit: def4567\n"
                          "reproduce_command: mv compare --release v1")


def test_an_entry_reproduced_on_a_release_must_say_how_to_rebuild_it(tmp_path):
    assert experiments.load_all(write(tmp_path, "2026-10-09-good-one.md", REPRODUCED))
    for field in experiments.REPRO_FIELDS:
        bad = "\n".join(line for line in REPRODUCED.split("\n")
                        if not line.startswith(field + ":"))
        with pytest.raises(experiments.BadEntry, match=field):
            experiments.load_all(write(tmp_path, "2026-10-09-good-one.md", bad))
    with pytest.raises(experiments.BadEntry, match="dataset_release must be v1"):
        experiments.load_all(write(tmp_path, "2026-10-09-good-one.md",
                                   REPRODUCED.replace("dataset_release: v1", "dataset_release: v2")))


def test_after_the_freeze_an_exploratory_entry_is_refused(tmp_path):
    d = write(tmp_path, "2026-10-09-good-one.md", GOOD)
    assert experiments.load_all(d, freeze_date="2026-11-01")          # dated before: fine
    with pytest.raises(experiments.BadEntry, match="after the freeze"):
        experiments.load_all(d, freeze_date="2026-10-09")
    assert experiments.load_all(write(tmp_path, "2026-10-09-good-one.md", REPRODUCED),
                                freeze_date="2026-10-01")


def test_a_file_whose_name_slug_or_date_disagree_is_refused(tmp_path):
    d = write(tmp_path, "2026-10-08-good-one.md", GOOD)
    with pytest.raises(experiments.BadEntry, match="match the file name"):
        experiments.load_all(d)


@pytest.mark.parametrize("bad, why", [
    (GOOD.replace("status: done", "status: maybe"), "status must be"),
    (GOOD.replace("verdict: V\n", ""), "missing verdict"),
    (GOOD.replace("reproducibility: exploratory-pre-freeze\n", ""), "missing reproducibility"),
    (GOOD.replace("exploratory-pre-freeze", "sort-of"), "reproducibility is"),
    (GOOD.replace("---\n", "", 1), "front matter"),
    (GOOD.replace("decision: pending", "decision: pending\nrelated: Not A Slug"), "related"),
])
def test_an_entry_breaking_a_rule_is_refused(tmp_path, bad, why):
    d = write(tmp_path, "2026-10-09-good-one.md", bad)
    with pytest.raises(experiments.BadEntry, match=why):
        experiments.load_all(d)


def site(conn, tmp_path, mode):
    seed_two_species(conn, tmp_path)
    conn.commit()
    d = write(tmp_path, "2026-10-09-good-one.md", GOOD)
    signin = SigninConfig(mode=mode, origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET)
    app = create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                     backbone_loader=lambda n: Const(n), limits=open_limits(), signin=signin,
                     background=False, experiments_dir=d)
    return TestClient(app, follow_redirects=False)


@pytest.mark.parametrize("mode", ["identify", "all"])
def test_signed_out_visitors_never_get_the_registry_even_on_an_open_site(conn, tmp_path, mode):
    c = site(conn, tmp_path, mode)
    assert c.get("/api/experiments").status_code == 401
    assert c.get("/api/experiments/good-one").status_code == 401


def test_members_read_the_list_and_each_entry_uncached(conn, tmp_path):
    c = site(conn, tmp_path, "identify")
    sign_in(c)
    r = c.get("/api/experiments")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    assert [x["slug"] for x in r.json()["experiments"]] == ["good-one"]
    one = c.get("/api/experiments/good-one").json()
    assert one["status"] == "done" and one["markdown"].startswith("## Question")
    assert c.get("/api/experiments/nope").status_code == 404
    assert c.get("/api/experiments/..%2Fsecrets").status_code == 404
