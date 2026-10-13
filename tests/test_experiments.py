"""The experiment registry: every entry follows the rules in its README, the site shows it to
signed-in members only (Steve, 2026-10-09), and it lives in the private research checkout,
never in this public repo (Steve, 2026-10-12). The entry checks run where that checkout is
present (MV_RESEARCH_DIR, or ../mycomap-vision-research); CI has none."""

import re

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import ISSUER, PUBLIC_KEY, SECRET, sign_in

from mycomap_vision import config, experiments
from mycomap_vision.api import create_app
from mycomap_vision.signin import SigninConfig

ENTRIES = experiments.load_all()
IDS = [e.slug for e in ENTRIES]
HAVE_REGISTRY = experiments.EXPERIMENTS_DIR.is_dir()


def test_the_registry_is_read_from_the_private_checkout_not_this_repo():
    assert experiments.EXPERIMENTS_DIR == config.RESEARCH_DIR / "experiments"
    assert not experiments.EXPERIMENTS_DIR.resolve().is_relative_to(config.REPO_ROOT.resolve())


@pytest.mark.skipif(not HAVE_REGISTRY, reason="no private research checkout here")
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


# --- v1: the freeze date, and pre-freeze entries re-run on a release --------------------------

RERUN = REPRODUCED.replace("slug: good-one", "slug: good-two").replace("date: 2026-10-09",
                                                                      "date: 2026-10-13")
HISTORY = GOOD.replace("decision: pending", "decision: pending\nreproduced_by: good-two")


def test_the_freeze_is_the_day_v1_was_cut():
    assert experiments.FREEZE_DATE == "2026-10-12"


def test_an_exploratory_entry_dated_on_the_cut_day_is_refused(tmp_path):
    d = write(tmp_path, "2026-10-12-good-one.md", GOOD.replace("2026-10-09", "2026-10-12"))
    with pytest.raises(experiments.BadEntry, match="after the freeze"):
        experiments.load_all(d)


def test_a_re_run_on_v1_turns_the_old_entry_into_method_history(tmp_path):
    write(tmp_path, "2026-10-09-good-one.md", HISTORY)
    d = write(tmp_path, "2026-10-13-good-two.md", RERUN)
    found = {e.slug: experiments.summary(e) for e in experiments.load_all(d)}
    assert found["good-one"]["reproduced_by"] == "good-two"
    assert found["good-one"]["reproduced_on"] == "v1"
    assert found["good-one"]["reproducibility"] == "exploratory-pre-freeze"   # its numbers stay old
    assert "reproduced_on" not in found["good-two"]


@pytest.mark.parametrize("files, why", [
    ({"2026-10-09-good-one.md": HISTORY}, "not an entry reproduced"),           # no such entry
    ({"2026-10-09-good-one.md": HISTORY,                                         # re-run not on v1
      "2026-10-08-good-two.md": GOOD.replace("good-one", "good-two").replace("2026-10-09",
                                                                             "2026-10-08")},
     "not an entry reproduced"),
    ({"2026-10-13-good-two.md": RERUN.replace("decision: pending",
                                              "decision: pending\nreproduced_by: good-two")},
     "belongs on the exploratory entry"),
    ({"2026-10-09-good-one.md": HISTORY.replace("reproduced_by: good-two",
                                                "reproduced_by: [good-two]"),
      "2026-10-13-good-two.md": RERUN}, "one slug"),
])
def test_a_wrong_reproduced_by_is_refused(tmp_path, files, why):
    for name, text in files.items():
        d = write(tmp_path, name, text)
    with pytest.raises(experiments.BadEntry, match=why):
        experiments.load_all(d)


def test_reproduced_on_is_never_taken_from_the_file(tmp_path):
    d = write(tmp_path, "2026-10-09-good-one.md",
              GOOD.replace("decision: pending", "decision: pending\nreproduced_on: v1"))
    (e,) = experiments.load_all(d)
    assert "reproduced_on" not in experiments.summary(e)


def test_members_see_which_entry_re_ran_an_old_one(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.commit()
    write(tmp_path, "2026-10-09-good-one.md", HISTORY)
    d = write(tmp_path, "2026-10-13-good-two.md", RERUN)
    signin = SigninConfig(mode="identify", origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET)
    c = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                              backbone_loader=lambda n: Const(n), limits=open_limits(),
                              signin=signin, background=False, experiments_dir=d),
                   follow_redirects=False)
    sign_in(c)
    listed = {x["slug"]: x for x in c.get("/api/experiments").json()["experiments"]}
    assert listed["good-one"]["reproduced_on"] == "v1"
    one = c.get("/api/experiments/good-one").json()
    assert (one["reproduced_by"], one["reproduced_on"]) == ("good-two", "v1")


# --- one bad entry never takes the list down ----------------------------------------------

def test_one_bad_entry_is_skipped_logged_and_counted_not_a_500(conn, tmp_path, caplog):
    seed_two_species(conn, tmp_path)
    conn.commit()
    write(tmp_path, "2026-10-09-good-one.md", GOOD)
    # Being written by another lane: exploratory and dated after the freeze.
    d = write(tmp_path, "2026-10-13-place-penalty.md",
              GOOD.replace("good-one", "place-penalty").replace("2026-10-09", "2026-10-13"))
    signin = SigninConfig(mode="identify", origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET)
    c = TestClient(create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                              backbone_loader=lambda n: Const(n), limits=open_limits(),
                              signin=signin, background=False, experiments_dir=d),
                   follow_redirects=False)
    sign_in(c)
    with caplog.at_level("WARNING", logger="mycomap_vision.api"):
        r = c.get("/api/experiments")
    assert r.status_code == 200
    assert [x["slug"] for x in r.json()["experiments"]] == ["good-one"]
    assert r.json()["skipped"] == 1
    assert "[experiments] bad entry 2026-10-13-place-penalty.md" in caplog.text
    assert c.get("/api/experiments/good-one").status_code == 200
    assert c.get("/api/experiments/place-penalty").status_code == 404


def test_load_checked_names_every_bad_file_and_keeps_the_rest(tmp_path):
    write(tmp_path, "2026-10-09-good-one.md", HISTORY)                 # its re-run is missing
    write(tmp_path, "2026-10-08-fine.md", GOOD.replace("good-one", "fine").replace("2026-10-09",
                                                                                   "2026-10-08"))
    d = write(tmp_path, "2026-10-09-broken.md", "no front matter")
    found, problems = experiments.load_checked(d)
    assert [e.slug for e in found] == ["fine"]
    assert sorted(n for n, _ in problems) == ["2026-10-09-broken.md", "2026-10-09-good-one.md"]
    with pytest.raises(experiments.BadEntry):                          # the strict loader
        experiments.load_all(d)


def test_experiments_check_exits_non_zero_on_a_bad_entry(tmp_path, capsys):
    from mycomap_vision import cli
    d = write(tmp_path, "2026-10-09-good-one.md", GOOD)
    assert cli.main(["experiments-check", str(d)]) == 0
    assert "1 entries ok, 0 bad" in capsys.readouterr().out
    write(tmp_path, "2026-10-13-late.md", GOOD.replace("good-one", "late").replace("2026-10-09",
                                                                                   "2026-10-13"))
    assert cli.main(["experiments-check", str(d)]) == 1
    out = capsys.readouterr().out
    assert "BAD 2026-10-13-late.md" in out and "after the freeze" in out
