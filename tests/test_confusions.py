"""Often confused with (confusions.py): mycomap.org's "misidentified as" table read nightly
and shown on species candidates that prompt (docs/api.md §4)."""

import gzip
import json
import tracemalloc

import pytest
import requests

from test_nightly import RID, FakeOutside, QUIET, box, night, seed_rows  # noqa: F401

from mycomap_vision import confusions, manifest, names, nightly, nightly_notes
from mycomap_vision.confusions import Table

COLUMNS = ["name", "called", "wrong", "broaderSpecies", "people", "confidence", "prompts",
           "dnaSaidInstead[name, records, people, w|b]", "aliases"]


def row(name, called, wrong, prompts, instead=(), aliases=()):
    """One /all row as .org sends it: [name, called, wrong, broaderSpecies, people,
    confidence, prompts, dnaSaidInstead[[name, records, people, w|b]], aliases]."""
    return [name, called, wrong, 0, called, 0.9 if prompts else 0.1, prompts,
            [list(a) for a in instead], list(aliases)]


def answer(*rows, build_id=7, built_at="2026-10-13T11:00:00.000Z"):
    return {"build": {"id": build_id, "builtAt": built_at, "records": 163427,
                      "identifications": 737996, "names": len(rows), "prompting": None},
            "rule": {"wrongShare": 0.25, "confidence": 0.8}, "columns": COLUMNS,
            "aliasesNote": "…", "names": [list(r) for r in rows]}


GYMNOPILUS = row("Gymnopilus luteus", 86, 29, True, [
    ["Gymnopilus sp. 'luteus-IN01'", 11, 9, "b"],          # a narrower DNA type: not listed
    ["Gymnopilus subspectabilis", 8, 6, "w"],
    ["Gymnopilus voitkii", 6, 5, "w"],
    ["Gymnopilus sapineus", 4, 4, "w"],
    ["Gymnopilus penetrans", 3, 3, "w"],
])


def table(*rows):
    return Table.from_answer(answer(*rows))


# --- the field and the prompting rule ---------------------------------------------------

def test_a_name_that_prompts_lists_the_top_three_species_dna_said_by_finds():
    cw = table(GYMNOPILUS).confused_with("Gymnopilus luteus")
    assert cw == {"names": [{"name": "Gymnopilus subspectabilis", "finds": 8},
                            {"name": "Gymnopilus voitkii", "finds": 6},
                            {"name": "Gymnopilus sapineus", "finds": 4}],
                  "called": 86, "wrong": 29,
                  "note": "Often confused with G. subspectabilis · 9% of DNA-checked finds"}


def test_the_field_follows_orgs_prompt_flag_not_a_rule_of_visions_own():
    t = table(row("Amanita muscaria", 40, 39, False, [["Amanita persicina", 30, 9, "w"]]),
              row("Amanita flavoconia", 2, 1, True, [["Amanita frostiana", 1, 1, "w"]]))
    assert t.confused_with("Amanita muscaria") is None       # mostly wrong, but .org says no
    assert t.confused_with("Amanita flavoconia")["called"] == 2
    assert [r.name for r in t.rows] == ["Amanita flavoconia"]   # only prompting rows are kept


def test_a_name_whose_only_other_answers_are_narrower_dna_types_gets_nothing():
    t = table(row("Leotia lubrica", 20, 9, True, [["Leotia sp. 'lubrica-CA01'", 9, 4, "b"]]))
    assert t.confused_with("Leotia lubrica") is None


def test_an_answer_that_is_the_candidate_in_another_spelling_is_never_listed():
    t = table(row("Cortinarius sp. 'IN7'", 5, 4, True,
                  [["Cortinarius sp. 'IN07'", 3, 2, "w"], ["Cortinarius sp. 'IN09'", 1, 1, "w"]]))
    cw = t.confused_with("Cortinarius sp. 'IN07'")
    assert [n["name"] for n in cw["names"]] == ["Cortinarius sp. 'IN09'"]
    assert cw["note"] == "Often confused with C. sp. 'IN09' · 1 of 5 DNA-checked finds"


def test_an_unknown_name_and_no_table_get_nothing():
    assert table(GYMNOPILUS).confused_with("Gymnopilus junonius") is None
    result = {"ranks": {"species": [{"name": "Gymnopilus luteus"}]}}
    assert confusions.annotate(result, None) == 0
    assert "confused_with" not in result["ranks"]["species"][0]


# --- matching ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, key", [
    # services/misidentifications.ts nameKey on mycomap.org, rule for rule.
    ("Gymnopilus  Luteus ", "gymnopilus luteus"),
    ("Russula sp. ‘IN05’", "russula sp. 'in05'"),
    ("Russula sp. “IN05”", "russula sp. 'in05'"),
    ('Russula sp. "IN05"', "russula sp. 'in05'"),
    ("Hygrocybe acutoconica var. microspora", "hygrocybe acutoconica microspora"),
    ("Amanita muscaria subsp. flavivolvata", "amanita muscaria flavivolvata"),
    ("Amanita muscaria ssp. flavivolvata", "amanita muscaria flavivolvata"),
    ("Clitocybe nebularis f. alba", "clitocybe nebularis alba"),
    ("Clitocybe nebularis fo. alba", "clitocybe nebularis alba"),
    ("Clitocybe nebularis forma alba", "clitocybe nebularis forma alba"),   # no dot: kept
    ("Mycena sp. 'IN7'", "mycena sp. 'in7'"),                    # .org pads no codes
    (None, ""),
])
def test_orgs_name_key_is_followed_rule_for_rule(name, key):
    assert confusions.org_key(name) == key


@pytest.mark.parametrize("org_name, candidate", [
    ("Hygrocybe acutoconica microspora", "Hygrocybe acutoconica var. microspora"),
    ("Hygrocybe acutoconica var. microspora", "Hygrocybe acutoconica microspora"),
    ("Amanita muscaria subsp. flavivolvata", "Amanita muscaria flavivolvata"),
    ("Amanita muscaria ssp. flavivolvata", "Amanita muscaria subsp. flavivolvata"),
    ("Clitocybe nebularis f. alba", "Clitocybe nebularis alba"),
    ("Russula sp. ‘IN05’", "Russula sp. 'IN05'"),             # curly quotes
    ('Russula sp. "IN05"', "Russula sp. 'IN05'"),             # double quotes
    ("Gymnopilus  luteus", "gymnopilus luteus"),              # spacing and case
    ("Mycena sp. 'IN7'", "Mycena sp. 'IN07'"),                # Vision's spelling key
    ("Mycena sp. IN07", "Mycena sp. 'IN07'"),                 # an unquoted code
])
def test_a_candidate_finds_its_row_through_orgs_key_or_visions_spelling(org_name, candidate):
    t = table(row(org_name, 4, 3, True, [["Mycena other", 3, 3, "w"]]))
    assert t.confused_with(candidate) is not None, (org_name, candidate)


def test_a_synonym_in_the_aliases_column_matches():
    t = table(row("Gymnopilus junonius", 120, 50, True, [["Gymnopilus ventricosus", 30, 9, "w"]],
                  aliases=["gymnopilus spectabilis", "gymnopilus junonius"]))
    cw = t.confused_with("Gymnopilus spectabilis")
    assert cw["called"] == 120
    assert cw["note"] == "Often confused with G. ventricosus · 25% of DNA-checked finds"


def test_a_name_with_its_own_row_that_does_not_prompt_never_borrows_a_synonyms():
    t = table(row("Gymnopilus junonius", 120, 50, True, [["Gymnopilus ventricosus", 30, 9, "w"]],
                  aliases=["gymnopilus spectabilis"]),
              row("Gymnopilus spectabilis", 40, 2, False, [["Gymnopilus ventricosus", 2, 2, "w"]],
                  aliases=["gymnopilus junonius"]))
    assert t.confused_with("Gymnopilus spectabilis") is None
    assert t.confused_with("Gymnopilus junonius")["called"] == 120


def test_orgs_own_aliases_come_before_visions_looser_spelling():
    # .org lists "mycena sp. 'in07'" as another name of Mycena alpha; Vision's spelling key
    # would also fold it into Mycena sp. 'IN7' (IN07 = IN7). .org's exact alias wins.
    t = table(row("Mycena sp. 'IN7'", 30, 1, False),
              row("Mycena alpha", 8, 6, True, [["Mycena beta", 5, 5, "w"]],
                  aliases=["mycena sp. 'in07'"]))
    assert t.confused_with("Mycena sp. 'IN07'")["called"] == 8
    assert t.confused_with("Mycena sp. 'IN7'") is None


def test_when_two_rows_share_an_alias_the_one_called_most_wins():
    t = table(row("Suillus a", 10, 5, True, [["Suillus x", 5, 5, "w"]], aliases=["suillus old"]),
              row("Suillus b", 50, 20, True, [["Suillus y", 9, 9, "w"]], aliases=["suillus old"]))
    assert t.confused_with("Suillus old")["called"] == 50


# --- wording ----------------------------------------------------------------------------

@pytest.mark.parametrize("candidate, called, top, finds, note", [
    ("Amanita flavoconia", 1, "Amanita frostiana", 1,
     "Often confused with A. frostiana · 1 of 1 DNA-checked find"),
    ("Amanita flavoconia", 9, "Amanita frostiana", 3,
     "Often confused with A. frostiana · 3 of 9 DNA-checked finds"),
    ("Amanita flavoconia", 10, "Amanita frostiana", 3,
     "Often confused with A. frostiana · 30% of DNA-checked finds"),
    ("Gymnopilus luteus", 86, "Gymnopilus subspectabilis", 8,
     "Often confused with G. subspectabilis · 9% of DNA-checked finds"),
    ("Amanita flavoconia", 200, "Amanita frostiana", 1,           # 0.5% rounds up, as .org's
     "Often confused with A. frostiana · 1% of DNA-checked finds"),
    ("Amanita flavoconia", 300, "Amanita frostiana", 1,
     "Often confused with A. frostiana · under 1% of DNA-checked finds"),
    ("Amanita flavoconia", 12, "Russula emetica", 5,              # another genus: in full
     "Often confused with Russula emetica · 42% of DNA-checked finds"),
    ("Amanita flavoconia", 4, "Amanitopsis vaginata", 2,          # a genus that starts alike
     "Often confused with Amanitopsis vaginata · 2 of 4 DNA-checked finds"),
])
def test_the_note_gives_counts_under_ten_finds_and_a_share_from_ten(candidate, called, top,
                                                                    finds, note):
    assert confusions.note_text(candidate, called, top, finds) == note


# --- where the field goes ---------------------------------------------------------------

def species(name):
    return {"name": name, "score": 0.5, "confidence": 0.4, "reference_records": 3, "place": None}


def test_every_species_list_gets_the_field_and_no_other_rank_does():
    result = {"ranks": {"species": [species("Gymnopilus luteus"), species("Gymnopilus junonius")],
                        "genus": [species("Gymnopilus luteus")]},
              "likely": {"species": {"names": [species("Gymnopilus luteus")]},
                         "genus": {"names": [species("Gymnopilus luteus")]}},
              "per_photo": [{"ranks": {"species": [species("Gymnopilus luteus")],
                                       "genus": [species("Gymnopilus luteus")]}}]}
    assert confusions.annotate(result, table(GYMNOPILUS)) == 3
    assert result["ranks"]["species"][0]["confused_with"]["called"] == 86
    assert result["ranks"]["species"][0]["place"] is None          # place keys untouched
    assert "confused_with" not in result["ranks"]["species"][1]
    assert "confused_with" in result["likely"]["species"]["names"][0]
    assert "confused_with" in result["per_photo"][0]["ranks"]["species"][0]
    for where in (result["ranks"]["genus"][0], result["likely"]["genus"]["names"][0],
                  result["per_photo"][0]["ranks"]["genus"][0]):
        assert "confused_with" not in where
    # Each candidate has its own copy: changing one answer can't change another.
    result["ranks"]["species"][0]["confused_with"]["names"][0]["finds"] = 0
    assert result["likely"]["species"]["names"][0]["confused_with"]["names"][0]["finds"] == 8


# --- reading .org -----------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def json(self):
        if isinstance(self.body, str):
            raise ValueError("not JSON")
        return self.body


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_the_read_is_public_and_identifies_vision():
    s = FakeSession(FakeResponse(200, answer(GYMNOPILUS)))
    got = confusions.fetch_all("https://org.example/", session=s, pauses=(), **QUIET)
    assert got["names"][0][0] == "Gymnopilus luteus"
    [call] = s.calls
    assert call["url"] == "https://org.example/api/misidentifications/all"
    assert call["headers"]["User-Agent"] == confusions.config.USER_AGENT
    assert "Authorization" not in call["headers"]           # public: no key is ever sent
    assert call["timeout"] == confusions.TIMEOUT


def test_a_network_failure_or_5xx_is_asked_again_but_not_ready_is_not():
    slept = []
    s = FakeSession(requests.ConnectionError("down"), FakeResponse(502, "<html>"),
                    FakeResponse(200, answer(GYMNOPILUS)))
    confusions.fetch_all("https://o", session=s, pauses=(1, 2), sleep=slept.append, **QUIET)
    assert slept == [1, 2] and len(s.calls) == 3
    s = FakeSession(FakeResponse(503, {"code": "not_ready"}))
    with pytest.raises(confusions.Unavailable, match="not_ready"):
        confusions.fetch_all("https://o", session=s, pauses=(1, 2), sleep=slept.append, **QUIET)
    assert len(s.calls) == 1
    s = FakeSession(FakeResponse(404, {"code": "not_found"}))
    with pytest.raises(confusions.Unavailable, match="404"):
        confusions.fetch_all("https://o", session=s, pauses=(1,), sleep=slept.append, **QUIET)
    assert len(s.calls) == 1


@pytest.mark.parametrize("payload, why", [
    ("<html>", "not JSON"),
    ({**answer(GYMNOPILUS), "names": []}, "no names"),
    ({**answer(GYMNOPILUS), "columns": ["name", "wrong", "called"]}, "columns"),
    (answer(GYMNOPILUS, ["Bad", 3, 4, 0, 1, 0.9, True, [], []]), "row 2"),     # wrong > called
    (answer(GYMNOPILUS, ["Bad", 3, 1, 0, 1, 0.9, "yes", [], []]), "row 2"),    # prompts not bool
    (answer(GYMNOPILUS, ["Bad", 3, 1, 0, 1, 0.9, True, [["X y", 1, 1, "q"]], []]), "row 2"),
])
def test_a_malformed_or_empty_answer_is_refused_whole(payload, why):
    s = FakeSession(FakeResponse(200, payload))
    with pytest.raises(confusions.Unavailable, match=why):
        confusions.fetch_all("https://o", session=s, pauses=(), **QUIET)


def test_an_answer_without_the_aliases_column_is_still_read():
    rows = [GYMNOPILUS[:8]]
    payload = {**answer(), "columns": COLUMNS[:8], "names": rows}
    assert Table.from_answer(payload).confused_with("Gymnopilus luteus")["called"] == 86


# --- the cache --------------------------------------------------------------------------

def fails(error):
    def fetch():
        raise error
    return fetch


def test_a_failed_read_keeps_the_previous_cache_and_counts_failures_in_a_row(tmp_path):
    cache = tmp_path / confusions.CACHE
    ok = confusions.refresh(cache, lambda: answer(GYMNOPILUS, build_id=7), **QUIET)
    assert ok["ok"] and ok["build_id"] == 7 and ok["prompting"] == 1 and ok["names"] == 1
    held = cache.read_bytes()
    for n, failure in enumerate([
            fails(confusions.Unavailable("mycomap.org has not built the table yet (503 not_ready)")),
            fails(requests.ConnectionError("down")),
            lambda: {**answer(), "names": []},                        # an empty answer
            lambda: answer(["Bad", 1, 2, 0, 1, 0.9, True, [], []])], 1):  # a malformed one
        report = confusions.refresh(cache, failure, **QUIET)
        assert report["ok"] is False and report["failures_in_a_row"] == n
        assert report["kept_previous"]["build_id"] == 7
        assert cache.read_bytes() == held                         # untouched
        assert confusions.load(cache).confused_with("Gymnopilus luteus")["called"] == 86
    assert confusions.health(cache)["failures_in_a_row"] == 4
    again = confusions.refresh(cache, lambda: answer(GYMNOPILUS, build_id=8), **QUIET)
    assert again["build_id"] == 8 and confusions.health(cache)["failures_in_a_row"] == 0


def test_a_first_read_that_fails_leaves_no_table(tmp_path):
    cache = tmp_path / confusions.CACHE
    report = confusions.refresh(cache, fails(requests.Timeout()), **QUIET)
    assert report["ok"] is False and report["kept_previous"] is None
    assert confusions.load(cache) is None
    assert confusions.health(cache) == {"cached": False, "built_at": None, "fetched_at": None,
                                        "prompting": None, "failures_in_a_row": 1}


def test_the_report_counts_visions_species_names_that_prompt(tmp_path):
    report = confusions.refresh(tmp_path / confusions.CACHE, lambda: answer(GYMNOPILUS),
                                labels={"Gymnopilus luteus", "Gymnopilus junonius",
                                        "Gymnopilus"}, **QUIET)
    assert report["vision_species"] == 2 and report["vision_species_prompting"] == 1


def test_the_server_picks_up_a_new_cache_and_keeps_its_table_when_one_is_unreadable(tmp_path):
    cache = tmp_path / confusions.CACHE
    now = [0.0]
    served = confusions.Served(cache, check_every=60, clock=lambda: now[0],
                                note=lambda s: None)
    assert served.table() is None                                    # no table yet
    confusions.refresh(cache, lambda: answer(GYMNOPILUS, build_id=7), **QUIET)
    assert served.table() is None                                    # not looked at again yet
    now[0] = 61
    assert served.table().build["id"] == 7
    cache.write_bytes(gzip.compress(b"{broken"))
    now[0] = 130
    assert served.table().build["id"] == 7                           # the old one serves on
    confusions.refresh(cache, lambda: answer(GYMNOPILUS, build_id=8), **QUIET)
    assert served.reload().build["id"] == 8


# --- the nightly ------------------------------------------------------------------------

def test_the_nightly_reads_the_table_as_its_own_step_into_the_layer(box, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MV_ORG_BASE_URL", "https://org.example")
    asked = []

    def fake_fetch_all(base_url, log=print, **_):
        asked.append(base_url)
        return answer(row("A x", 12, 7, True, [["A z", 5, 3, "w"]]), row("B y", 9, 0, False))
    monkeypatch.setattr(confusions, "fetch_all", fake_fetch_all)
    layer = nightly.prepare(box, **QUIET)
    outside = FakeOutside({})
    steps = nightly.Steps(fetch_inat=outside.fetch_inat, taxonomy=outside.taxonomy,
                          download=outside.download)               # the real step
    conn = manifest.connect(layer.manifest)
    try:
        report = nightly.run_once(conn, layer, lambda: (seed_rows(), "2026-10-13T07:00:00Z"),
                                  {}, steps=steps, **QUIET)
    finally:
        conn.close()
    assert asked == ["https://org.example"]
    cw = report["confused_with"]
    assert cw["ok"] and cw["names"] == 2 and cw["prompting"] == 1 and cw["build_id"] == 7
    assert cw["vision_species"] == 2 and cw["vision_species_prompting"] == 1
    assert (layer.root / confusions.CACHE).is_file()
    saved = json.loads(sqlite_report(layer))
    assert saved["confused_with"]["prompting"] == 1              # in nightly_runs too


def sqlite_report(layer):
    conn = manifest.connect(layer.manifest)
    try:
        return conn.execute("select report from nightly_runs order by id desc").fetchone()[0]
    finally:
        conn.close()


def test_a_refused_green_list_still_reads_the_table_and_a_failed_read_is_reported(box):  # noqa: F811
    layer = nightly.prepare(box, **QUIET)
    calls = []

    def misid(cache, labels=None, log=print):
        calls.append(cache)
        raise RuntimeError("boom")
    outside = FakeOutside({})
    steps = nightly.Steps(fetch_inat=outside.fetch_inat, taxonomy=outside.taxonomy,
                          download=outside.download, misidentifications=misid)

    def refused():
        raise nightly.NightlyRefused("mycomap.org answered 502; nothing was changed")
    conn = manifest.connect(layer.manifest)
    try:
        report = nightly.run_once(conn, layer, refused, {}, steps=steps, **QUIET)
    finally:
        conn.close()
    assert calls == [layer.root / confusions.CACHE]
    assert report["ok"] is False and report["confused_with"] == {"error": "RuntimeError: boom"}


def test_the_nightly_without_an_org_address_skips_the_read(box):  # noqa: F811
    layer = nightly.prepare(box, **QUIET)
    outside = FakeOutside({})
    steps = nightly.Steps(fetch_inat=outside.fetch_inat, taxonomy=outside.taxonomy,
                          download=outside.download)
    conn = manifest.connect(layer.manifest)
    try:
        report = nightly.run_once(conn, layer, lambda: (seed_rows(), "2026-10-13T07:00:00Z"),
                                  {}, steps=steps, **QUIET)
    finally:
        conn.close()
    assert report["confused_with"] == {"skipped": "MV_ORG_BASE_URL not set"}


def test_health_notes_warn_after_two_failed_reads_in_a_row():
    base = {"enabled": True, "state": "never", "last_ok_at": None, "last_run_at": None}
    ok = dict(cached=True, built_at="2026-10-13T11:00:00Z", fetched_at="x", prompting=3108,
              failures_in_a_row=1)
    lines = nightly_notes.notes({**base, "confused_with": ok})
    assert lines[-1] == ("ok", "often confused with: table built 2026-10-13T11:00:00Z, "
                               "3,108 names prompt")
    lines = nightly_notes.notes({**base, "confused_with": {**ok, "failures_in_a_row": 2}})
    assert lines[-1][0] == "warn" and "last 2 reads" in lines[-1][1]


# --- memory -----------------------------------------------------------------------------

def full_size_answer():
    """As large as the 13 Oct build: 17,160 names, 3,108 prompting, six answers each,
    two aliases each."""
    rows = []
    for i in range(17160):
        prompts = i % 5 == 0 and len([r for r in rows if r[6]]) < 3108
        rows.append(row(f"Genus{i % 1600} epithet{i}", 5 + i % 90, 2 + i % 3, prompts,
                        [[f"Genus{i % 1600} other{i + k}", 6 - k, 3, "w" if k % 3 else "b"]
                         for k in range(6)],
                        aliases=[f"genus{i % 1600} epithet{i}", f"genus{i % 1600} syn{i}"]))
    return answer(*rows)


def test_the_parsed_table_stays_compact():
    payload = full_size_answer()
    raw = len(json.dumps(payload, separators=(",", ":")))
    parsed_before = names._parse.cache_info().currsize
    tracemalloc.start()
    t = Table.from_answer(payload)
    held, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(t.rows) == 3108
    assert raw > 2_000_000                     # at least as big as .org's answer
    assert held < 8 * 2**20, f"{held / 2**20:.1f} MB held"     # 4.4 MB on 2026-10-13
    # names.py's parse cache keeps every name for good: .org's names must not go there.
    assert names._parse.cache_info().currsize == parsed_before


# --- for the API suites -----------------------------------------------------------------

# The two-species test set (test_models_and_scoreboard.seed_two_species): "A x" prompts,
# "B y" is known to .org and doesn't.
A_X = row("A x", 12, 7, True, [["A z", 5, 3, "w"], ["A x-CA01", 4, 2, "b"], ["B y", 2, 2, "w"],
                               ["A q", 1, 1, "w"], ["A r", 1, 1, "w"]])
B_Y = row("B y", 30, 1, False, [["A x", 1, 1, "w"]])
A_X_FIELD = {"names": [{"name": "A z", "finds": 5}, {"name": "B y", "finds": 2},
                       {"name": "A q", "finds": 1}],
             "called": 12, "wrong": 7, "note": "Often confused with A. z · 42% of DNA-checked finds"}


def write_table(beside_manifest, *rows):
    """The nightly's cache for a server whose manifest is in `beside_manifest`."""
    report = confusions.refresh(beside_manifest / confusions.CACHE, lambda: answer(*rows),
                                **QUIET)
    assert report["ok"], report


def species_candidates(body: dict) -> list[dict]:
    """Every species name in an answer: ranks, the likely set, each photo's ranks."""
    out = list(body["ranks"]["species"])
    out += ((body.get("likely") or {}).get("species") or {}).get("names") or []
    for p in body.get("per_photo") or []:
        out += p["ranks"]["species"]
    return out


def all_candidates(body: dict) -> list[dict]:
    out = [c for cands in body["ranks"].values() for c in cands]
    out += [n for s in (body.get("likely") or {}).values() for n in s["names"]]
    for p in body.get("per_photo") or []:
        out += [c for cands in p["ranks"].values() for c in cands]
    return out
