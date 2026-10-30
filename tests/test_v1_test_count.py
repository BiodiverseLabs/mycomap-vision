"""The v1-test counter (v1_test_count.py, `mv v1-test-count`): records first green at or after
the seal, not in the base release, that the release builder would include, by green day; in
scope only for the scope's countries; vision_assisted reported per day; the manifest only
read."""

import hashlib
import json
import sqlite3
import sys

import pytest

from mycomap_vision import cli, manifest, v1_test_count
from test_dataset_release import Store, add_photo, add_record, jpeg, migrate

SEAL = "2026-10-12T13:17:51+00:00"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


class Base:
    """A base release as the counter uses it: its id, candidate records and photo bytes."""

    def __init__(self, keys=(), shas=()):
        self.id, self._keys, self._shas = "v1", list(keys), set(shas)

    def record_keys(self):
        return self._keys

    def photo_shas(self):
        return self._shas


def green(conn, key, *, vdate, seen=None, first_green=None, country="US", assisted=0):
    conn.execute("update records set validated_on = ?, first_seen_at = ?, first_green_at = ?, "
                 "country = ?, vision_assisted = ? where observation_id = ?",
                 (vdate, seen or f"{vdate}T06:00:00+00:00", first_green, country, assisted, key))


@pytest.fixture
def world(tmp_path):
    path = tmp_path / "data" / "manifest.sqlite"
    conn = manifest.connect(path)
    migrate(conn)
    store = Store(tmp_path / "store")
    pid = iter(range(100, 1000))

    def rec(key, name="Russula emetica", conflict=0, copy=True, seed=None, photo="done", **g):
        add_record(conn, key, name, conflict=conflict)
        n = next(pid)
        add_photo(conn, store, key, n, copy=copy, seed=seed)
        conn.execute("update photos set status = ? where photo_id = ?", (photo, n))
        green(conn, key, **g)

    rec("20", vdate="2026-10-20")                                         # US, in scope
    rec("21", vdate="2026-10-20", first_green="2026-10-19T22:00:00.000Z",  # CA, its moment
        country="CA", assisted=1)                                         # the day before
    rec("22", vdate="2026-10-21", country="MX")                           # outside the scope
    rec("23", vdate="2026-10-21", country=None)                           # no country: outside
    rec("24", vdate="2026-10-01")                                         # green before the seal
    rec("25", vdate="2026-10-21", first_green="2026-10-11T09:00:00Z")     # moment before it
    rec("26", vdate="2026-10-22", conflict=1, assisted=1,                 # label conflict
        first_green="2026-10-22T10:00:00Z")
    rec("27", vdate="2026-10-22")                                         # in the base already
    rec("28", vdate="2026-10-22", seed=999)                               # its photo is a base photo
    rec("29", vdate="2026-10-22", country="PR", copy=False)               # downloaded, no original kept
    rec("31", vdate="2026-10-22", copy=False, photo="pending")            # never downloaded
    rec("30", vdate="2026-10-23", country=" pr")                          # read as PR
    add_photo(conn, store, "30", next(pid))                               # 30 has two photos
    conn.commit()
    base = Base(keys=["27", "1"], shas=[hashlib.sha256(jpeg(999)).hexdigest()])
    yield {"conn": conn, "path": path, "base": base}
    conn.close()


def test_only_records_first_green_after_the_seal_and_not_in_the_base_count(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    assert rep["records_first_green_after_seal"] == 10     # 24 and 25 were green before it
    assert rep["already_in_base"] == 1                     # 27
    t = rep["total"]
    assert t["candidates"] == 9
    assert (t["in_scope"], t["outside_scope"], t["excluded"]) == (4, 2, 3)
    assert rep["outside_scope_by_country"] == {"MX": 1, "unknown": 1}
    assert rep["excluded_by_reason"] == {"label_conflict": 1, "no_photos": 2}
    assert rep["photos_identical_to_base_dropped"] == 1


def test_a_downloaded_photo_counts_without_its_original_and_is_reported_apart(world):
    # 29's photo was downloaded (status done) but no original is kept here, as on the box:
    # it counts, and is counted apart; 31's photo was never downloaded: no photo
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    assert rep["total"]["originals_not_fetched"] == 1
    by_day = {r["day"]: r for r in rep["days"]}
    assert (by_day["2026-10-22"]["in_scope"], by_day["2026-10-22"]["originals_not_fetched"],
            by_day["2026-10-22"]["excluded"]) == (1, 1, 3)   # 26, 28, 31
    assert "originals not fetched" in v1_test_count.format_count(rep)


def test_strict_originals_applies_the_cuts_rule(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"],
                              strict_originals=True)
    t = rep["total"]
    assert (t["in_scope"], t["excluded"], t["originals_not_fetched"]) == (3, 4, 0)
    assert rep["excluded_by_reason"]["no_photos"] == 3
    assert rep["originals"].startswith("required")


def test_a_build_still_needs_the_original(world):
    from mycomap_vision import dataset_release as dr
    built = dr.collect(world["conn"], only_keys={"29"})
    assert built.records[0]["reason"] == "no_photos"
    assert built.photos[0]["reason"] == "no_original"


def test_a_find_is_counted_on_the_day_it_first_turned_green(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    by_day = {r["day"]: r for r in rep["days"]}
    assert by_day["2026-10-19"]["in_scope"] == 1           # 21: .org's moment, not its day
    assert by_day["2026-10-20"]["in_scope"] == 1           # 20
    assert by_day["2026-10-21"]["outside_scope"] == 2
    assert by_day["2026-10-23"]["in_scope"] == 1           # 30, " pr" read as PR
    assert [r["in_scope_cumulative"] for r in rep["days"]] == [1, 2, 2, 3, 4]


def test_the_scope_is_the_countries_given(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"],
                              countries=("US", "CA", "PR", "MX"))
    assert rep["total"]["in_scope"] == 5 and rep["outside_scope_by_country"] == {"unknown": 1}


def test_vision_fields_are_counted_over_every_post_seal_candidate(world):
    # 26 is excluded (label conflict) but its feed fields still count among the candidates
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    by_day = {r["day"]: r for r in rep["days"]}
    assert (by_day["2026-10-19"]["vision_assisted_1"], by_day["2026-10-19"]["vision_assisted_0"],
            by_day["2026-10-19"]["first_green_at_given"]) == (1, 0, 1)
    assert (by_day["2026-10-22"]["vision_assisted_1"], by_day["2026-10-22"]["first_green_at_given"],
            by_day["2026-10-22"]["in_scope_vision_assisted"]) == (1, 1, 0)
    assert rep["vision_assisted_column"] == "present"
    t = rep["total"]
    assert (t["vision_assisted_1"], t["vision_assisted_0"], t["first_green_at_given"]) == (2, 7, 2)
    assert t["in_scope_vision_assisted"] == 1


def test_a_manifest_without_the_column_says_not_populated(world):
    world["conn"].execute("alter table records drop column vision_assisted")
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    assert rep["vision_assisted_column"] == "not populated"
    assert rep["total"]["vision_assisted_1"] == rep["total"]["vision_assisted_0"] == 0
    assert "not populated" in v1_test_count.format_count(rep)


def test_a_review_list_leaves_its_records_out(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"],
                              review_records={"20": "review:people"})
    assert rep["total"]["in_scope"] == 3
    assert rep["excluded_by_reason"]["review:people"] == 1
    assert "review lists" not in rep["not_applied"]


def test_with_no_base_the_seal_rule_alone_counts(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=None)
    assert rep["already_in_base"] == 0 and rep["base"] is None
    assert rep["total"]["in_scope"] == 6                   # 27 and 28 come back
    assert "the base release (seal rule only)" in rep["not_applied"]


def test_the_command_only_reads_the_manifest(world, capsys):
    world["conn"].execute("pragma journal_mode=delete")
    world["conn"].close()
    before = hashlib.sha256(world["path"].read_bytes()).hexdigest()
    assert cli.main(["v1-test-count", "--manifest", str(world["path"]), "--no-base",
                     "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["total"]["in_scope"] == 6 and rep["seal"] == SEAL
    assert cli.main(["v1-test-count", "--manifest", str(world["path"]), "--no-base",
                     "--json", "--strict-originals"]) == 0
    assert json.loads(capsys.readouterr().out)["total"]["in_scope"] == 5
    assert hashlib.sha256(world["path"].read_bytes()).hexdigest() == before
    assert not world["path"].with_name("manifest.sqlite-wal").exists()
    ro = v1_test_count.open_manifest(world["path"])
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute("insert into records (observation_id) values ('x')")
    ro.close()
    world["conn"] = sqlite3.connect(":memory:")            # the fixture closes this one


def test_the_help_says_to_run_it_on_the_laptop_against_a_copy(capsys):
    with pytest.raises(SystemExit):
        cli.main(["v1-test-count", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Run it on the laptop" in text and "copy of the box's live manifest" in text


def test_the_text_report_has_one_line_per_day_and_the_totals(world):
    text = v1_test_count.format_count(v1_test_count.count(world["conn"], seal=SEAL,
                                                          base=world["base"]))
    assert "2026-10-19" in text and "total in scope 4" in text
    assert "MX 1" in text and "crosspost_twin" in text


def test_gif_photos_and_finds_whose_photos_are_all_gifs_are_counted(world):
    c = world["conn"]
    c.execute("update photos set source_url = 'https://example.org/photos/1/original.gif?x=1' "
              "where photo_id = (select photo_id from observation_photos where "
              "observation_id = '20')")
    c.execute("update photos set source_url = 'https://example.org/photos/2/original.gif' "
              "where photo_id = (select min(photo_id) from observation_photos where "
              "observation_id = '30')")                       # one of its two photos
    rep = v1_test_count.count(c, seal=SEAL, base=world["base"])
    assert (rep["total"]["gif_photos"], rep["total"]["all_gif_finds"]) == (2, 1)
    by_day = {r["day"]: r for r in rep["days"]}
    assert by_day["2026-10-20"]["all_gif_finds"] == 1
    assert "all GIFs" in v1_test_count.format_count(rep)
    assert v1_test_count.is_gif({"original_path": "photos/large/1/9.GIF"})
    assert not v1_test_count.is_gif({"source_url": "https://x/gif/1.jpg"})
