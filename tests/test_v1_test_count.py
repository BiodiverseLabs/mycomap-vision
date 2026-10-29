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

    def rec(key, name="Russula emetica", conflict=0, copy=True, seed=None, **g):
        add_record(conn, key, name, conflict=conflict)
        add_photo(conn, store, key, next(pid), copy=copy, seed=seed)
        green(conn, key, **g)

    rec("20", vdate="2026-10-20")                                         # US, in scope
    rec("21", vdate="2026-10-20", first_green="2026-10-19T22:00:00.000Z",  # CA, its moment
        country="CA", assisted=1)                                         # the day before
    rec("22", vdate="2026-10-21", country="MX")                           # outside the scope
    rec("23", vdate="2026-10-21", country=None)                           # no country: outside
    rec("24", vdate="2026-10-01")                                         # green before the seal
    rec("25", vdate="2026-10-21", first_green="2026-10-11T09:00:00Z")     # moment before it
    rec("26", vdate="2026-10-22", conflict=1)                             # label conflict
    rec("27", vdate="2026-10-22")                                         # in the base already
    rec("28", vdate="2026-10-22", seed=999)                               # its photo is a base photo
    rec("29", vdate="2026-10-22", country="PR", copy=False)               # no kept original
    rec("30", vdate="2026-10-23", country=" pr")                          # read as PR
    conn.commit()
    base = Base(keys=["27", "1"], shas=[hashlib.sha256(jpeg(999)).hexdigest()])
    yield {"conn": conn, "path": path, "base": base}
    conn.close()


def test_only_records_first_green_after_the_seal_and_not_in_the_base_count(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    assert rep["records_first_green_after_seal"] == 9      # 24 and 25 were green before it
    assert rep["already_in_base"] == 1                     # 27
    t = rep["total"]
    assert (t["in_scope"], t["outside_scope"], t["excluded"]) == (3, 2, 3)
    assert rep["outside_scope_by_country"] == {"MX": 1, "unknown": 1}
    assert rep["excluded_by_reason"] == {"label_conflict": 1, "no_photos": 2}
    assert rep["photos_identical_to_base_dropped"] == 1


def test_a_find_is_counted_on_the_day_it_first_turned_green(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    by_day = {r["day"]: r for r in rep["days"]}
    assert by_day["2026-10-19"]["in_scope"] == 1           # 21: .org's moment, not its day
    assert by_day["2026-10-20"]["in_scope"] == 1           # 20
    assert by_day["2026-10-21"]["outside_scope"] == 2
    assert by_day["2026-10-23"]["in_scope"] == 1           # 30, " pr" read as PR
    assert [r["in_scope_cumulative"] for r in rep["days"]] == [1, 2, 2, 2, 3]


def test_the_scope_is_the_countries_given(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"],
                              countries=("US", "CA", "PR", "MX"))
    assert rep["total"]["in_scope"] == 4 and rep["outside_scope_by_country"] == {"unknown": 1}


def test_vision_assisted_and_first_green_at_are_reported_per_day(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    by_day = {r["day"]: r for r in rep["days"]}
    assert (by_day["2026-10-19"]["vision_assisted_1"], by_day["2026-10-19"]["vision_assisted_0"],
            by_day["2026-10-19"]["first_green_at_given"]) == (1, 0, 1)
    assert (by_day["2026-10-20"]["vision_assisted_1"],
            by_day["2026-10-20"]["vision_assisted_0"]) == (0, 1)
    assert rep["vision_assisted_column"] == "present"
    assert (rep["total"]["vision_assisted_1"], rep["total"]["vision_assisted_0"]) == (1, 2)


def test_a_manifest_without_the_column_says_not_populated(world):
    world["conn"].execute("alter table records drop column vision_assisted")
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"])
    assert rep["vision_assisted_column"] == "not populated"
    assert rep["total"]["vision_assisted_1"] == rep["total"]["vision_assisted_0"] == 0
    assert "not populated" in v1_test_count.format_count(rep)


def test_a_review_list_leaves_its_records_out(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=world["base"],
                              review_records={"20": "review:people"})
    assert rep["total"]["in_scope"] == 2
    assert rep["excluded_by_reason"]["review:people"] == 1
    assert "review lists" not in rep["not_applied"]


def test_with_no_base_the_seal_rule_alone_counts(world):
    rep = v1_test_count.count(world["conn"], seal=SEAL, base=None)
    assert rep["already_in_base"] == 0 and rep["base"] is None
    assert rep["total"]["in_scope"] == 5                   # 27 and 28 come back
    assert "the base release (seal rule only)" in rep["not_applied"]


def test_the_command_only_reads_the_manifest(world, capsys):
    world["conn"].execute("pragma journal_mode=delete")
    world["conn"].close()
    before = hashlib.sha256(world["path"].read_bytes()).hexdigest()
    assert cli.main(["v1-test-count", "--manifest", str(world["path"]), "--no-base",
                     "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["total"]["in_scope"] == 5 and rep["seal"] == SEAL
    assert hashlib.sha256(world["path"].read_bytes()).hexdigest() == before
    assert not world["path"].with_name("manifest.sqlite-wal").exists()
    ro = v1_test_count.open_manifest(world["path"])
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ro.execute("insert into records (observation_id) values ('x')")
    ro.close()
    world["conn"] = sqlite3.connect(":memory:")            # the fixture closes this one


def test_the_text_report_has_one_line_per_day_and_the_totals(world):
    text = v1_test_count.format_count(v1_test_count.count(world["conn"], seal=SEAL,
                                                          base=world["base"]))
    assert "2026-10-19" in text and "total in scope 3" in text
    assert "MX 1" in text and "crosspost_twin" in text
