"""The nightly update and the advance predictions take a record's source from
mycomap.org's feed, never from its id (sources.py)."""

import pytest
from test_nightly import QUIET, Pages, page, seed_rows

from mycomap_vision import nightly, prospective
from mycomap_vision.records import build_records


def without_source(rows):
    return [{k: v for k, v in r.items() if k != "source"} for r in rows]


@pytest.mark.parametrize("fetch", [nightly.fetch_green, nightly.fetch_pending])
def test_a_feed_without_source_is_refused_as_such_not_read_as_unknown_records(fetch):
    rows = without_source(seed_rows())
    with pytest.raises(nightly.NightlyRefused, match="lacks source") as e:
        fetch("https://mycomap.org", "k" * 40,
              session=Pages({"": page(rows, None, total=len(rows))}), **QUIET)
    assert "nothing was changed" in str(e.value)


def test_a_feed_with_some_rows_lacking_source_is_refused_too():
    rows = seed_rows()
    rows[1] = without_source([rows[1]])[0]
    with pytest.raises(nightly.NightlyRefused, match=r"1 of \d+ rows"):
        nightly.fetch_green("https://mycomap.org", "k" * 40,
                            session=Pages({"": page(rows, None, total=len(rows))}), **QUIET)


def test_a_null_source_is_an_answer_not_a_missing_field():
    rows = seed_rows()
    rows[0]["source"] = None                  # .org has the row but no source: 'unknown'
    got, _ = nightly.fetch_green("https://mycomap.org", "k" * 40,
                                 session=Pages({"": page(rows, None, total=len(rows))}), **QUIET)
    assert len(got) == len(rows)


def org(oid, source, name="Hydnum repandum"):
    return {"observation_id": oid, "source": source, "scientific_name": name,
            "genus": name.split()[0], "continent": "North America",
            "validation_status_1": "yes", "validation_date_1": "9/2/2026"}


def legacy(conn, *ids):
    """A manifest written by older code: bare ids, source guessed 'inat', not migrated."""
    for oid in ids:
        conn.execute("insert into records (observation_id, source, north_america, exported_at, "
                     "scientific_name) values (?, 'inat', 1, 'old', 'Hydnum repandum')", (oid,))
    conn.execute("delete from manifest_migrations")


def test_a_rekey_by_the_source_migration_is_not_a_removal(conn):
    legacy(conn, "123", "456", "789", "MK1")
    recs = build_records([org("123", "MO Observations"), org("456", "iNaturalist"),
                          org("MK1", "GenBank Accessions")], "t")     # 789 really left
    plan = nightly.changes(conn, recs)
    assert (plan["removed"], plan["new"], plan["rekeyed"]) == (1, 0, 2)
    assert plan["examples"]["removed"] == ["789"]
    assert plan["examples"]["rekeyed"] == ["123 -> mo:123", "MK1 -> genbank:MK1"]
    assert plan["source_migration"]["wrong_sourced_records"] == 2


def test_after_the_migration_a_vanished_key_is_a_removal_again(conn):
    # A migrated manifest never re-keys: 'mo:5' leaving the feed is a removal.
    conn.execute("insert into records (observation_id, source, source_id, north_america, "
                 "exported_at, scientific_name) values ('mo:5', 'mo', '5', 1, 't', 'A b')")
    plan = nightly.changes(conn, build_records([org("6", "MO Observations")], "t"))
    assert (plan["removed"], plan["new"]) == (1, 1)
    assert "rekeyed" not in plan and "source_migration" not in plan


def test_a_pending_record_is_predicted_only_when_its_source_says_inat(conn):
    rows = [{"observation_id": "1", "source": "iNaturalist", "scientific_name": "A b",
             "continent": "North America"},
            {"observation_id": "2", "source": "MO Observations", "scientific_name": "A b",
             "continent": "North America"},
            {"observation_id": "3", "source": "MycoPortal", "continent": "North America"},
            {"observation_id": "4", "continent": "North America"}]     # no source at all
    assert prospective.save_candidates(conn, rows, "t") == 1
    assert [r[0] for r in conn.execute("select observation_id from candidates")] == ["1"]


def test_the_candidate_export_asks_for_inat_rows_by_source():
    assert "source = 'iNaturalist'" in prospective.CANDIDATE_SQL
