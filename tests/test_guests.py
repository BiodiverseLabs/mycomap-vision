"""A record whose DNA name is a guest of the fungus in the photo (a yeast inside a puffball,
a parasite on a mushroom) is left out of every reference set, comparison, training run and
advance-prediction score; a mould that is itself the subject stays (guests.py)."""

import json

from test_one_word_names import seed

from mycomap_vision import evaluate, guests, prospective

# (name on .org, genus column, family column, red of its photo, validated)
RECORDS = [
    ("Teunomyces lycoperdinae", "Teunomyces", "Metschnikowiaceae", 200, "1/1/2026"),  # hidden
    ("Spinellus fusiger", "Spinellus", "Phycomycetaceae", 120, "1/1/2026"),           # on host
    ("Trichoderma viride", "Trichoderma", "Hypocreaceae", 60, "1/1/2026"),            # visible
    ("Lycoperdon perlatum", "Lycoperdon", "Lycoperdaceae", 205, "1/1/2026"),          # a host
]


def loaded(conn, tmp_path):
    seed(conn, tmp_path, RECORDS)
    ids = [r[0] for r in conn.execute("select photo_id from observation_photos")]
    return {r.genus for r in evaluate.load_records(conn, {p: p for p in ids})}


def test_a_hidden_guest_is_left_out_and_the_photos_fungus_and_visible_moulds_stay(conn, tmp_path,
                                                                                   monkeypatch):
    monkeypatch.delenv("MV_KEEP_ON_HOST", raising=False)
    genera = loaded(conn, tmp_path)
    assert "Teunomyces" not in genera
    assert {"Trichoderma", "Lycoperdon"} <= genera


def test_a_parasite_on_a_host_is_left_out_unless_asked_for(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("MV_KEEP_ON_HOST", raising=False)
    assert "Spinellus" not in loaded(conn, tmp_path)


def test_keep_on_host_brings_the_parasites_back_but_never_the_hidden_guests(conn, tmp_path,
                                                                            monkeypatch):
    monkeypatch.setenv("MV_KEEP_ON_HOST", "1")
    genera = loaded(conn, tmp_path)
    assert "Spinellus" in genera
    assert "Teunomyces" not in genera


def test_the_groups_and_what_each_leaves_out(monkeypatch):
    monkeypatch.delenv("MV_KEEP_ON_HOST", raising=False)
    assert guests.group_of("Teunomyces") == "hidden" and guests.excluded("Teunomyces") == "hidden"
    assert guests.excluded("Spinellus") == "on_host"
    assert guests.group_of("Trichoderma") == "visible" and guests.excluded("Trichoderma") is None
    assert guests.excluded("Russula") is None and guests.excluded("") is None
    assert guests.excluded(None) is None
    for a in guests.GROUPS:
        for b in guests.GROUPS:
            if a < b:
                assert not set(guests.GROUPS[a]) & set(guests.GROUPS[b]), (a, b)


def test_an_advance_prediction_of_a_guest_record_is_not_scored(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("MV_KEEP_ON_HOST", raising=False)
    seed(conn, tmp_path, RECORDS)
    conn.executescript(prospective.SCHEMA)
    result = {"species": [{"name": "Lycoperdon perlatum", "confidence": 0.9}],
              "genus": [{"name": "Lycoperdon", "confidence": 0.9}],
              "family": [{"name": "Lycoperdaceae", "confidence": 0.9}]}
    cols = [r[1] for r in conn.execute("pragma table_info(predictions)")]
    for oid in ("100", "103"):        # the Teunomyces record, and the real puffball
        row = {"observation_id": oid, "backbone": "m1", "method": "nearest",
               "result_json": json.dumps(result), "predicted_at": "2000-01-01T00:00:00+00:00",
               "code_version": "t"}
        keys = [k for k in row if k in cols]
        conn.execute(f"insert into predictions ({', '.join(keys)}) values "
                     f"({', '.join('?' for _ in keys)})", [row[k] for k in keys])
    conn.execute("update records set first_seen_at = '2026-01-02T00:00:00+00:00'")
    conn.commit()
    [row] = prospective.report(conn)
    assert row["resolved"] == 1, "only the puffball counts; the yeast's DNA name is no answer key"
    assert row["species_top1"] == 1.0
