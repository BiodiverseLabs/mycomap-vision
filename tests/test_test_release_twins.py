"""A test release leaves out the cross-post twins of its base (Vision Coordinator, 2026-10-18,
the rule of v1's erratum rev 4): a test record that is the same find as an included base
record, posted once to Mushroom Observer and once to iNaturalist (the same label, a photo
within 3 dHash bits), would be scored against its own photo. The test side is dropped as
`crosspost_twin` and counted in MANIFEST. A different label, two records of one source, a
template hash, and a base photo on the exclusion list never pair."""

import pytest
from test_dataset_release import add_photo, add_record, build, world  # noqa: F401
from test_test_release_copies import blocks, commit, encode, ids_file, index_of, near_copy

from mycomap_vision import dataset_release as dr

BASE = {  # key: (label, source, picture seed)
    "40": ("Pluteus cervinus", "inat", 40),
    "mo:41": ("Gymnopus dryophilus", "mo", 41),
    "42": ("Russula emetica", "inat", 42),
    "43": ("Laccaria laccata", "inat", 50),       # 43, 44, 45, 47: one picture, one hash on
    "44": ("Laccaria laccata", "inat", 50),       # four records: a template, never a twin
    "45": ("Laccaria laccata", "inat", 50),
    "47": ("Laccaria laccata", "inat", 50),
    "46": ("Hydnum repandum", "inat", 46),
}
TEST = {
    "mo:60": ("Pluteus cervinus", "mo", 40),      # twin of 40: dropped
    "61": ("Gymnopus dryophilus", "inat", 41),    # twin of mo:41: dropped
    "mo:62": ("Amanita muscaria", "mo", 42),      # another label: stays
    "63": ("Russula emetica", "inat", 42),        # iNat and iNat: not a cross-post, stays
    "mo:53": ("Laccaria laccata", "mo", 50),      # template: stays
    "mo:64": ("Hydnum repandum", "mo", 46),       # base photo excluded: stays (else dropped)
}
PID = {k: 7000 + i for i, k in enumerate(list(BASE) + list(TEST))}


def put(world, key, source, body):
    pid = PID[key] + (9_000_000_000 if source == "mo" else 0)
    add_photo(world["conn"], world["store"], key, pid, copy=False, source=source)
    rel = world["store"].put(pid, body)
    world["conn"].execute("insert into photo_copies values (?, ?, 'large', ?, ?, ?, 'now')",
                          (pid, str(world["store"].root), rel, len(body), dr.sha256_bytes(body)))


@pytest.fixture
def twin_world(world):
    for k, (label, source, seed) in BASE.items():
        add_record(world["conn"], k, label, source=source, vdate="2026-02-01")
        put(world, k, source, encode(blocks(seed)))
    commit(world)
    base = build(world, name="dry-20261018-100000")["id"]
    for k, (label, source, seed) in TEST.items():          # Steve's new records, after the cut
        add_record(world["conn"], k, label, source=source, vdate="2026-10-15")
        # re-encoded (other bytes); 63 and mo:62 come from one picture, so each gets its own
        # encoding, or the builder would drop both as one photo under two labels
        body = encode(blocks(seed), quality=80) if k == "63" else near_copy(blocks(seed))
        put(world, k, source, body)
    commit(world)
    return world, base, ids_file(world["tmp"], *TEST)


def released(world, rid):
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        recs = {k: (inc, reason) for k, inc, reason in rel.db.execute(
            "select record_key, included, reason from records")}
        return recs, set(rel.near_base())
    finally:
        rel.close()


def excluded_list(world, base, key):
    rel = dr.load_release(base, world["root"], allow_draft=True)
    try:
        pk = rel.db.execute("select photo_key from photos where record_key = ?",
                            (key,)).fetchone()[0]
    finally:
        rel.close()
    path = world["tmp"] / "micrographs.tsv"
    path.write_text(f"photo_key\tp\n{pk}\t0.9\n", encoding="utf-8")
    return path


def test_the_base_holds_every_base_record(twin_world):
    world, base, _ = twin_world
    rel = dr.load_release(base, world["root"], allow_draft=True)
    try:
        inc = {r[0] for r in rel.db.execute("select record_key from records where included = 1")}
    finally:
        rel.close()
    assert set(BASE) <= inc and not inc & set(TEST)


def test_a_test_record_whose_twin_is_in_the_base_is_left_out_and_counted(twin_world):
    world, base, ids = twin_world
    m = build(world, name="dry-20261018-110000-test", test_for=base, test_ids=ids,
              base_dhash=index_of(world, base),
              twin_exclude_photos=excluded_list(world, base, "46"))
    recs, flagged = released(world, m["id"])
    assert recs["mo:60"] == (0, "crosspost_twin")             # MO test side of an iNat base
    assert recs["61"] == (0, "crosspost_twin")                # iNat test side of an MO base
    for k in ("mo:62", "63", "mo:53", "mo:64"):
        assert recs[k] == (1, "ok"), k
    tw = m["crosspost_twin"]
    assert tw["records_dropped"] == 2 and tw["photos_dropped"] == 2
    assert tw["by_source"] == {"inat": 1, "mo": 1} and tw["max_bits"] == 3
    assert tw["exclude_photos"]["photos"] == 1
    # near_base still flags the near-copies that stay; the dropped ones are gone from it
    assert not {f"mo:{PID['mo:60'] + 9_000_000_000}", f"inat:{PID['61']}"} & flagged
    assert f"mo:{PID['mo:62'] + 9_000_000_000}" in flagged


def test_without_the_exclusion_list_the_excluded_base_photo_makes_a_twin(twin_world):
    world, base, ids = twin_world
    m = build(world, name="dry-20261018-120000-test", test_for=base, test_ids=ids,
              base_dhash=index_of(world, base))
    recs, _ = released(world, m["id"])
    assert recs["mo:64"] == (0, "crosspost_twin")
    assert m["crosspost_twin"]["records_dropped"] == 3
    assert m["crosspost_twin"]["exclude_photos"] is None


def test_a_draft_without_the_base_index_drops_nothing_and_says_so(twin_world):
    world, base, ids = twin_world
    m = build(world, name="dry-20261018-130000-test", test_for=base, test_ids=ids)
    recs, _ = released(world, m["id"])
    assert recs["mo:60"] == (1, "ok") and m["crosspost_twin"] is None
