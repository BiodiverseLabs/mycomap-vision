"""Human Feedback (Steve, 2026-10-18): a review queue of where Vision and the reference data
disagree. Each item carries the rule that picked it; a decision is one of its type's options,
stored with who made it and where it takes effect (.org, Vision, or a note); a seed never
duplicates an item; leaders and contributors see their projects' items; finds Vision got right
are mixed blind into the model-flagged queues."""

import pytest

from mycomap_vision import feedback as fb

ANN = {"sub": "u1", "name": "Ann"}


@pytest.fixture
def store(tmp_path):
    s = fb.Store(tmp_path / "state" / "human-feedback.sqlite")
    s.seed([
        {"item_type": "group_label", "subject": "Laccaria laccata group", "rule": "R2",
         "records": 12, "current": "Laccaria laccata group", "projects": ["p1"]},
        {"item_type": "photos_unlike_label", "subject": "123", "rule": "M1",
         "record_keys": ["123"], "current": "Morchella americana", "projects": ["p2"]},
        {"item_type": "non_specimen_photo", "subject": "inat:555", "rule": "R6",
         "record_keys": ["55"], "projects": ["p1"]},
    ], "test-seed")
    return s


def test_a_seed_never_duplicates_an_item_and_refuses_an_unknown_type(store):
    again = store.seed([{"item_type": "group_label", "subject": "Laccaria laccata group",
                         "rule": "R2"}], "again")
    assert again == {"seed": "again", "rows": 1, "added": 0}
    with pytest.raises(fb.FeedbackError, match="unknown item type"):
        store.seed([{"item_type": "looks_odd", "subject": "x", "rule": "?"}], "bad")


def test_every_item_says_which_rule_picked_it_and_model_flags_follow_the_type(store):
    items = store.items("photos_unlike_label")["items"]
    assert items[0]["rule"] == "M1" and items[0]["model_flagged"] is True
    assert store.items("group_label")["items"][0]["model_flagged"] is False


def test_a_decision_must_be_one_of_the_types_options_and_records_where_it_applies(store):
    item = store.items("group_label")["items"][0]
    with pytest.raises(fb.FeedbackError, match="not a decision"):
        store.decide(item["id"], "exclude_photo", ANN)
    with pytest.raises(fb.FeedbackError, match="needs a value"):
        store.decide(item["id"], "rename", ANN)
    out = store.decide(item["id"], "rename", ANN, value="Laccaria sp. 'laccata-PNW01'",
                       note="per DNA")
    assert out["status"] == "decided"
    d = out["decisions"][-1]
    assert (d["decision"], d["applies"], d["by_user"], d["value"]) == \
        ("rename", "org", "u1", "Laccaria sp. 'laccata-PNW01'")
    assert store.items("group_label")["total"] == 0                  # off the open queue
    assert store.items("group_label", status="decided")["total"] == 1
    assert store.reopen(item["id"])["status"] == "open"


def test_changes_list_the_latest_decision_per_item_until_applied(store):
    item = store.items("group_label")["items"][0]
    store.decide(item["id"], "keep_group", ANN)
    store.decide(item["id"], "rename", ANN, value="Laccaria nobilis")
    ch = store.changes(applies="org")
    assert len(ch) == 1 and ch[0]["value"] == "Laccaria nobilis" and ch[0]["records"] == 12
    store.mark_applied(ch[0]["id"], ANN)
    assert store.changes(applies="org") == []
    assert len(store.changes(applies="org", pending_only=False)) == 1


def test_photo_exclusions_vision_applies_come_from_both_photo_decisions(store):
    photo = store.items("non_specimen_photo")["items"][0]
    rec = store.items("photos_unlike_label")["items"][0]
    store.decide(photo["id"], "exclude_photo", ANN)
    store.decide(rec["id"], "exclude_photos", ANN, photo_keys=["inat:1", "inat:2"])
    assert store.excluded_photos() == {"inat:555", "inat:1", "inat:2"}
    store.decide(photo["id"], "keep_photo", ANN)                     # changed its mind
    assert store.excluded_photos() == {"inat:1", "inat:2"}


def test_leaders_and_contributors_see_only_their_projects_items(store):
    assert store.items("group_label", projects={"p2"})["total"] == 0
    assert store.items("photos_unlike_label", projects={"p2"})["total"] == 1
    item = store.items("group_label")["items"][0]
    with pytest.raises(KeyError):
        store.get(item["id"], projects={"p2"})
    with pytest.raises(KeyError):
        store.decide(item["id"], "keep_group", ANN, projects={"p2"})
    by_type = {s["type"]: s for s in store.summary(projects={"p1"})}
    assert by_type["group_label"]["open"] == 1 and by_type["photos_unlike_label"]["open"] == 0
    assert {s["type"] for s in store.summary()} == set(fb.TYPES)        # admins: every type


def test_finds_vision_got_right_are_mixed_in_blind(store):
    hits = [{"record_key": str(k), "label": "Amanita muscaria"} for k in range(100)]
    rows = fb.blind_sample(hits, "photos_unlike_label", 5, seed=7)
    assert rows == fb.blind_sample(hits, "photos_unlike_label", 5, seed=7)    # seeded
    store.seed(rows, "blind")
    items = store.items("photos_unlike_label")["items"]
    blind = [i for i in items if i["blind"]]
    assert len(blind) == 5 and all(i["hidden_until_decided"] for i in blind)
    assert all(i["model_flagged"] for i in items)


def test_refreshes_are_kept_with_the_item(store):
    item = store.items("photos_unlike_label")["items"][0]
    store.record_refresh(item["id"], ANN, "Xylodon ovisporus", [["Xylodon ovisporus", 0.6]], False)
    r = store.get(item["id"])["refreshes"][0]
    assert r["name_now"] == "Xylodon ovisporus" and r["still_disagrees"] == 0


def test_the_catalogue_lists_every_type_with_options_and_where_each_applies():
    cat = {t["type"]: t for t in fb.type_catalogue()}
    assert set(cat) == set(fb.TYPES)
    for t in cat.values():
        assert t["options"] and all(o["applies"] in ("org", "vision", "note") for o in t["options"])
    assert cat["crosswalk"]["granularity"] == "name pair"
    assert {o["code"] for o in cat["crosswalk"]["options"]} == {"same", "nested", "different"}


def test_seed_files_from_the_miss_review_load(tmp_path):
    q = tmp_path / "review_queue_seed.tsv"
    q.write_text("item_id\titem_type\tgranularity\tselection_rule\tmodel_flagged\tsubject\tlink\t"
                 "records\tcurrent\tproposed\tevidence\tin_v1\tv1_val_outcome\n"
                 "q1\tcrosspost_twin\trecord pair\tR7\t0\tmo:1 + 2\t"
                 "https://mushroomobserver.org/obs/1 ; https://www.inaturalist.org/observations/2\t2\t"
                 "Amanita muscaria\tone find\tphoto dHash\ttrain/val\t\n", encoding="utf-8")
    rows = fb.rows_from_queue_seed(q)
    assert rows[0]["record_keys"] == ["mo:1", "2"] and len(rows[0]["links"]) == 2
    c = tmp_path / "crosswalk_candidates.tsv"
    cols = ["name_a", "kind_a", "records_a", "name_b", "kind_b", "records_b", "evidence",
            "v1_val_misses_resolved", "misses_detail", "dna_records", "dna_detail",
            "example_sequences", "name_authority", "shared_stem", "inat_status", "decision",
            "decision_options"]
    c.write_text("\t".join(cols) + "\n" + "\t".join([
        "Hygrophorus chrysodon", "formal", "40", "Hygrophorus sp. 'chrysodon-CA01'", "code", "9",
        "vision, stem", "2", "", "0", "", "", "", "True", "", "", ""]) + "\n", encoding="utf-8")
    cw = fb.rows_from_crosswalk(c)
    assert cw[0]["item_type"] == "crosswalk" and cw[0]["records"] == 49
    assert cw[0]["evidence"]["v1_val_misses_resolved"] == "2"


def test_seeding_from_files_adds_projects_and_a_blind_sample(tmp_path, conn):
    from mycomap_vision import manifest  # noqa: F401  (conn fixture's schema)
    conn.execute("insert into records (observation_id, source, scientific_name, north_america, "
                 "green_projects, exported_at) values ('7', 'inat', 'Laccaria nobilis', 1, "
                 "'[\"Idaho Fungi\"]', 'now'), ('8', 'inat', 'Laccaria laccata group', 1, "
                 "'[\"General Validation\"]', 'now')")
    q = tmp_path / "q.tsv"
    q.write_text("item_id\titem_type\tgranularity\tselection_rule\tmodel_flagged\tsubject\tlink\t"
                 "records\tcurrent\tproposed\tevidence\tin_v1\tv1_val_outcome\n"
                 "q1\tgroup_label\tname\tR2\t0\tLaccaria laccata group\t\t1\t"
                 "Laccaria laccata group\t\t\t\t\n"
                 "q2\tphotos_unlike_label\trecord\tM1\t1\t7\thttps://x/7\t1\tLaccaria nobilis\t\t\tval\tmiss\n"
                 "q3\tphotos_unlike_label\trecord\tM1\t1\t9\thttps://x/9\t1\tAmanita\t\t\tval\tmiss\n",
                 encoding="utf-8")
    h = tmp_path / "hits.tsv"
    h.write_text("record_key\tlabel\n" + "".join(f"{i}\tRussula emetica\n" for i in range(100, 120)),
                 encoding="utf-8")
    s = fb.Store(tmp_path / "f.sqlite")
    out = fb.seed_files(s, conn, queue=q, hits=h, blind_ratio=0.5, seed=3)
    assert out[0]["added"] == 3 and out[1]["added"] == 1          # 2 flagged x 0.5 = 1 blind
    group = s.items("group_label")["items"][0]
    assert group["projects"] == ["General Validation"]               # every record of the name
    flagged = s.items("photos_unlike_label")["items"]
    assert {tuple(i["projects"]) for i in flagged if not i["blind"]} == {("Idaho Fungi",), ()}
    assert sum(i["blind"] for i in flagged) == 1
    again = fb.seed_files(s, conn, queue=q, hits=h, blind_ratio=0.5, seed=3)
    assert again[0]["added"] == 0 and again[1]["added"] == 0         # never duplicated


def test_photos_left_out_by_a_decision_leave_the_reference_at_the_next_sync(tmp_path, conn):
    """Vision applies its own decisions: a photo left out never reaches load_records (the
    served index, training) once synced; changing the decision brings it back."""
    from test_models_and_scoreboard import seed_two_species
    from mycomap_vision import evaluate
    seed_two_species(conn, tmp_path)
    ids = {p for p in conn.execute("select photo_id from observation_photos")}
    before = evaluate.load_records(conn, {int(p[0]): int(p[0]) for p in ids})
    assert any(1000 in r.photo_rows for r in before)
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([{"item_type": "non_specimen_photo", "subject": "inat:1000", "rule": "R6"}], "t")
    iid = s.items("non_specimen_photo")["items"][0]["id"]
    s.decide(iid, "exclude_photo", ANN)
    fp0 = fb.exclusions_fingerprint(conn)
    out = fb.sync_exclusions(s, conn)
    assert out["photos"] == 1 and out["fingerprint"] != fp0
    after = evaluate.load_records(conn, {int(p[0]): int(p[0]) for p in ids})
    assert not any(1000 in r.photo_rows for r in after)
    s.decide(iid, "keep_photo", ANN)
    assert fb.sync_exclusions(s, conn)["photos"] == 0
    assert any(1000 in r.photo_rows for r in evaluate.load_records(conn, {int(p[0]): int(p[0]) for p in ids}))
    assert fb.photo_id_of("mo:45") == 1_000_000_000_045 and fb.photo_id_of("x") is None


def test_twin_location_conflicts_seed_without_coordinates_and_decide_on_org(tmp_path, conn):
    """One find posted to MO and iNat whose copies sit over 5 km apart: a pair item with the
    distance, how each copy is placed and which looks wrong, and no coordinate anywhere;
    fixing either copy is a decision for mycomap.org / MO / iNat (codes kept stable)."""
    f = tmp_path / "twins.tsv"
    f.write_text("distance_km\tmo_org_id\tmo_number\tmo_placement\tinat_org_id\tinat_number\t"
                 "inat_accuracy_m\ttaxon\tlikely_wrong_copy\n"
                 "9336.2\t180324\t324331\tMO GPS\t560948\t172307494\t20\txylodon spathulatus\t"
                 "unclear: both look precise\n"
                 "7.4\t1\t2\tnamed location\t3\t4\t\tagrocybe arvalis\tMO copy: named location\n",
                 encoding="utf-8")
    s = fb.Store(tmp_path / "f.sqlite")
    out = fb.seed_files(s, conn, twin_locations=f)
    assert out[0]["added"] == 2
    items = {i["subject"]: i for i in s.items("twin_location_conflict")["items"]}
    far = items["mo:324331 + 172307494"]
    assert far["record_keys"] == ["mo:324331", "172307494"] and far["records"] == 2
    assert far["evidence"]["distance_km"] == "9336" and items["mo:2 + 4"]["evidence"]["distance_km"] == "7.4"
    assert items["mo:2 + 4"]["evidence"]["inat_accuracy_m"] == "not recorded"
    assert set(far["evidence"]) == {"distance_km", "mo_placement", "inat_accuracy_m",
                                    "likely_wrong_copy"}               # no coordinate field
    assert not {"latitude", "longitude", "lat", "lon", "lng"} & {k.lower() for k in far}
    codes = {o["code"]: o["applies"] for o in
             next(t for t in fb.type_catalogue() if t["type"] == "twin_location_conflict")["options"]}
    assert codes == {"fix_mo_location": "org", "fix_inat_location": "org", "both_right": "note"}
    d = s.decide(far["id"], "fix_inat_location", ANN)
    assert d["decisions"][-1]["applies"] == "org"
    assert [c["subject"] for c in s.changes(applies="org")] == ["mo:324331 + 172307494"]
