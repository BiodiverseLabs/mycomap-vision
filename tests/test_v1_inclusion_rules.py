"""Three inclusion rules from the v1 audit (Steve, 2026-10-12):
1. the same image on records with different labels is left out of all of them;
2. an image that is also in a train record is left out of its val records;
3. a record .org gives no place counts as North American when its own source says so."""

import pytest
from test_dataset_release import add_photo, add_record, build, reasons, world  # noqa: F401

from mycomap_vision import dataset_release as dr
from mycomap_vision import mopoints, placefallback, records


def photo_reasons(world, rid):
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        return {r["photo_key"]: (r["reason"], r["included"])
                for r in rel.db.execute("select photo_key, reason, included from photos")}
    finally:
        rel.close()


# --- 1. one image, two names ------------------------------------------------------------

def test_an_image_on_records_with_different_labels_is_left_out_of_both(world):
    c, s = world["conn"], world["store"]
    add_record(c, "40", "Pseudoboletus parasiticus")
    add_photo(c, s, "40", 401, seed=777)                    # the host and its parasite
    add_photo(c, s, "40", 402)                              # a photo of its own
    add_record(c, "41", "Scleroderma citrinum")
    add_photo(c, s, "41", 411, seed=777)                    # the same bytes
    c.commit()
    m = build(world)
    pr = photo_reasons(world, m["id"])
    assert pr["inat:401"] == ("shared_other_label", 0)
    assert pr["inat:411"] == ("shared_other_label", 0)
    assert pr["inat:402"] == ("ok", 1)
    got = reasons(world, m["id"])
    assert got["40"][0] == "ok"                             # keeps its other photo
    assert got["41"][0] == "no_photos"                      # had only the shared one


def test_the_same_image_under_the_same_name_is_kept(world):
    c, s = world["conn"], world["store"]
    add_record(c, "42", "Russula emetica")
    add_photo(c, s, "42", 421, seed=888)
    add_record(c, "43", "Russula emetica")
    add_photo(c, s, "43", 431, seed=888)
    c.commit()
    pr = photo_reasons(world, build(world)["id"])
    assert pr["inat:421"][1] == 1 and pr["inat:431"][1] == 1


# --- 2. a val photo the model learnt from -----------------------------------------------

def test_a_val_copy_of_a_train_image_is_left_out_and_a_val_record_left_bare_leaves(world):
    c, s = world["conn"], world["store"]
    add_record(c, "50", "Russula emetica", vdate="2026-01-10")          # train
    add_photo(c, s, "50", 501, seed=999)
    add_record(c, "51", "Russula emetica", vdate="2026-09-20")          # val
    add_photo(c, s, "51", 511, seed=999)                                 # same bytes as 501
    add_photo(c, s, "51", 512)
    add_record(c, "52", "Russula emetica", vdate="2026-09-21")          # val, only the copy
    add_photo(c, s, "52", 521, seed=999)
    c.commit()
    m = build(world, val_cutoff="2026-09-01")
    pr = photo_reasons(world, m["id"])
    assert pr["inat:501"] == ("ok", 1)
    assert pr["inat:511"] == ("copy_in_train", 0)
    assert pr["inat:521"] == ("copy_in_train", 0)
    got = reasons(world, m["id"])
    assert got["51"] == ("ok", "val") and got["52"] == ("no_photos", None)
    assert m["val_copies_of_train_dropped"] == 2


# --- 3. no place on .org ----------------------------------------------------------------

def placeless(c, key):
    c.execute("update records set country = null, continent = null, latitude = null, "
              "longitude = null, north_america = 0 where observation_id = ?", (key,))


class INat:
    def __init__(self, places):
        self.places, self.headers, self.asked = places, {}, []

    def get(self, url, params, timeout):
        self.asked.append(params)
        body = {"results": [{"id": int(i), "place_ids": self.places[i]}
                            for i in params["id"].split(",") if i in self.places]}
        return type("R", (), {"status_code": 200, "json": lambda self: body, "text": ""})()


def test_a_placeless_record_is_north_american_when_its_source_says_so(world):
    from mycomap_vision.ratelimit import MinInterval
    c, s = world["conn"], world["store"]
    add_record(c, "60", "Russula emetica")
    add_photo(c, s, "60", 601)
    add_record(c, "61", "Russula emetica")
    add_photo(c, s, "61", 611)
    for k in ("60", "61"):
        placeless(c, k)
    c.commit()
    assert reasons(world, build(world)["id"])["60"][0] == "not_north_america"
    session = INat({"60": [1, 97394, 46], "61": [6744]})     # 61: elsewhere
    out = placefallback.fetch_inat(c, session=session, limiter=MinInterval(0),
                                   log=lambda s: None)
    assert out == {"inat_asked": 2, "inat_north_america": 1}
    assert session.asked[0]["fields"] == "id,place_ids"     # nothing else read
    assert placefallback.apply(c) == 1
    m = build(world)
    got = reasons(world, m["id"])
    assert got["60"][0] == "ok" and got["61"][0] == "not_north_america"
    assert m["north_america_from_source_place"] == 1


def test_a_reexport_keeps_the_source_place(world):
    c = world["conn"]
    add_record(c, "62", "Russula emetica")
    placeless(c, "62")
    placefallback.ensure_schema(c)
    c.execute("insert into place_fallback values ('62', 'inat', 1, 'inat_place_ids', null, 't')")
    c.commit()
    rows = [dict(zip([d[0] for d in cur.description], r)) for cur in
            [c.execute("select * from records")] for r in cur.fetchall()]
    for r in rows:
        r.setdefault("exported_at", "t2")
    records.save_records(c, [{k: r.get(k) for k in records._COLUMNS} | {"exported_at": "t2"}
                             for r in rows])
    assert c.execute("select north_america from records where observation_id = '62'"
                     ).fetchone()[0] == 1


def test_an_mo_record_takes_its_country_from_its_location_name(world):
    c = world["conn"]
    add_record(c, "mo:70", "Russula emetica", source="mo", status="ok")
    add_record(c, "mo:71", "Russula emetica", source="mo", status="ok")
    for k in ("mo:70", "mo:71"):
        placeless(c, k)
        c.execute("delete from mo_public_points where observation_id = ?", (k,))
    mopoints.ensure_schema(c)
    c.execute("insert into mo_public_points values ('mo:70', null, null, 0, 1, 0, 1, 0, "
              "'Monteverde, Puntarenas, Costa Rica', 'ok', 't')")
    c.execute("insert into mo_public_points values ('mo:71', null, null, 0, 1, 0, 1, 0, "
              "'Kew Gardens, London, England', 'ok', 't')")
    c.commit()
    out = placefallback.decide_mo(c)
    assert out["mo_north_america"] == 1
    assert placefallback.apply(c) == 1
    na = dict(c.execute("select observation_id, north_america from records where "
                        "observation_id in ('mo:70', 'mo:71')"))
    assert na == {"mo:70": 1, "mo:71": 0}


def test_a_record_with_a_place_on_org_is_never_overridden(world):
    c = world["conn"]
    add_record(c, "63", "Russula emetica", na=0)            # .org says France
    placefallback.ensure_schema(c)
    c.execute("insert into place_fallback values ('63', 'inat', 1, 'inat_place_ids', null, 't')")
    c.commit()
    assert placefallback.apply(c) == 0
    assert placefallback.country_of("Point Reyes, Marin Co., California, USA") == "USA"
    assert placefallback.in_north_america("USA") and not placefallback.in_north_america("Spain")
