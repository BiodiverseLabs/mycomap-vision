import pytest

from mycomap_vision import manifest


@pytest.fixture
def conn(tmp_path):
    c = manifest.connect(tmp_path / "manifest.sqlite")
    yield c
    c.close()


def inat_obs(obs_id, user_login="alice", photos=(), **extra):
    """A v2 observation shaped like the API's field-selected answer."""
    return {
        "id": obs_id,
        "uuid": f"uuid-{obs_id}",
        "quality_grade": "research",
        "geoprivacy": None,
        "obscured": False,
        "location": "39.1,-86.5",
        "observed_on": "2025-09-01",
        "user": {"id": 7, "login": user_login, "name": user_login.title()},
        "taxon": {"id": 55, "name": "Russula sp.", "rank": "genus", "ancestor_ids": [48460, 55]},
        "observation_photos": [
            {"position": pos, "photo": {"id": pid, "license_code": lic,
                                         "url": f"https://{host}/photos/{pid}/square.jpg",
                                         "attribution": "(c) someone"}}
            for pos, pid, lic, host in photos
        ],
        **extra,
    }
