import pytest

from mycomap_vision import manifest


@pytest.fixture(autouse=True)
def no_org_address(monkeypatch):
    """No test reads mycomap.org: a nightly run with the default steps would read its
    misidentifications table when a developer's .env names it. A test that needs the
    setting sets it."""
    monkeypatch.delenv("MV_ORG_BASE_URL", raising=False)


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


class _NoNetworkSession:
    """The review pages' source fetcher in tests: refuses to reach any outside host unless a
    test hands it a session of its own (photo_review.SourceFetcher)."""

    def __init__(self):
        self.headers = {}

    def get(self, url, **kw):
        raise AssertionError(f"tests must not fetch {url}")


@pytest.fixture(autouse=True)
def _review_photos_stay_offline(monkeypatch):
    from mycomap_vision import photo_review
    real = photo_review.SourceFetcher.__init__

    def offline(self, session=None, clock=None, sleep=None):
        real(self, session=session or _NoNetworkSession(), clock=clock, sleep=sleep)
    monkeypatch.setattr(photo_review.SourceFetcher, "__init__", offline)
