import pytest

from mycomap_vision.licenses import (license_class, looks_like_image, photo_relpath, sized_url,
                                     url_extension)
from mycomap_vision.ratelimit import ByteBudget, MinInterval


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


@pytest.mark.parametrize("code,expected", [
    (None, "arr"), ("", "arr"), ("  ", "arr"),
    ("cc0", "open"), ("CC-BY", "open"), ("cc-by-sa", "open"),
    ("cc-by-nc", "nc"), ("cc-by-nd", "nc"), ("cc-by-nc-sa", "nc"),
])
def test_missing_license_counts_as_all_rights_reserved(code, expected):
    assert license_class(code) == expected


def test_sized_url_swaps_size_and_drops_cache_buster():
    assert sized_url("https://static.inaturalist.org/photos/9/square.jpeg?1600000",
                     "medium") == "https://static.inaturalist.org/photos/9/medium.jpeg"
    assert sized_url("https://inaturalist-open-data.s3.amazonaws.com/photos/9/square.jpg",
                     "large").endswith("/photos/9/large.jpg")


def test_sized_url_refuses_unknown_urls_and_sizes():
    with pytest.raises(ValueError):
        sized_url("https://example.org/foo.jpg", "medium")
    with pytest.raises(ValueError):
        sized_url("https://x/photos/1/square.jpg", "huge")


def test_photo_paths_are_bucketed_by_id():
    assert photo_relpath(123456, "medium", "jpg") == "photos/medium/456/123456.jpg"
    assert url_extension("https://x/photos/1/square.PNG?5") == "png"


def test_image_sniffing_rejects_html_error_pages():
    assert looks_like_image(b"\xff\xd8\xff\xe0rest")
    assert looks_like_image(b"\x89PNG\r\n\x1a\n....")
    assert looks_like_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert not looks_like_image(b"<!DOCTYPE html>")


def test_min_interval_spaces_calls():
    clock = FakeClock()
    lim = MinInterval(1.0, clock=clock, sleep=clock.sleep)
    lim.wait()
    lim.wait()
    lim.wait()
    assert clock.slept == [1.0, 1.0]


def test_min_interval_pause_delays_next_call():
    clock = FakeClock()
    lim = MinInterval(1.0, clock=clock, sleep=clock.sleep)
    lim.wait()
    lim.pause(30)
    lim.wait()
    assert clock.slept == [30.0]


def test_byte_budget_blocks_until_old_bytes_age_out():
    clock = FakeClock()
    b = ByteBudget(limit=100, window=3600, clock=clock)
    b.add(60)
    assert b.wait_time() == 0
    clock.t += 600
    b.add(60)                    # 120 used: over the limit
    assert b.wait_time() == pytest.approx(3000)   # the first 60 ages out at +3600
    clock.t += 3000
    assert b.wait_time() == 0
    assert b.used() == 60


def test_bytes_from_an_earlier_run_count_until_they_age_out():
    clock = FakeClock()
    b = ByteBudget(limit=100, window=3600, clock=clock)
    b.add_past(3000, 80)         # fetched 50 minutes before this run started
    b.add_past(4000, 999)        # outside the window: ignored
    assert b.used() == 80
    b.add(40)                    # 120: over the limit until the old 80 ages out
    assert b.wait_time() == pytest.approx(600)
    clock.t += 600
    assert b.wait_time() == 0 and b.used() == 40


def test_old_photos_without_an_extension_still_resize_and_save_as_jpg():
    url = "https://inaturalist-open-data.s3.amazonaws.com/photos/2511038/square."
    assert sized_url(url, "large") == "https://inaturalist-open-data.s3.amazonaws.com/photos/2511038/large."
    assert url_extension(url) == "jpg"


def test_a_copyright_takedown_placeholder_is_recognised():
    from mycomap_vision.licenses import taken_down
    assert taken_down("https://www.inaturalist.org/assets/copyright-infringement-square.png")
    assert not taken_down("https://static.inaturalist.org/photos/1/square.jpg")
