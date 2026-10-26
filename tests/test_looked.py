"""Where Vision looked (looked.py; Steve, 2026-10-26): each photo's map of where its evidence
for the answer's name was, its match with the specimen photo, and the worker that makes them
one photo at a time behind every identification."""

import io
import threading

import numpy as np
import pytest
from PIL import Image

from mycomap_vision import looked, workqueue


def jpeg(size=(80, 60), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


# --- the crop, cleaning, smoothing, the stored map ------------------------------------------

def test_the_map_covers_the_centre_square_the_model_sees():
    assert looked.crop_box(768, 1024) == {"x": 0.0, "y": 0.125, "w": 1.0, "h": 0.75}
    assert looked.crop_box(1000, 500) == {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}
    assert looked.crop_box(300, 300) == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def test_a_register_patch_takes_its_neighbours_median_and_nothing_else_changes():
    m = np.full((5, 5), 1.0)
    m[2, 2] = 50.0                                       # a register's speck
    m[0, 0] = 3.0
    reg = np.zeros((5, 5), bool)
    reg[2, 2] = True
    out = looked.clean(m, reg)
    assert out[2, 2] == 1.0
    assert out[0, 0] == 3.0 and (out[~reg] == m[~reg]).all()


def test_smoothing_keeps_a_flat_map_flat_and_spreads_a_peak_to_its_neighbours():
    assert np.allclose(looked.smooth(np.full((6, 6), 2.0)), 2.0)
    m = np.zeros((6, 6))
    m[3, 3] = 16.0
    s = looked.smooth(m)
    assert s[3, 3] == 4.0 and s[2, 3] == 2.0 and s[2, 2] == 1.0 and s[0, 0] == 0.0


def test_the_kept_map_runs_from_0_to_255_with_nothing_below_zero():
    m = np.linspace(-1, 9, 100).reshape(10, 10)
    q = looked.quantise(m)
    assert len(q) == 100 and min(q) == 0 and max(q) == 255
    assert q[0] == 0                                     # negative evidence shows as none


# --- matches ------------------------------------------------------------------------------

def tokens(grid: int, dim: int = 32, seed: int = 0):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(grid * grid, dim, generator=g)
    return t / t.norm(dim=1, keepdim=True)


def test_matches_pair_each_patch_with_its_own_match_in_the_specimen_photo():
    torch = pytest.importorskip("torch")
    g = 24
    tq = tokens(g)
    perm = torch.randperm(g * g, generator=torch.Generator().manual_seed(1))
    tr = tq[perm]                                         # the specimen: the same patches, moved
    where = {int(perm[j]): j for j in range(g * g)}       # query patch -> its place in the specimen
    qmap = np.random.default_rng(2).random((g, g))
    none = np.zeros((g, g), bool)
    pairs = looked.match_pairs(tq, tr, qmap, none, none)
    assert 1 <= len(pairs) <= looked.MATCH_PAIRS
    for qx, qy, rx, ry in pairs:
        i = int(round(qy * g - 0.5)) * g + int(round(qx * g - 0.5))
        j = int(round(ry * g - 0.5)) * g + int(round(rx * g - 0.5))
        assert where[i] == j


def test_matches_stay_off_the_edges_off_registers_and_apart():
    pytest.importorskip("torch")
    g = 24
    tq = tokens(g, seed=3)
    qmap = np.random.default_rng(4).random((g, g))
    skip = np.zeros((g, g), bool)
    skip[10:14, 10:14] = True
    pairs = looked.match_pairs(tq, tq.clone(), qmap, skip, skip, k=10)
    cells = [(int(round(x * g - 0.5)), int(round(y * g - 0.5))) for x, y, _, _ in pairs]
    assert cells
    for x, y in cells:
        assert 2 <= x < g - 2 and 2 <= y < g - 2
        assert not skip[y, x]
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            assert max(abs(cells[a][0] - cells[b][0]), abs(cells[a][1] - cells[b][1])) >= 5


# --- the model: a tiny real open_clip tower ----------------------------------------------------

def tiny_tower():
    torch = pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    from open_clip.transformer import VisionTransformer
    from torchvision import transforms as T
    torch.manual_seed(0)
    vit = VisionTransformer(image_size=56, patch_size=14, width=64, layers=6, heads=4,
                            mlp_ratio=2.0, output_dim=32).eval()
    tf = T.Compose([T.Resize(56), T.CenterCrop(56), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    class Loaded:                                      # models.FinetunedBackbone's shape
        class base:
            model = vit
        transform = tf
    return vit, tf, Loaded


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_the_block_by_block_tower_gives_the_towers_own_vector():
    torch = pytest.importorskip("torch")
    vit, tf, _ = tiny_tower()
    tower = looked.Tower(vit, tf)
    x, _ = tower.prepare(jpeg())
    with torch.no_grad():
        own = torch.nn.functional.normalize(vit(x[None]), dim=-1)
        mine, atts, _ = tower.run(x[None], attention=True)
    assert float((own * mine).sum()) > 0.9999
    assert len(atts) == 6 and tuple(atts[0].shape) == (1, 4, 17, 17)


def test_the_trained_tower_is_found_inside_a_loaded_model_and_a_stand_in_is_refused():
    vit, tf, Loaded = tiny_tower()
    assert looked.vit_of(Loaded) == (vit, tf)
    assert looked.vit_of(object()) is None


def test_the_name_score_is_the_served_score():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    v = unit(rng.normal(size=(7, 16))).astype(np.float32)
    record = np.array([0, 0, 1, 2, 2, 2, 3])
    mean = unit(rng.normal(size=16)).astype(np.float32)
    t = looked.Target("X", v, record, mean, weight=0.6, k=2)
    q = unit(rng.normal(size=(3, 16))).astype(np.float32)
    got = t.score(torch.from_numpy(q)).numpy()
    sims = q @ v.T
    best = np.stack([sims[:, record == r].max(1) for r in range(4)], 1)
    near = -np.sort(-best, 1)[:, :2].mean(1)
    assert np.allclose(got, 0.6 * near + 0.4 * (q @ mean), atol=1e-6)
    one = looked.Target("Y", v[:1], np.array([0]), mean)       # a name with one record
    assert np.allclose(one.score(torch.from_numpy(q)).numpy(),
                       0.6 * (q @ v[0]) + 0.4 * (q @ mean), atol=1e-6)


def target_of(tower, body: bytes):
    """A name whose only reference photo is this one: its evidence is the whole photo."""
    torch = pytest.importorskip("torch")
    x, _ = tower.prepare(body)
    with torch.no_grad():
        v = tower.run(x[None], attention=False)[0][0].numpy()
    return looked.Target("X", v[None].astype(np.float32), np.array([0]), v.astype(np.float32))


def test_a_photo_map_has_one_value_per_patch_and_hiding_its_brightest_patches_costs_most():
    torch = pytest.importorskip("torch")
    vit, tf, _ = tiny_tower()
    tower = looked.Tower(vit, tf)
    body = jpeg((70, 56), seed=7)
    target = target_of(tower, body)
    x, _ = tower.prepare(body)
    m, reg, _ = looked.evidence(tower, x, target)
    assert m.shape == (4, 4) and np.isfinite(m).all() and (m >= 0).all()
    assert reg.shape == (4, 4)
    # Faithfulness, as in the research look: blanking the map's top quarter of patches lowers
    # the name's score more than blanking its bottom quarter.
    order = np.argsort(-m.ravel())

    def without(cells):
        y = x.clone()
        for c in cells:
            r, k = divmod(int(c), 4)
            y[:, r * 14:(r + 1) * 14, k * 14:(k + 1) * 14] = 0
        with torch.no_grad():
            return float(target.score(tower.run(y[None], attention=False)[0])[0])
    assert without(order[:4]) < without(order[-4:])


def test_a_photo_entry_carries_its_crop_map_and_match():
    pytest.importorskip("torch")
    vit, tf, _ = tiny_tower()
    tower = looked.Tower(vit, tf)
    body = jpeg((80, 60), seed=8)
    entry = looked.photo_map(tower, body, target_of(tower, body), jpeg((60, 60), seed=9))
    assert entry["grid"] == 4 and len(entry["map"]) == 16
    assert entry["crop"] == {"x": 0.125, "y": 0.0, "w": 0.75, "h": 1.0}
    assert isinstance(entry["pairs"], list)
    assert entry["specimen_crop"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    assert looked.photo_map(tower, body, target_of(tower, body), None)["pairs"] is None


# --- the worker: behind every identification -------------------------------------------------------

def job(rid="r1", photos=2, made=None, fail_at=None, fetched=None):
    made = made if made is not None else []

    def fetch(i):
        if fetched is not None:
            fetched.append(i)
        return f"photo {i}"

    def make(got, i):
        if fail_at == i:
            raise RuntimeError("broken photo")
        made.append(got)
        return {"map": [i]}
    kept = []
    return looked.Job(rid, [looked.Task(i, lambda i=i: fetch(i), lambda got, i=i: make(got, i))
                            for i in range(photos)], kept.append), made, kept


def test_maps_wait_while_identifications_are_queued_or_running():
    lock = threading.Lock()
    busy = {"polls": 3}
    seen = []

    def idle():
        busy["polls"] -= 1
        return busy["polls"] < 0

    w = looked.Worker(idle, lock, background=False, sleep=lambda s: seen.append(s))
    j, made, kept = job(photos=1)
    w.submit(j)
    assert len(seen) == 3                                 # waited three polls first
    assert made == ["photo 0"] and kept and kept[0][0]["photo"] == 0


def test_each_map_holds_the_model_lock_for_one_photo_only_and_photos_are_read_outside_it():
    lock = threading.Lock()
    states = []

    def idle():
        states.append(("between photos, lock free", not lock.locked()))
        return True

    w = looked.Worker(idle, lock, background=False)
    made = []

    def make(got, i):
        states.append(("making, lock held", lock.locked()))
        made.append(i)
        return {}

    def fetch(i):
        states.append(("reading, lock free", not lock.locked()))
        return i
    w.submit(looked.Job("r", [looked.Task(i, lambda i=i: fetch(i), lambda g, i=i: make(g, i))
                              for i in range(3)], lambda e: None))
    assert made == [0, 1, 2]
    assert all(ok for _, ok in states) and len(states) == 9


def test_the_worker_waits_for_the_real_model_queue_to_be_empty():
    lock = threading.Lock()
    q = workqueue.ModelQueue(lock, start=False)
    assert q.idle()
    waiting = threading.Thread(target=lambda: q.run(workqueue.Job(
        key="k", photos=1, embed=lambda: np.zeros((1, 2)), search=lambda v: np.zeros((1, 1)),
        finish=lambda v, s: {}), timeout=5), daemon=True)
    waiting.start()
    for _ in range(100):
        if not q.idle():
            break
        threading.Event().wait(0.01)
    assert not q.idle()                                   # an identification is waiting
    polls = []

    def sleep(_):
        polls.append(1)
        if len(polls) == 2:                               # the identification gets its turn
            with q._cond:
                batch = q._take_batch()
                q._busy = True
            q.run_batch(batch)
            with q._cond:
                q._busy = False
                q._running_photos = 0

    w = looked.Worker(q.idle, lock, background=False, sleep=sleep)
    j, made, _ = job(photos=1)
    w.submit(j)
    assert len(polls) == 2 and made == ["photo 0"]
    waiting.join(5)


def test_too_many_photos_waiting_is_refused_with_how_long_to_wait():
    gate = threading.Event()
    w = looked.Worker(lambda: gate.is_set(), threading.Lock(), max_waiting_photos=5,
                      seconds_per_photo=3.0, poll=0.01)
    try:
        w.submit(job("a", photos=4)[0])
        with pytest.raises(looked.Busy) as e:
            w.submit(job("b", photos=3)[0])
        assert e.value.wait_seconds == pytest.approx(12.0)
        assert w.status("b") is None
        assert w.status("a")["status"] == "working"
    finally:
        gate.set()


def test_asking_again_for_the_same_run_joins_the_first_ask():
    gate = threading.Event()
    w = looked.Worker(lambda: gate.is_set(), threading.Lock(), poll=0.01)
    fetched = []
    try:
        first, made, kept = job("same", photos=2, fetched=fetched)
        w.submit(first)
        w.submit(job("same", photos=2, fetched=fetched)[0])
        assert w.waiting_photos() == 2
    finally:
        gate.set()
    first.finished.wait(5)
    assert made == ["photo 0", "photo 1"] and len(kept) == 1


def test_a_failed_run_says_so_and_can_be_asked_again():
    w = looked.Worker(lambda: True, threading.Lock(), background=False, note=lambda s: None)
    j, made, kept = job("f", photos=2, fail_at=1)
    w.submit(j)
    assert w.status("f")["status"] == "failed" and not kept
    j2, _, kept2 = job("f", photos=2)
    w.submit(j2)
    assert kept2 and w.status("f") is None
