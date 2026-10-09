"""Likely sets fitted from compact rows (not every record's full row of scores), only
where they are read, and not at all when asked not to."""
import json

import numpy as np
import pytest

from test_likely import compared
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate, likely
from mycomap_vision.embed import embed_photos, photos_to_embed


def random_rows(n, labels, rng):
    """Scores like a model's (some names unscored), with true names near the top, far
    down (past the cut) or absent from the reference set."""
    out = []
    for i in range(n):
        s = rng.normal(0, 0.03, labels)
        s[rng.random(labels) < 0.05] = -np.inf
        t = int(rng.integers(labels))
        if rng.random() < 0.75:                      # most true names near the top
            s[t] = np.nanmax(np.where(np.isfinite(s), s, np.nan)) + rng.normal(0, 0.02)
        if rng.random() < 0.09:
            t = None                                                        # unknown name
        elif not np.isfinite(s[t]):
            s[t] = 0.0
        out.append((s, t))
    return out


def test_compact_rows_fit_exactly_what_full_rows_fit():
    rng = np.random.default_rng(5)
    rows = random_rows(600, 400, rng)
    ti = 40
    T = float(evaluate.T_GRID[ti])
    full = [(likely.probabilities(s, T), t) for s, t in rows]
    small = [likely.expand(likely.compact(s, t, evaluate.T_GRID), ti, T) for s, t in rows]
    assert [likely.score(*r) for r in full] == pytest.approx([likely.score(*r) for r in small])
    # Spread-out synthetic scores: lift the list-length limit so a target is fitted.
    a = likely.fit_and_check(full, max_mean_size=1000)
    b = likely.fit_and_check(small, max_mean_size=1000)
    assert a is not None
    assert a["floor"] == pytest.approx(b["floor"], abs=1e-6)
    assert {k: v for k, v in a.items() if k != "floor"} == {k: v for k, v in b.items() if k != "floor"}


def test_the_temperature_fit_is_unchanged_by_sharing_the_normaliser():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 0.03, 300)
    s[:7] = -np.inf
    lse = likely.logsumexp_by_temperature(s, evaluate.T_GRID)
    direct = np.array([np.log(np.exp(s[np.isfinite(s)] / T).sum()) - s[42] / T
                       for T in evaluate.T_GRID])
    assert evaluate.nll_by_temperature(s, 42, lse) == pytest.approx(direct, rel=1e-9)


def test_evaluate_keeps_only_compact_rows(conn, tmp_path, monkeypatch):
    kept = []
    real = likely.compact

    def spy(*a, **kw):
        row = real(*a, **kw)
        kept.append(row)
        return row
    monkeypatch.setattr(likely, "compact", spy)
    compared(conn, tmp_path)
    assert kept
    for top, lse, _ in kept:
        assert len(top) <= likely.KEEP and len(lse) == len(evaluate.T_GRID)


def calibration(conn, cid, which):
    row = conn.execute("select report_json from eval_runs where comparison_id = ?",
                       (cid,)).fetchone()
    return json.loads(row[0])[which].get("calibration", {})


def test_sets_are_fitted_on_the_all_photos_pass_only(conn, tmp_path):
    cid = compared(conn, tmp_path)["comparison_id"]
    assert all("sets" in c for c in calibration(conn, cid, "all_photos").values())
    assert not any("sets" in c for c in calibration(conn, cid, "first_photo_only").values())


def test_a_comparison_can_skip_sets_entirely(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    cid = evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=tmp_path / "emb",
                           sets=False, log=lambda s: None)["comparison_id"]
    cal = calibration(conn, cid, "all_photos")
    assert cal and not any("sets" in c for c in cal.values())     # temperatures still fitted


def test_mv_compare_no_sets(conn, tmp_path, monkeypatch, capsys):
    from mycomap_vision import cli, config
    seen = {}
    monkeypatch.setattr(evaluate, "compare",
                        lambda *a, **kw: seen.update(kw) or {"comparison_id": "c"})
    monkeypatch.setattr(evaluate, "format_report", lambda r: "{}")
    monkeypatch.setattr(evaluate, "scoreboard", lambda conn, cid: [])
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "m.sqlite")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    cli.main(["compare", "--backbones", "bioclip-2", "--no-sets"])
    assert seen["sets"] is False
    cli.main(["compare", "--backbones", "bioclip-2"])
    assert seen["sets"] is True
