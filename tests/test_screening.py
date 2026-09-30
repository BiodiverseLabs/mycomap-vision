from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate, models
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.screening import screen


def test_a_failing_candidate_is_skipped_and_the_rest_are_compared(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "base", "large", store.location, "local")
    embed_photos(conn, store, Const("base"), todo, root / "base", log=lambda s: None)

    def loader(spec):
        if spec == "timm:broken":
            raise OSError("weights are gated")
        return Const(models.storage_name(spec), twist=0.3)

    result = screen(conn, store, ["timm:good", "timm:broken"], ["base"], size="large",
                    loader=loader, embeddings_root=root, log=lambda s: None)
    assert list(result.embedded) == ["timm_good"]
    assert "weights are gated" in result.failed["timm_broken"]
    assert result.compared == ["base", "timm_good"]
    board = evaluate.scoreboard(conn, result.comparison["comparison_id"])
    assert {r["backbone"] for r in board} == {"base", "timm_good"}


def test_an_already_embedded_candidate_is_not_embedded_again(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    calls = []
    loader = lambda spec: calls.append(spec) or Const("timm_x")  # noqa: E731
    screen(conn, store, ["timm:x"], [], size="large", loader=loader, embeddings_root=root,
           compare=False, log=lambda s: None)
    screen(conn, store, ["timm:x"], [], size="large", loader=loader, embeddings_root=root,
           compare=False, log=lambda s: None)
    assert calls == ["timm:x"]


def test_a_loader_that_misnames_the_backbone_is_reported_not_silently_mixed(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    result = screen(conn, store, ["timm:y"], [], size="large", loader=lambda s: Const("other"),
                    embeddings_root=tmp_path / "emb", compare=False, log=lambda s: None)
    assert "expected 'timm_y'" in result.failed["timm_y"]


def break_photo(store, rel="p/1003.png"):
    store.put(rel, b"not an image")


def test_one_unreadable_photo_is_skipped_and_counted_not_fatal(conn, tmp_path, monkeypatch):
    from mycomap_vision import screening
    store = seed_two_species(conn, tmp_path)
    break_photo(store)
    monkeypatch.setattr(screening, "SKIP_THRESHOLD", 0.2)      # 1 of 8 is under 20%
    result = screen(conn, store, ["timm:a"], [], size="large",
                    loader=lambda s: Const("timm_a"), embeddings_root=tmp_path / "emb",
                    compare=False, log=lambda s: None)
    assert result.embedded["timm_a"]["skipped"] == 1
    assert result.embedded["timm_a"]["photos"] == 7
    assert [pid for pid, _ in result.skipped["timm_a"]] == [1003]


def test_too_many_unreadable_photos_fail_the_backbone_as_a_systemic_problem(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    break_photo(store)                                          # 1 of 8 is over 1%
    result = screen(conn, store, ["timm:a"], [], size="large",
                    loader=lambda s: Const("timm_a"), embeddings_root=tmp_path / "emb",
                    compare=False, log=lambda s: None)
    assert "timm_a" not in result.embedded
    assert "1 of 8 photos unreadable" in result.failed["timm_a"]


def test_a_backbone_stopped_by_the_time_limit_is_never_counted_as_embedded(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    after, batches = [], []

    class Counted(Const):
        def encode(self, images):
            batches.append(len(images))
            return super().encode(images)
    result = screen(conn, store, ["timm:a", "timm:b"], [], size="large", batch_size=2,
                    loader=lambda s: Counted(models.storage_name(s)),
                    embeddings_root=tmp_path / "emb", compare=False,
                    should_stop=lambda: len(batches) >= 2,
                    after_embed=after.append, log=lambda s: None)
    assert result.embedded == {} and after == []
    assert set(result.stopped) == {"timm_a", "timm_b"}
