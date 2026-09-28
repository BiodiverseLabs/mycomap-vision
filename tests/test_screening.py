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
