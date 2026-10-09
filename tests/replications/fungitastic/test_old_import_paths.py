"""The replication moved into mycomap_vision.replications.fungitastic. Everything that named
it before must still resolve: the old import paths (the AWS trainer of an older commit,
saved references, other branches), pickles made under the old module names, the `mv`
commands and options, the heldout method names and the model kind saved in model JSON."""

import importlib
import pickle

import pytest

from mycomap_vision import cli, methods
from mycomap_vision.replications.fungitastic import (crosswalk, presets, published,
                                                     published_report, retrain)

OLD_TO_NEW = {
    "mycomap_vision.picek": retrain,
    "mycomap_vision.external": published,
    "mycomap_vision.external_report": published_report,
    "mycomap_vision.gbif": crosswalk,
}


@pytest.mark.parametrize("old", sorted(OLD_TO_NEW))
def test_an_old_import_path_is_the_moved_module_itself(old):
    assert importlib.import_module(old) is OLD_TO_NEW[old]
    package, _, leaf = old.rpartition(".")
    assert getattr(importlib.import_module(package), leaf) is OLD_TO_NEW[old]


def test_old_from_imports_still_find_every_name():
    from mycomap_vision.external import MODELS, ExternalModel, model_for
    from mycomap_vision.external_report import EXACT, format_report
    from mycomap_vision.gbif import Crosswalk, Matcher
    from mycomap_vision.picek import (CLASSIFIER_METHODS, PRESETS, ClassifierBackbone,
                                      PicekConfig, cache_of, parse_spec, train)
    assert PRESETS is presets.PRESETS and MODELS is presets.MODELS
    assert PicekConfig is presets.PicekConfig and ExternalModel is presets.ExternalModel
    assert (train, ClassifierBackbone, CLASSIFIER_METHODS) == (
        retrain.train, retrain.ClassifierBackbone, retrain.CLASSIFIER_METHODS)
    assert parse_spec("fungitastic-beit-b384@15") == ("fungitastic-beit-b384", 15)
    assert cache_of("fungitastic-beit-b384@15@440") == 440
    assert model_for("external:df20-vit-l384").hf_id == "BVRA/vit_large_patch16_384.ft_df20_384"
    assert (EXACT, format_report) == (published_report.EXACT, published_report.format_report)
    assert (Crosswalk, Matcher) == (crosswalk.Crosswalk, crosswalk.Matcher)


def test_a_patch_through_the_old_path_reaches_the_moved_code(monkeypatch):
    import mycomap_vision.picek as old
    monkeypatch.setattr(old, "create_model", "patched")
    assert retrain.create_model == "patched"


@pytest.mark.parametrize("obj,new_module,old_module", [
    (presets.PRESETS["fungitastic-beit-b384"], presets.__name__, "mycomap_vision.picek"),
    (presets.MODELS["df20-vit-l384"], presets.__name__, "mycomap_vision.external"),
])
def test_a_pickle_made_under_the_old_module_name_still_loads(obj, new_module, old_module):
    data = pickle.dumps(obj, protocol=0)        # text protocol: module names stand alone
    assert new_module.encode() in data
    assert pickle.loads(data.replace(new_module.encode(), old_module.encode())) == obj


def test_heldout_method_names_and_the_saved_model_kind_are_unchanged():
    for name in ("classifier", "classifier+month", "classifier+month-raw",
                 "classifier+month+place"):
        assert methods.METHODS[name] in retrain.CLASSIFIER_METHODS
    assert retrain.KIND == "classifier"            # meta["kind"] in data/models/<name>.json
    assert presets.PREFIX == "external:"           # heldout_runs backbone external:<short>
    assert sorted(presets.MODELS) == ["df20-vit-l384", "fungitastic-beit-b384",
                                      "fungitastic-vit-b384"]
    assert sorted(presets.PRESETS) == ["df20-vit-l384", "fungitastic-beit-b224",
                                       "fungitastic-beit-b384", "vit-b384-ce"]


@pytest.mark.parametrize("argv,options", [
    (["picek-train"], ["--preset", "--max-steps", "--effective-batch", "--val-max-photos"]),
    (["picek-bench"], ["--preset"]),
    (["aws-launch-trainer"], ["--picek", "--picek-exclude-benchmarks", "--dry-run"]),
    (["external"], ["labels", "coverage", "predict", "report", "baseline", "crosswalk"]),
    (["heldout", "import-external"], ["--results"]),
])
def test_the_mv_commands_keep_their_names_and_options(argv, options, capsys):
    with pytest.raises(SystemExit) as done:
        cli.main(argv + ["--help"])
    assert done.value.code == 0
    out = capsys.readouterr().out
    for option in options:
        assert option in out
