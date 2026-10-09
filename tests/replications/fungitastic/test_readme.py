"""replications/fungitastic/README.md is the page an outside reader lands on. It must name
what the code actually runs: every published checkpoint, every recipe preset, every
classifier method, and every module of the subpackage."""

from pathlib import Path

import pytest

from mycomap_vision import config
from mycomap_vision.replications.fungitastic import presets, retrain

README = config.REPO_ROOT / "replications" / "fungitastic" / "README.md"
PACKAGE = Path(retrain.__file__).parent


@pytest.fixture(scope="module")
def text():
    return README.read_text(encoding="utf-8")


def test_the_readme_names_every_published_checkpoint_and_its_licence(text):
    for model in presets.MODELS.values():
        assert f"`{model.hf_id}`" in text
    assert "CC BY-NC 4.0" in text


def test_the_readme_names_every_recipe_preset_and_classifier_method(text):
    for preset in presets.PRESETS:
        assert f"`{preset}`" in text
    for method in retrain.CLASSIFIER_METHODS:
        assert method.name in text


def test_the_readme_code_table_matches_the_subpackage(text):
    modules = sorted(p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py")
    assert modules == ["crosswalk.py", "presets.py", "published.py", "published_report.py",
                       "retrain.py"]
    for name in modules:
        assert f"| `{name}` |" in text
