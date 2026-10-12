"""The FungiTastic / Danish Fungi 2020 replication (Picek et al., BVRA).

Start at replications/fungitastic/README.md (repository root). The modules:

    presets           the recipe's presets and hyperparameters; the published checkpoints
    published         their published checkpoints scored on our records, photo only
                      (`mv external ...`, `mv heldout import-external`)
    published_report  the comparison protocol for models without provisional names
    crosswalk         the GBIF Backbone + Catalogue of Life synonym crosswalk (scoring only)
    retrain           their recipe retrained on our records (`mv picek-train`,
                      `mv picek-bench`, `mv aws-launch-trainer --picek`)
    dataset           a dataset release, verified and read-only (adapter for the reader)
    release_eval      scoring on a release's split (`mv compare --release`)
    published_release the published checkpoints on a release (`mv external run`)

The old import paths (mycomap_vision.picek, .external, .external_report, .gbif) are the
same module objects, kept so saved references and older commands still resolve.
"""

__all__ = ["crosswalk", "dataset", "presets", "published", "published_release",
           "published_report", "release_eval", "retrain"]
