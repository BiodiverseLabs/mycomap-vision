"""The FungiTastic / DF20 replication's settings, in one place for a reader to check.

Two halves (README.md in replications/fungitastic at the repo root explains both):

- PRESETS and PicekConfig: the Picek group's training recipe, as we retrain it on our
  records (retrain.py). Sources: the Hugging Face config of
  BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384, BohemianVRA/FungiTastic
  baselines/closed_set/train.py and the fgvc library (BohemianVRA/FGVC-Tools).
- MODELS: their published checkpoints, run as they ship them on our records
  (published.py). Licence CC BY-NC 4.0: research use only.

Nothing here reads data or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ... import config


# --- the retrain recipe (retrain.py) --------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    timm_name: str
    image_size: int
    loss: str                  # "seesaw" or "ce"
    lr: float
    epochs: int
    augment: str               # "vit_heavy" or "light"
    note: str


PRESETS: dict[str, Preset] = {
    # FungiTastic's best closed-set model (HF BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384).
    "fungitastic-beit-b384": Preset("beit_base_patch16_384.in22k_ft_in22k_in1k", 384, "seesaw",
                                    0.01, 50, "vit_heavy",
                                    "FungiTastic best: BEiT-B/16 384, Seesaw, SGD 0.01, heavy aug"),
    # The same at 224 px (FungiTastic reports 224 models too): the cheaper variant.
    "fungitastic-beit-b224": Preset("beit_base_patch16_224.in22k_ft_in22k_in1k", 224, "seesaw",
                                    0.01, 50, "vit_heavy",
                                    "FungiTastic recipe at 224 px (BEiT-B/16 224)"),
    # FungiTastic's ViT-B variant: plain cross-entropy.
    "vit-b384-ce": Preset("vit_base_patch16_384.augreg_in21k_ft_in1k", 384, "ce", 0.001, 50,
                          "vit_heavy", "ViT-B/16 384, cross-entropy, SGD 0.001, heavy aug"),
    # DF20's production model (Danish Fungi / Atlas of Danish Fungi "FungiVision").
    "df20-vit-l384": Preset("vit_large_patch16_384.augreg_in21k_ft_in1k", 384, "ce", 0.005, 100,
                            "light", "DF20 production: ViT-L/16 384, cross-entropy, light aug"),
}


@dataclass
class PicekConfig:
    preset: str = "fungitastic-beit-b384"
    epochs: int | None = None          # None: the preset's
    lr: float | None = None            # None: the preset's
    effective_batch: int = 256
    micro_batch: int = 16              # photos per forward pass; accumulated to effective_batch
    momentum: float = 0.9
    weight_decay: float = 0.0
    plateau_factor: float = 0.9
    plateau_patience: int = 1
    plateau_eps: float = 1e-6
    seesaw_p: float = 0.8
    seesaw_q: float = 2.0
    val_days: int = 28                 # the validation slice: the last days before the cutoff
    val_max_photos: int | None = None  # cap the validation pass (smoke tests)
    # Gradient checkpointing: None = on only when the GPU has under 12 GB (it costs ~20% of
    # the speed: 44 vs 55 photos/s for BEiT-B 384 on the laptop's 8 GB RTX 4070, where
    # micro-batch 16 without it peaks at 4.5 GB and 32 spills over 8 GB).
    grad_checkpointing: bool | None = None
    amp: bool = True
    workers: int = 6
    seed: int = 0
    max_steps: int | None = None       # cap on optimizer steps (smoke tests)
    cell_degrees: float = 4.0          # the place prior's grid (our extension)
    # Training photos pre-resized once to this shorter side on local disk (cache_photos),
    # so 4 vCPUs can keep the GPU fed; None (the default) reads the originals. Validation
    # always reads the originals, like the embedding at test time.
    cache_px: int | None = None
    cache_dir: str | None = None       # default: <data>/picek-cache/<px>
    # Which benchmark records to leave out (EXCLUDE_MODES). "match": exactly what a Vision
    # fine-tune leaves out (sealed benchmarks, evaluate.load_records), so the two train on
    # the same records; "all": also every record of every benchmark, released or not.
    exclude_benchmarks: str = "match"
    exclude_ids_file: str | None = None   # "all" on a trainer instance: the ids, shipped
    expect_labels_hash: str | None = None # refuse to train on other labels than at launch

    def resolved(self) -> "PicekConfig":
        p = PRESETS[self.preset]
        return replace(self, epochs=self.epochs or p.epochs, lr=self.lr or p.lr)


# Augmentation and evaluation normalisation (their model cards): mean = std = 0.5.
MEAN = STD = (0.5, 0.5, 0.5)


# Which held-out benchmark records a retrain leaves out (--picek-exclude-benchmarks).
EXCLUDE_MODES = {
    "match": "only sealed benchmarks (as every Vision model; evaluate.load_records)",
    "all": "also every record of every held-out benchmark, sealed, released or not",
}


# --- the published checkpoints (published.py) ---------------------------------------------

PREFIX = "external:"
METHOD = "mean-logits"          # the authors' observation rule: mean of the photos' logits
LICENCE = "CC-BY-NC-4.0"
TOP_K = 10


def external_dir() -> Path:
    return config.DATA_DIR / "external"


@dataclass(frozen=True)
class ExternalModel:
    short: str                   # backbone is external:<short>
    hf_id: str
    architecture: str            # the timm architecture its config.json names
    num_classes: int
    dataset: str
    metadata: str                # training metadata CSV, under <data>/external/metadata
    id_column: str
    input_size: int = 384
    mean: tuple = (0.5, 0.5, 0.5)
    std: tuple = (0.5, 0.5, 0.5)

    @property
    def backbone(self) -> str:
        return PREFIX + self.short

    def folder(self, root: Path | None = None) -> Path:
        return (root or external_dir()) / "bvra" / self.hf_id.split("/", 1)[1]


MODELS = {m.short: m for m in (
    ExternalModel("fungitastic-beit-b384", "BVRA/beit_base_patch16_384.in1k_ft_fungitastic_384",
                  "beit_base_patch16_384.in22k_ft_in22k_in1k", 2829,
                  "FungiTastic (Danish Fungi 2024 observations; its config.yaml says DF24)",
                  "FungiTastic/FungiTastic-Train.csv", "category_id"),
    ExternalModel("fungitastic-vit-b384", "BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384",
                  "vit_base_patch16_384.augreg_in21k_ft_in1k", 2829, "FungiTastic",
                  "FungiTastic/FungiTastic-Train.csv", "category_id"),
    ExternalModel("df20-vit-l384", "BVRA/vit_large_patch16_384.ft_df20_384",
                  "vit_large_patch16_384", 1604, "Danish Fungi 2020 (Production, DF20_FIX)",
                  "DanishFungi2024-train.csv", "class_id"),
)}
