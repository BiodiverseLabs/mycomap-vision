"""Genera whose DNA name is a guest of the fungus in the photo, not the fungus itself.

A sequence can come from a yeast living inside a puffball, or from a parasite growing on a
mushroom, while the photos show the host. Such a record teaches the wrong thing: Vision
learned "puffball-looking photo = Teunomyces" from nine of them, and answered Teunomyces
for real Pisolithus (comparison 20261006-183858-b9ca61).

Each listed genus is in one group:

- hidden:   lives inside the fungus and is never what the photo shows. Always left out.
- on_host:  a parasite visible on a mushroom, but the host fills the photo. Left out
            unless MV_KEEP_ON_HOST=1.
- visible:  a mould or other fungus that is itself the subject. Kept; listed so a reader
            can see it was considered.

A record left out here is left out of reference sets, comparisons, training and the
served index alike (evaluate.load_records), and is not scored as an advance prediction.
Edit the lists below to change the rule; `mv guests` shows what it covers in the data.
"""

from __future__ import annotations

import os

HIDDEN = {
    "Teunomyces": "yeast living inside puffballs and other fruit bodies",
    "Candida": "yeast",
    "Meyerozyma": "yeast",
    "Rhodotorula": "yeast",
    "Vishniacozyma": "yeast",
    "Saitozyma": "yeast",
    "Geotrichum": "yeast-like mould",
    "Dipodascus": "yeast-like",
}

ON_HOST = {
    "Spinellus": "pin mould on Mycena and other mushrooms",
    "Syzygites": "pin mould on rotting mushrooms",
    "Mycogone": "parasite on Agaricus and others",
    "Sepedonium": "parasite on boletes",
    "Cladobotryum": "cobweb mould on mushrooms",
}

VISIBLE = {
    "Trichoderma": "green mould, the subject", "Penicillium": "mould, the subject",
    "Aspergillus": "mould, the subject", "Botrytis": "grey mould, the subject",
    "Pilobolus": "hat thrower, the subject", "Phycomyces": "pin mould, the subject",
    "Modicella": "the subject", "Purpureocillium": "insect pathogen, the subject",
    "Lecanicillium": "insect pathogen, the subject", "Microbotryum": "smut, the subject",
    "Taphrina": "leaf curl, the subject",
}

GROUPS = {"hidden": HIDDEN, "on_host": ON_HOST, "visible": VISIBLE}


def keep_on_host() -> bool:
    return os.environ.get("MV_KEEP_ON_HOST", "").strip().lower() in ("1", "true", "yes")


def group_of(genus: str | None) -> str | None:
    g = (genus or "").strip()
    for group, genera in GROUPS.items():
        if g in genera:
            return group
    return None


def excluded(genus: str | None) -> str | None:
    """The group that leaves a record of this genus out, or None when it stays."""
    group = group_of(genus)
    if group == "hidden" or (group == "on_host" and not keep_on_host()):
        return group
    return None
