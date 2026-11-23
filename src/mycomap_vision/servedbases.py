"""Which not-a-fungus gate and photo-type tagger each served backbone reads (Steve,
2026-11-23: serve MycoMap Vision v2 beside v1).

The gate (fungusgate.py) and the tagger (phototypes.py) read a photo's FROZEN base-model
vector, which a fine-tuned backbone forks from its own forward pass
(models.FinetunedBackbone.encode_with_base). A gate or head fits the vectors of ONE base:
v1 (fine-tune of bioclip-2-336) needs the bioclip-2-336 gate and head, v2 (fine-tune of
bioclip-2.5-336) the bioclip-2.5-336 ones. The server holds every gate and tagger it was
given, by base, and hands each backbone the pair of its own base.

Configuration (a v1-only server reads exactly what it read before):
  - gates: data/models/fungus-gate.npz + .json as before (its json names its base), and a
    gate for another base beside it as fungus-gate.<base>.npz + .json
    (fungusgate.load_all; FungusGate.save(by_base=True) writes one).
  - taggers: MV_PHOTO_TAGGER names one file as before, or several, comma-separated, one
    per base (PhotoTagger.all_from_setting). A head is served only where a gate reads its
    base (or no gate is served at all).
  - MV_CROSS_BASE_GATE (`backbone=gate base`, comma-separated; default none; Steve,
    2026-11-23: v2 keeps today's bioclip-2-336 gate until a 2.5 gate passes): the backbone
    is served with the gate of ANOTHER base, whose vector comes from a second pass over the
    photos (api.py second_pass: a served backbone that forks that base, i.e. v1, else the
    frozen base model itself), while its tagger is still its own base's head reading its
    own fork. Without the entry such a backbone is refused, as below.
  - MV_UNTAGGED_BACKBONES (comma-separated backbone names, default none): backbones that
    may answer WITHOUT photo-type tags while taggers are served for other bases. Shown in
    /api/health. There is no such setting for the gate.

Fail closed: when any gate is served, a backbone whose base has no gate is REFUSED, never
served ungated: it is left out of the models offered (/api/models, /api/v1/models, the
site's model choice, the preload), its index is never built, and /api/health and
/api/models say why. The same holds for the tagger while any tagger is served, unless
MV_UNTAGGED_BACKBONES names the backbone. A server without gates or taggers refuses
nothing, as before; a tagger file that fails its checks is not served (as before), so it
refuses nothing either.

A backbone's base is the one its fine-tune records (data/models/<backbone>.json, "base").
A backbone with no such record (a frozen model, which reads its own vectors, or a stand-in)
keeps the rule from before when one gate is served: the backbone gets that gate, and the
identifier checks the loaded model's own base at run time (identify.Identifier.encode),
refusing a photo it can't read the base vector of rather than answering ungated. With
several gates served it gets the gate of its own name, if any.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config

UNTAGGED_SETTING = "MV_UNTAGGED_BACKBONES"
CROSS_SETTING = "MV_CROSS_BASE_GATE"


class NotServed(RuntimeError):
    """A backbone refused here: no gate (or tagger) reads its base's vectors."""


def recorded_base(name: str, root: Path | None = None) -> str | None:
    """The base model a fine-tuned backbone records in data/models/<name>.json, or None
    when there is no such file (a frozen model, a stand-in) or it names none."""
    from .finetune import models_dir
    path = (root or models_dir()) / f"{name}.json"
    if not path.is_file():
        return None
    try:
        base = json.loads(path.read_text(encoding="utf-8")).get("base")
    except (OSError, ValueError):
        return None
    return str(base) if base else None


@dataclass
class Choice:
    """What one backbone is served with: its base (None: not recorded), gate, tagger, and
    why it is refused (None: served)."""
    base: str | None
    gate: object | None
    tagger: object | None
    refused: str | None = None
    untagged_allowed: bool = False
    # The base the gate reads when it isn't the backbone's own (MV_CROSS_BASE_GATE): its
    # vector then comes from a second pass over the photos, never from this backbone's fork.
    gate_base: str | None = None


@dataclass
class Readers:
    gates: dict = field(default_factory=dict)       # {base: FungusGate}
    taggers: dict = field(default_factory=dict)     # {base: PhotoTagger}, in the order named
    untagged: frozenset = frozenset()               # MV_UNTAGGED_BACKBONES
    base_of: Callable[[str], str | None] = recorded_base
    cross: dict = field(default_factory=dict)       # MV_CROSS_BASE_GATE {backbone: gate base}

    @property
    def gate(self):
        """The first gate (the default file's): what a one-base server serves."""
        return next(iter(self.gates.values()), None)

    @property
    def tagger(self):
        """The first tagger named: what a one-base server serves."""
        return next(iter(self.taggers.values()), None)

    def __post_init__(self):
        self._bases: dict = {}

    def base(self, backbone: str) -> str | None:
        """base_of(backbone), read once (a release's fine-tune records don't change)."""
        if backbone not in self._bases:
            self._bases[backbone] = self.base_of(backbone)
        return self._bases[backbone]

    def choose(self, backbone: str) -> Choice:
        base = self.base(backbone)
        if backbone in self.cross:
            return self._cross(backbone, base, self.cross[backbone])
        if base is None:
            if len(self.gates) <= 1 and len(self.taggers) <= 1:
                # As before: the one gate and tagger, checked against the loaded model.
                return Choice(None, self.gate, self.tagger)
            key = backbone                          # a frozen model reads its own vectors
        else:
            key = base
        gate, tagger = self.gates.get(key), self.taggers.get(key)
        refused = None
        if self.gates and gate is None:
            refused = (f"no not-a-fungus gate reads {key} vectors (gates served for "
                       f"{', '.join(sorted(self.gates))}): add "
                       f"fungus-gate.{key}.npz + .json beside fungus-gate.npz")
        elif self.taggers and tagger is None and backbone not in self.untagged:
            refused = (f"no photo-type tagger reads {key} vectors (taggers served for "
                       f"{', '.join(sorted(self.taggers))}): add its head to "
                       f"MV_PHOTO_TAGGER, or name {backbone} in {UNTAGGED_SETTING} to "
                       "serve it untagged")
        return Choice(base, gate, tagger, refused,
                      untagged_allowed=bool(self.taggers) and tagger is None
                      and backbone in self.untagged)

    def _cross(self, backbone: str, base: str | None, gate_base: str) -> Choice:
        """A backbone named in MV_CROSS_BASE_GATE: the gate of `gate_base`, read from a second
        pass; the tagger of the backbone's OWN base, read from its own fork. Refused when its
        base isn't recorded, when no gate reads `gate_base`, or (taggers served) when no head
        reads its own base."""
        gate = self.gates.get(gate_base)
        tagger = self.taggers.get(base) if base else None
        refused = None
        if base is None:
            refused = (f"{CROSS_SETTING} names {backbone}, but its base model isn't recorded "
                       f"(data/models/{backbone}.json)")
        elif gate is None:
            refused = (f"{CROSS_SETTING} gives {backbone} the {gate_base} gate, but no "
                       f"not-a-fungus gate reads {gate_base} vectors here")
        elif self.taggers and tagger is None and backbone not in self.untagged:
            refused = (f"no photo-type tagger reads {base} vectors (taggers served for "
                       f"{', '.join(sorted(self.taggers))}): add its head to "
                       f"MV_PHOTO_TAGGER, or name {backbone} in {UNTAGGED_SETTING} to "
                       "serve it untagged")
        return Choice(base, gate, tagger, refused,
                      untagged_allowed=bool(self.taggers) and tagger is None
                      and backbone in self.untagged,
                      gate_base=gate_base if base != gate_base else None)

    def forked_bases(self, allowed) -> set[str]:
        """The own bases of the allowed backbones served with a cross-base gate: a head for
        one of them is served though no gate reads its base."""
        return {b for b in (self.base(n) for n in self.cross if n in allowed) if b}

    def bases(self) -> list[str]:
        """Every base a gate or tagger is served for (health hides them in public)."""
        return sorted({*self.gates, *self.taggers})


def untagged_setting() -> frozenset:
    return frozenset(b.strip() for b in (config.setting(UNTAGGED_SETTING) or "").split(",")
                     if b.strip())


def cross_setting() -> tuple[dict, list[str]]:
    """MV_CROSS_BASE_GATE: `backbone=gate base`, comma-separated ({backbone: base}, and the
    entries that don't parse, which grant nothing)."""
    out: dict = {}
    errors: list[str] = []
    for part in (config.setting(CROSS_SETTING) or "").split(","):
        if not part.strip():
            continue
        backbone, sep, base = (x.strip() for x in part.partition("="))
        if not sep or not backbone or not base:
            errors.append(f"{CROSS_SETTING}: {part.strip()!r} is not backbone=gate-base")
        else:
            out[backbone] = base
    return out, errors


def pair_taggers(gates: dict, taggers: list, mismatch: Callable[[str, str], str],
                 forked: set | frozenset = frozenset()) -> tuple[dict, list[str]]:
    """The taggers that may be served beside these gates, by base, and why the others are
    not: a head reads the vector its own base's gate already computes, never a second
    tower's, so with gates served a head is served only where a gate reads its base, or
    where a backbone with a cross-base gate (MV_CROSS_BASE_GATE) forks that base itself
    (`forked`). A head whose base no served backbone forks stays unserved.
    `mismatch(tagger base, gate bases)` words the refusal (phototypes.base_mismatch)."""
    served: dict = {}
    errors: list[str] = []
    for t in taggers:
        if gates and t.base not in gates and t.base not in forked:
            errors.append(mismatch(t.base, ", ".join(sorted(gates))))
        else:
            served[t.base] = t
    return served, errors
