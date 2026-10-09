"""The .org source of every green record (iNaturalist, MO Observations, MycoPortal, ...),
from a read-only export: observation_id <TAB> source per line."""
from collections import defaultdict


def load_sources(path) -> dict[str, set[str]]:
    src = defaultdict(set)
    for line in open(path, encoding="utf-8", newline="").read().split("\n")[1:]:
        line = line.rstrip("\r")
        if line:
            o, s = line.split("\t", 1)
            src[o].add(s)
    return src


def source_of(src: dict, oid: str) -> str:
    s = src.get(oid, set())
    if s == {"iNaturalist"}:
        return "iNaturalist"
    if len(s) > 1:
        return "mixed:" + "+".join(sorted(s))
    return next(iter(s)) if s else "not-green-now"
