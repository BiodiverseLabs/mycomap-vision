"""Small synthetic stand-ins for iNat's open-data taxa.csv.gz and observations.csv.gz."""

from __future__ import annotations

import gzip
from pathlib import Path

LIFE, FUNGI, ANIMALIA, PROTOZOA, MYCETOZOA = 48460, 47170, 1, 47686, 47685

# taxon_id, ancestry, rank_level, rank, name, active
TAXA = [
    (LIFE, "", 100, "stateofmatter", "Life", "true"),
    (ANIMALIA, f"{LIFE}", 70, "kingdom", "Animalia", "true"),
    (2, f"{LIFE}/{ANIMALIA}", 20, "genus", "Animalus", "true"),
    (3, f"{LIFE}/{ANIMALIA}/2", 10, "species", "Animalus vulgaris", "true"),
    (PROTOZOA, f"{LIFE}", 70, "kingdom", "Protozoa", "true"),
    (MYCETOZOA, f"{LIFE}/{PROTOZOA}", 60, "phylum", "Mycetozoa", "true"),
    (900, f"{LIFE}/{PROTOZOA}/{MYCETOZOA}", 20, "genus", "Physarum", "true"),
    (901, f"{LIFE}/{PROTOZOA}/{MYCETOZOA}/900", 10, "species", "Physarum polycephalum", "true"),
    (FUNGI, f"{LIFE}", 70, "kingdom", "Fungi", "true"),
    (100, f"{LIFE}/{FUNGI}", 30, "family", "Amanitaceae", "true"),
    (101, f"{LIFE}/{FUNGI}/100", 20, "genus", "Amanita", "true"),
    (102, f"{LIFE}/{FUNGI}/100/101", 10, "species", "Amanita muscaria", "true"),
    (103, f"{LIFE}/{FUNGI}/100/101/102", 5, "variety", "Amanita muscaria guessowii", "true"),
    (104, f"{LIFE}/{FUNGI}/100/101", 10, "species", "Amanita orientalis", "true"),
    (105, f"{LIFE}/{FUNGI}/100/101", 10, "species", "Amanita occidentalis", "true"),
    (106, f"{LIFE}/{FUNGI}/100/101", 10, "species", "Amanita rara", "true"),
    (107, f"{LIFE}/{FUNGI}/100/101", 11, "complex", "Amanita muscaria complex", "true"),
    (110, f"{LIFE}/{FUNGI}", 30, "family", "Tricholomataceae", "true"),
    (111, f"{LIFE}/{FUNGI}/110", 20, "genus", "Lepista", "true"),
    (112, f"{LIFE}/{FUNGI}/110/111", 10, "species", "Lepista nuda", "false"),
    (113, f"{LIFE}/{FUNGI}/110", 20, "genus", "Collybia", "true"),
    (114, f"{LIFE}/{FUNGI}/110/113", 10, "species", "Collybia nuda", "true"),
    (120, f"{LIFE}/{FUNGI}", 30, "family", "Parmeliaceae", "true"),
    (121, f"{LIFE}/{FUNGI}/120", 20, "genus", "Parmelia", "true"),
    (122, f"{LIFE}/{FUNGI}/120/121", 10, "species", "Parmelia sulcata", "true"),     # a lichen
    (130, f"{LIFE}/{FUNGI}", 30, "family", "Westaceae", "true"),
    (131, f"{LIFE}/{FUNGI}/130", 20, "genus", "Westia", "true"),                        # west only
    (132, f"{LIFE}/{FUNGI}/130/131", 10, "species", "Westia pacifica", "true"),
]

HEADER = ["observation_uuid", "observer_id", "latitude", "longitude", "positional_accuracy",
          "taxon_id", "quality_grade", "observed_on", "anomaly_score"]

EAST = (40.0, -80.0)
WEST = (40.0, -120.0)        # ~3,400 km west of EAST


def obs(uuid, taxon, lat, lon, date="2025-10-01", grade="research", acc="10"):
    return [uuid, "7", str(lat), str(lon), acc, str(taxon), grade, date, "0.1"]


def standard_observations() -> list[list[str]]:
    """Amanita orientalis in the east in autumn, occidentalis in the west in spring,
    muscaria (mostly as its variety) everywhere, rara: 5 records in the west only,
    Westia pacifica: 30 in the west, background fungi in both places."""
    rows = []
    for i in range(30):
        rows.append(obs(f"e-{i}", 104, EAST[0] + (i % 5) * 0.1, EAST[1], "2025-10-01"))
        rows.append(obs(f"w-{i}", 105, WEST[0] + (i % 5) * 0.1, WEST[1], "2025-04-15"))
        rows.append(obs(f"p-{i}", 132, WEST[0], WEST[1] + (i % 3) * 0.2, "2025-05-01"))
    for i in range(20):
        lat, lon = (EAST if i % 2 else WEST)
        rows.append(obs(f"m-{i}", 103 if i % 4 else 102, lat, lon, "2025-08-01"))
    for i in range(5):
        rows.append(obs(f"r-{i}", 106, WEST[0], WEST[1], "2025-06-01"))
    for i in range(200):           # sampling effort in both places, identified to genus or less
        lat, lon = (EAST if i % 2 else WEST)
        rows.append(obs(f"b-{i}", 101 if i % 3 else 100, lat, lon, f"2025-{1 + i % 12:02d}-10"))
    return rows


def write_taxa(path: Path, rows=TAXA) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("taxon_id\tancestry\trank_level\trank\tname\tactive\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    return path


def write_observations(path: Path, rows, header=HEADER) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return path


def build_store(tmp_path: Path, rows=None, exclude: list[str] | None = None, **options):
    from mycomap_vision.occurrence import BuildOptions, OccurrenceStore, build
    tmp_path.mkdir(parents=True, exist_ok=True)
    taxa = write_taxa(tmp_path / "taxa.csv.gz")
    observations = write_observations(tmp_path / "observations.csv.gz",
                                      standard_observations() if rows is None else rows)
    files = []
    if exclude is not None:
        f = tmp_path / "exclude.txt"
        f.write_text("# held out\n" + "\n".join(exclude) + "\n", encoding="utf-8")
        files.append(f)
    out = tmp_path / "store.npz"
    meta = build(observations, taxa, out, BuildOptions(**options), files, log=lambda *a: None)
    return OccurrenceStore.load(out), meta
