"""One refresh: bring the reference set up to date with mycomap.org, then re-score.

    export green records -> fetch iNat details for new ones -> iNat's taxonomy for
    genera not asked about before -> download photos -> embed with every active
    backbone -> compare them on the newest weeks

This is the weekly loop (docs/PLAN.md). `scope="new"` downloads photos only for
records that joined in this refresh (a laptop keeping a sample current);
`scope="all"` downloads every photo still missing (the S3 store).

The taxonomy step (taxonomy.fetch) asks only about genera the cache has never
answered, stops asking after `taxonomy_minutes` (the next refresh carries on), and
never fails the refresh: when iNat is down it is logged and the loop goes on with
the answers it already has.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config, evaluate, inat, models, photos, records, taxonomy
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import embed_photos, photos_to_embed
from .storage import PhotoStore


TAXONOMY_MINUTES = 20.0
TAXONOMY_ATTEMPTS = 3          # tries per iNat request here (mv fetch-taxonomy: 6)


def new_genera_taxonomy(conn: sqlite3.Connection, max_minutes: float | None = TAXONOMY_MINUTES,
                        cache: Path | None = None, client: taxonomy.InatTaxa | None = None,
                        log=print) -> dict:
    """iNat's taxonomy for the genera no earlier lookup has answered (never asks again
    about one already answered: `mv fetch-taxonomy --refresh / --older-than` does that)."""
    return taxonomy.fetch(conn, cache=cache,
                          client=client or taxonomy.InatTaxa(attempts=TAXONOMY_ATTEMPTS),
                          refresh=False, older_than_days=None, max_minutes=max_minutes,
                          log=log)


@dataclass
class Steps:
    """The stages, injectable so the orchestration can be tested without the network."""
    export: Callable[[sqlite3.Connection], dict] = records.export_records
    fetch: Callable[..., dict] = inat.fetch_all
    taxonomy: Callable[..., dict] = new_genera_taxonomy
    download: Callable[..., dict] = photos.download_all
    load_backbone: Callable[[str], object] = models.load_backbone
    compare: Callable[..., dict] = evaluate.compare


@dataclass
class RefreshReport:
    export: dict = field(default_factory=dict)
    fetch: dict = field(default_factory=dict)
    taxonomy: dict = field(default_factory=dict)
    download: dict = field(default_factory=dict)
    embedded: dict[str, int] = field(default_factory=dict)
    comparison: dict | None = None


def active_backbones(conn: sqlite3.Connection) -> list[str]:
    """Backbones that already have embeddings, in the order they were first embedded."""
    conn.executescript(EMBED_SCHEMA)
    rows = conn.execute("select backbone, min(created_at) from embeddings group by backbone "
                        "order by 2").fetchall()
    return [r[0] for r in rows]


def refresh(conn: sqlite3.Connection, store: PhotoStore, scope: str = "new",
            size: str = "medium", backbones: list[str] | None = None,
            methods: list[str] | None = None, compare: bool = True, steps: Steps | None = None,
            embeddings_root=None, lookup_taxonomy: bool = True,
            taxonomy_minutes: float | None = TAXONOMY_MINUTES, log=print) -> RefreshReport:
    if scope not in ("new", "all"):
        raise ValueError("scope is 'new' or 'all'")
    steps = steps or Steps()
    report = RefreshReport()

    log("1/6 export green records from mycomap.org")
    report.export = steps.export(conn)
    log(f"    {report.export.get('new', 0):,} new, {report.export.get('removed', 0):,} removed, "
        f"{report.export.get('renamed', 0):,} renamed")

    log("2/6 fetch iNat details for records not seen before")
    report.fetch = steps.fetch(conn, log=log)

    if lookup_taxonomy:
        log("3/6 iNat taxonomy for genera not asked about before"
            + (f" (at most {taxonomy_minutes:g} min)" if taxonomy_minutes is not None else ""))
        try:
            report.taxonomy = steps.taxonomy(conn, max_minutes=taxonomy_minutes, log=log)
        except Exception as e:      # iNat down, a network error: never the refresh's end
            report.taxonomy = {"error": f"{e.__class__.__name__}: {e}"}
            log(f"    taxonomy lookup failed, carrying on with the answers already kept: "
                f"{report.taxonomy['error']}")
    else:
        report.taxonomy = {"skipped": True}
        log("3/6 iNat taxonomy: skipped")

    log(f"4/6 download photos ({scope})")
    since = report.export.get("exported_at") if scope == "new" else None
    report.download = steps.download(conn, store, size=size, first_seen_since=since, log=log)

    wanted = backbones or active_backbones(conn)
    log(f"5/6 embed new photos with {', '.join(wanted) or 'no backbones yet'}")
    for spec in wanted:
        name = models.storage_name(spec)
        todo = photos_to_embed(conn, name, size, store.location, str(config.DATA_DIR))
        report.embedded[name] = len(todo)
        if todo:
            root = (embeddings_root or config.DATA_DIR / "embeddings") / name
            embed_photos(conn, store, steps.load_backbone(spec), todo, root, log=log)

    if compare and wanted:
        log("6/6 compare on the newest weeks")
        report.comparison = steps.compare(conn, [models.storage_name(b) for b in wanted],
                                          methods or ["nearest", "species-mean"],
                                          embeddings_root=embeddings_root, log=log)
    return report
