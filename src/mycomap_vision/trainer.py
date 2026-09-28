"""The full-data run on a GPU instance, and bringing its results home.

`run_job` runs on the instance (see aws.launch_trainer): it embeds every large
photo in S3 with each backbone, compares every backbone x method on the newest
weeks, and uploads the embeddings, the manifest, the reports and a result.json
to s3://<bucket>/runs/<run>/. `merge_results` runs on the laptop: the run's
backbones replace the local (sample) embeddings, which are archived rather than
deleted, and the run's scoreboard rows are added once.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Callable

from . import config, evaluate
from .embed import SCHEMA as EMBED_SCHEMA
from .manifest import snapshot
from .screening import screen

RESULT_FILE = "result.json"


def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}/"


def clear_embeddings(conn: sqlite3.Connection) -> int:
    """Forget every embedding row. The manifest arrives from the laptop with rows for
    shard files that stay on the laptop; left in place they would make the instance skip
    those photos and then fail to load them."""
    conn.executescript(EMBED_SCHEMA)
    with conn:
        return conn.execute("delete from embeddings").rowcount


def run_job(conn: sqlite3.Connection, store, backbones: list[str], methods: list[str],
            upload: Callable[[Path, str], None], run_id: str, size: str = "large",
            test_days: int = 28, batch_size: int = 64, readers: int = 16,
            loader=None, data_dir: Path | None = None, log=print) -> dict:
    """Embed, compare and upload. `upload(local_path, key)` copies one file to the bucket."""
    data_dir = data_dir or config.DATA_DIR
    root = data_dir / "embeddings"
    prefix = run_prefix(run_id)
    cleared = clear_embeddings(conn)
    log(f"cleared {cleared:,} embedding rows brought from the laptop")

    def upload_backbone(name: str) -> None:
        files = sorted(p for p in (root / name).glob("*.npy"))
        for p in files:
            upload(p, f"{prefix}embeddings/{name}/{p.name}")
        log(f"[{name}] uploaded {len(files)} shard files")

    kw = {"loader": loader} if loader is not None else {}
    result = screen(conn, store, backbones, [], size=size, methods=methods,
                    batch_size=batch_size, embeddings_root=root, readers=readers,
                    after_embed=upload_backbone, compare=False, log=log, **kw)
    comparison = None
    if result.embedded:
        comparison = evaluate.compare(conn, list(result.embedded), methods, test_days=test_days,
                                      embeddings_root=root, log=log)
        report = data_dir / "reports" / f"compare-{comparison['comparison_id']}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(evaluate.format_report(comparison), encoding="utf-8")
        upload(report, f"{prefix}reports/{report.name}")
    out = {"run_id": run_id, "size": size, "methods": methods,
           "embedded": result.embedded, "failed": result.failed,
           "comparison_id": comparison["comparison_id"] if comparison else None,
           "code_version": config.code_version()}
    snap = snapshot(conn, data_dir / "manifest-out.sqlite")
    upload(snap, f"{prefix}manifest-out.sqlite")
    # result.json goes last: its presence means everything else is in place.
    res = data_dir / RESULT_FILE
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")
    upload(res, prefix + RESULT_FILE)
    return out


def merge_results(conn: sqlite3.Connection, remote_path: Path, staged: Path, result: dict,
                  data_dir: Path | None = None) -> dict:
    """Bring a finished run home.

    `staged` holds the run's embeddings as <backbone>/shard-*.npy. For each backbone
    the run embedded, the local folder moves to embeddings-archive/<backbone>-before-<run>
    and the run's shards and rows take its place. Other backbones are left alone. The
    run's embedding speeds and its comparison's scoreboard rows are added once, so
    pulling the same run twice changes nothing.
    """
    data_dir = data_dir or config.DATA_DIR
    run_id = result["run_id"]
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    replaced, archived = [], []
    conn.execute("attach database ? as remote", (str(remote_path),))
    try:
        for name in result.get("embedded", {}):
            src = staged / name
            if not src.is_dir():
                raise FileNotFoundError(f"run {run_id} has no embeddings staged for {name}")
            dest = data_dir / "embeddings" / name
            already = dest.is_dir() and (dest / f".run-{run_id}").exists()
            if already:
                continue
            if dest.exists():
                archive = data_dir / "embeddings-archive" / f"{name}-before-{run_id}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(archive))
                archived.append(str(archive))
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            with conn:
                conn.execute("delete from embeddings where backbone = ?", (name,))
                conn.execute("insert into embeddings select * from remote.embeddings "
                             "where backbone = ?", (name,))
                conn.execute("insert into embed_runs select * from remote.embed_runs "
                             "where backbone = ? except select * from main.embed_runs", (name,))
            (dest / f".run-{run_id}").write_text("", encoding="utf-8")
            replaced.append(name)
        cid = result.get("comparison_id")
        scored = 0
        if cid and not conn.execute("select 1 from eval_runs where comparison_id = ?",
                                    (cid,)).fetchone():
            cols = [c for c in evaluate.SCOREBOARD_COLUMNS if c != "id"] + ["report_json"]
            with conn:
                scored = conn.execute(
                    f"insert into eval_runs ({', '.join(cols)}) select {', '.join(cols)} "
                    "from remote.eval_runs where comparison_id = ? order by id", (cid,)).rowcount
    finally:
        conn.execute("detach database remote")
    return {"run_id": run_id, "replaced": replaced, "archived": archived,
            "scoreboard_rows_added": scored, "comparison_id": result.get("comparison_id")}
