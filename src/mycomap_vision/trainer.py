"""The full-data run on a GPU instance, and bringing its results home.

`run_job` runs on the instance (see aws.launch_trainer): it embeds every large
photo in S3 with each backbone, optionally fine-tunes some of them on the
reference records (finetune.py) and embeds with the result, compares every
backbone x method on the newest weeks, and uploads the embeddings, fine-tuned
weights, the manifest, the reports and a result.json to s3://<bucket>/runs/<run>/. `merge_results` runs on the laptop: the run's
backbones replace the local (sample) embeddings, which are archived rather than
deleted, and the run's scoreboard rows are added once.
"""

from __future__ import annotations

import json
import random
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


def restrict_to_sample(conn: sqlite3.Connection, n: int, store_location: str, size: str,
                       seed: int = 0) -> int:
    """Keep only n random records (with photos in the store) in this manifest copy, for a
    cheap rehearsal of the whole run. Only for the instance's own copy: it deletes the
    other records. Returns how many were kept."""
    ids = [r[0] for r in conn.execute(
        "select distinct r.observation_id from records r "
        "join observation_photos op on op.observation_id = r.observation_id "
        "join photo_copies c on c.photo_id = op.photo_id "
        "where c.store = ? and c.size = ? and r.north_america = 1 and r.label_conflict = 0 "
        "order by r.observation_id", (store_location, size))]
    keep = random.Random(seed).sample(ids, min(n, len(ids)))
    with conn:
        conn.execute("create temp table if not exists keep_records (id text primary key)")
        conn.execute("delete from keep_records")
        conn.executemany("insert into keep_records values (?)", [(i,) for i in keep])
        conn.execute("delete from records where observation_id not in "
                     "(select id from keep_records)")
    return len(keep)


def finetuned_name(base: str, run_id: str) -> str:
    return f"{base}-ft-{run_id}"


def run_job(conn: sqlite3.Connection, store, backbones: list[str], methods: list[str],
            upload: Callable[[Path, str], None], run_id: str, size: str = "large",
            test_days: int = 28, batch_size: int = 64, readers: int = 16,
            loader=None, data_dir: Path | None = None, finetune: list[str] | None = None,
            finetuner=None, sample_records: int | None = None, log=print) -> dict:
    """Embed, fine-tune, compare and upload. `upload(local_path, key)` copies one file to
    the bucket. `finetune` names backbones (also in `backbones`) to fine-tune;
    `finetuner(conn, base, name, embeddings_root, models_dir)` does it (default:
    finetune.finetune with its default settings) and returns the model's metadata."""
    data_dir = data_dir or config.DATA_DIR
    root = data_dir / "embeddings"
    prefix = run_prefix(run_id)
    cleared = clear_embeddings(conn)
    log(f"cleared {cleared:,} embedding rows brought from the laptop")
    kept = None
    if sample_records:
        kept = restrict_to_sample(conn, sample_records, store.location, size)
        log(f"REHEARSAL: kept {kept:,} random records; results won't be merged home")

    def upload_backbone(name: str) -> None:
        files = sorted(p for p in (root / name).glob("*.npy"))
        for p in files:
            upload(p, f"{prefix}embeddings/{name}/{p.name}")
        log(f"[{name}] uploaded {len(files)} shard files")

    kw = {"loader": loader} if loader is not None else {}
    result = screen(conn, store, backbones, [], size=size, methods=methods,
                    batch_size=batch_size, embeddings_root=root, readers=readers,
                    after_embed=upload_backbone, compare=False, log=log, **kw)
    finetuned, ft_failed = {}, {}
    for base in finetune or []:
        from . import models
        base = models.storage_name(base)
        name = finetuned_name(base, run_id)
        if base not in result.embedded:
            ft_failed[name] = f"{base} was not embedded"
            continue
        try:
            meta = (finetuner or _default_finetuner(store, size, test_days, log))(
                conn, base, name, root, data_dir / "models")
            for suffix in (".pt", ".json"):
                upload(data_dir / "models" / f"{name}{suffix}", f"{prefix}models/{name}{suffix}")
            finetuned[name] = meta
        except Exception as e:  # keep the rest of the run
            ft_failed[name] = f"{e.__class__.__name__}: {e}"
            log(f"[{name}] FINE-TUNING FAILED: {ft_failed[name]}")
    if finetuned:
        ft = screen(conn, store, list(finetuned), [], size=size, methods=methods,
                    batch_size=batch_size, embeddings_root=root, readers=readers,
                    after_embed=upload_backbone, compare=False, log=log, **kw)
        result.embedded.update(ft.embedded)
        result.failed.update(ft.failed)
    result.failed.update(ft_failed)
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
           "finetuned": {n: {k: m.get(k) for k in ("base", "trained_through", "records",
                                                    "photos", "species", "steps", "minutes",
                                                    "final_loss")}
                         for n, m in finetuned.items()},
           "comparison_id": comparison["comparison_id"] if comparison else None,
           "sample_records": kept,
           "code_version": config.code_version()}
    snap = snapshot(conn, data_dir / "manifest-out.sqlite")
    upload(snap, f"{prefix}manifest-out.sqlite")
    # result.json goes last: its presence means everything else is in place.
    res = data_dir / RESULT_FILE
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")
    upload(res, prefix + RESULT_FILE)
    return out


def _default_finetuner(store, size, test_days, log):
    def run(conn, base, name, embeddings_root, models_dir):
        from .finetune import finetune
        return finetune(conn, base, store, size, name, test_days=test_days,
                        embeddings_root=embeddings_root, out_dir=models_dir, log=log)
    return run


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
    if result.get("sample_records"):
        raise ValueError(f"run {run_id} was a rehearsal on {result['sample_records']:,} records; "
                         "its results are not merged (they would replace real embeddings)")
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    replaced, archived = [], []
    finetuned = list(result.get("finetuned", {}))
    # Fine-tuned weights first: their embeddings are no use without them.
    for name in finetuned:
        dest = data_dir / "models"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in (".pt", ".json"):
            src = staged.parent / "models" / f"{name}{suffix}"
            if not src.is_file():
                raise FileNotFoundError(f"run {run_id} has no {src.name} staged")
            shutil.copyfile(src, dest / src.name)
    conn.execute("attach database ? as remote", (str(remote_path),))
    try:
        if finetuned:
            with conn:
                conn.execute("insert or replace into finetunes select * from remote.finetunes "
                             f"where name in ({','.join('?' * len(finetuned))})", finetuned)
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
