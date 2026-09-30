"""The full-data run on a GPU instance, and bringing its results home.

`run_job` runs on the instance (see aws.launch_trainer). It works through stages
in a fixed order (plan_stages): each backbone to fine-tune is embedded, then
fine-tuned, then embedded again with the result, before any other backbone
starts, so the stages that matter most finish first. Every stage that finishes is
uploaded to s3://<bucket>/runs/<run>/ at once (embeddings, fine-tuned weights,
the list of skipped photos, a small index.sqlite with what the merge needs) and
recorded in progress.json. At the time limit the job stops between batches or
steps, well before the hard kill, and says so in progress.json. When every stage
has had its turn it compares every backbone x method on the newest weeks and
uploads the report, the manifest and result.json (last: its presence means the
run is complete).

`merge_results` runs on the laptop: each backbone the run embedded completely
replaces the local (sample) embeddings, which are archived rather than deleted,
and the run's scoreboard rows are added once. A stopped run brings home the
stages it finished; a backbone it didn't finish never replaces anything.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import config, evaluate
from .embed import SCHEMA as EMBED_SCHEMA
from .manifest import snapshot
from .screening import Stopped, screen

RESULT_FILE = "result.json"
PROGRESS_FILE = "progress.json"
INDEX_FILE = "index.sqlite"     # embeddings index, speeds, fine-tunes, scoreboard
INDEX_TABLES = ("embeddings", "embed_runs", "finetunes", "eval_runs")

# --- how long a run takes ------------------------------------------------------
# Photos per second at LARGE photos, measured on the LAPTOP GPU (Sep 2026), not on
# the instance's L4. An L4 is not expected to be faster, so estimates multiply the
# time by L4_FACTOR to stay on the safe side. Replace these with the instance's own
# speeds (progress.json records them per stage) after the first run.
EMBED_RATES = {"bioclip-2": 55.0, "dinov3-l16-512": 18.8}
FINETUNE_RATES = {"bioclip-2": 83.0}   # photos seen per second while training
UNMEASURED_EMBED_RATE = 15.0           # a backbone never timed here: assume a slow one
UNMEASURED_FINETUNE_RATE = 40.0
L4_FACTOR = 1.5
SETUP_HOURS = 0.5                      # boot, installing packages, model weights
COMPARE_HOURS = 0.5                    # the comparison and the final uploads
STOP_MARGIN_HOURS = 0.75               # the job stops itself this long before the hard kill


def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}/"


def finetuned_name(base: str, run_id: str) -> str:
    return f"{base}-ft-{run_id}"


@dataclass
class Stage:
    kind: str                 # "embed" or "finetune"
    name: str                 # what the stage produces (a backbone's storage name)
    spec: str                 # what the loader is given
    base: str | None = None   # for a fine-tune and its embedding: the backbone it starts from


def plan_stages(backbones: list[str], finetune: list[str] | None, run_id: str) -> list[Stage]:
    """The order a run works in. Each backbone to fine-tune comes first, followed directly
    by its fine-tune and the fine-tuned model's embedding; the other backbones follow. If
    time runs out, what's left undone is the extra backbones, not the fine-tune."""
    from .models import storage_name
    specs = {}
    for b in backbones:
        specs.setdefault(storage_name(b), b)
    ft = []
    for f in finetune or []:
        n = storage_name(f)
        if n not in specs:
            raise ValueError(f"fine-tuning {f} needs it in the backbones too")
        if n not in ft:
            ft.append(n)
    stages = []
    for n in ft + [n for n in specs if n not in ft]:
        stages.append(Stage("embed", n, specs[n]))
        if n in ft:
            name = finetuned_name(n, run_id)
            stages.append(Stage("finetune", name, name, base=n))
            stages.append(Stage("embed", name, name, base=n))
    return stages


def estimate(stages: list[Stage], photos: int, epochs: float | None = None,
             factor: float = L4_FACTOR) -> dict:
    """Hours a run should take: photos / rate per stage (EMBED_RATES, FINETUNE_RATES,
    laptop-measured) times `factor`, plus setup and the comparison. Fine-tuning counts
    every photo `epochs` times (it trains on the reference records only, a little less)."""
    if epochs is None:
        from .finetune import FinetuneConfig
        epochs = FinetuneConfig.epochs
    rows, total = [], SETUP_HOURS + COMPARE_HOURS
    for s in stages:
        key = s.base or s.name
        if s.kind == "embed":
            rate = EMBED_RATES.get(key)
            work = photos
            fallback = UNMEASURED_EMBED_RATE
        else:
            rate = FINETUNE_RATES.get(key)
            work = photos * epochs
            fallback = UNMEASURED_FINETUNE_RATE
        hours = work / (rate or fallback) / 3600 * factor
        total += hours
        rows.append({"stage": f"{s.kind} {s.name}", "photos_per_second": rate or fallback,
                     "measured": rate is not None, "hours": round(hours, 1)})
    return {"photos": photos, "factor": factor, "stages": rows,
            "setup_hours": SETUP_HOURS, "compare_hours": COMPARE_HOURS,
            "total_hours": round(total, 1)}


def format_estimate(est: dict) -> str:
    lines = [f"Estimate for {est['photos']:,} photos (laptop speeds x {est['factor']} for "
             "the L4; conservative):"]
    for r in est["stages"]:
        lines.append(f"  {r['stage']:<40} {r['hours']:>5.1f} h at {r['photos_per_second']}/s"
                     + ("" if r["measured"] else " (NOT MEASURED: assumed)"))
    lines.append(f"  setup {est['setup_hours']} h + comparison {est['compare_hours']} h "
                 f"= {est['total_hours']} h in all")
    return "\n".join(lines)


def deadline(hours: float | None, clock: Callable[[], float] = time.monotonic
             ) -> Callable[[], bool] | None:
    """should_stop() for a job that must stop `hours` from now (None: never)."""
    if hours is None:
        return None
    end = clock() + hours * 3600
    return lambda: clock() >= end


# --- on the instance -------------------------------------------------------------

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


def write_index(conn: sqlite3.Connection, dest: Path) -> Path:
    """The tables a merge reads (INDEX_TABLES), in a small file of their own, so each
    finished stage can be uploaded without the whole manifest."""
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn.execute("attach database ? as idx", (str(tmp),))
    try:
        with conn:
            for t in INDEX_TABLES:
                conn.execute(f"create table idx.{t} as select * from main.{t}")
    finally:
        conn.execute("detach database idx")
    tmp.replace(dest)
    return dest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Progress:
    """progress.json: where the run is, rewritten and uploaded after every stage."""

    def __init__(self, path: Path, key: str, upload, **info):
        self.path, self.key, self.upload = path, key, upload
        self.data = {"state": "running", "started_at": _now(), "updated_at": _now(), **info}

    def stage(self, i: int, **fields) -> None:
        self.data["stages"][i].update(fields)
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        self.upload(self.path, self.key)


def run_job(conn: sqlite3.Connection, store, backbones: list[str], methods: list[str],
            upload: Callable[[Path, str], None], run_id: str, size: str = "large",
            test_days: int = 28, batch_size: int = 64, readers: int = 16,
            loader=None, data_dir: Path | None = None, finetune: list[str] | None = None,
            finetuner=None, sample_records: int | None = None,
            should_stop: Callable[[], bool] | None = None, log=print) -> dict:
    """Embed, fine-tune, compare and upload, stage by stage (plan_stages). `upload(path,
    key)` copies one file to the bucket. `finetuner(conn, base, name, embeddings_root,
    models_dir)` fine-tunes (default: finetune.finetune with its default settings) and
    returns the model's metadata. `should_stop()` (the time limit, see deadline) is asked
    between batches, steps and stages; once it says stop, the stage under way is
    abandoned (never uploaded), no new stage starts, and the run ends "stopped"."""
    data_dir = data_dir or config.DATA_DIR
    root = data_dir / "embeddings"
    prefix = run_prefix(run_id)
    stages = plan_stages(backbones, finetune, run_id)
    cleared = clear_embeddings(conn)
    log(f"cleared {cleared:,} embedding rows brought from the laptop")
    kept = None
    if sample_records:
        kept = restrict_to_sample(conn, sample_records, store.location, size)
        log(f"REHEARSAL: kept {kept:,} random records; results won't be merged home")
    log("stages: " + ", ".join(f"{s.kind} {s.name}" for s in stages))
    progress = Progress(data_dir / PROGRESS_FILE, prefix + PROGRESS_FILE, upload,
                        run_id=run_id, code_version=config.code_version(), size=size,
                        methods=methods, sample_records=kept,
                        stages=[{**asdict(s), "status": "pending"} for s in stages])
    progress.save()
    stopping = should_stop or (lambda: False)
    kw = {"loader": loader} if loader is not None else {}

    def publish(files: list[tuple[Path, str]]) -> None:
        """A finished stage's files, then the index; progress.json is written after."""
        for path, key in files:
            upload(path, key)
        upload(write_index(conn, data_dir / INDEX_FILE), prefix + INDEX_FILE)

    embedded, failed, stopped, finetuned, skipped_counts = {}, {}, {}, {}, {}
    for i, s in enumerate(stages):
        # A stage whose input didn't come about: stopped with it, or failed with it.
        needs = None
        if s.kind == "finetune" and s.base not in embedded:
            needs, why = s.base, f"{s.base} was not embedded"
        elif s.kind == "embed" and s.base and s.name not in finetuned:
            needs, why = s.name, f"{s.name} was not fine-tuned"
        if needs is not None:
            if needs in stopped:
                why = f"not started (time limit): {why}"
                stopped.setdefault(s.name, why)
                progress.stage(i, status="stopped", error=why)
            else:
                if s.kind == "finetune":
                    failed[s.name] = why
                progress.stage(i, status="skipped", error=why)
            continue
        if stopping():
            stopped[s.name] = "not started (time limit)"
            progress.stage(i, status="stopped", error=stopped[s.name])
            continue
        progress.stage(i, status="running", started_at=_now())
        t0 = time.monotonic()
        if s.kind == "embed":
            r = screen(conn, store, [s.spec], [], size=size, methods=methods,
                       batch_size=batch_size, embeddings_root=root, readers=readers,
                       compare=False, should_stop=should_stop, log=log, **kw)
            bad = r.skipped.get(s.name, [])
            skip_file = None
            if bad:
                skip_file = data_dir / "skipped" / f"{s.name}.tsv"
                skip_file.parent.mkdir(parents=True, exist_ok=True)
                skip_file.write_text("photo_id\treason\n" + "".join(
                    f"{pid}\t{why}\n" for pid, why in bad), encoding="utf-8")
                upload(skip_file, f"{prefix}skipped/{s.name}.tsv")
            skipped_counts[s.name] = len(bad)
            if s.name in r.embedded:
                info = r.embedded[s.name]
                shards = sorted((root / s.name).glob("*.npy"))
                publish([(p, f"{prefix}embeddings/{s.name}/{p.name}") for p in shards])
                log(f"[{s.name}] uploaded {len(shards)} shard files")
                embedded[s.name] = info
                progress.stage(i, status="done", finished_at=_now(), **info)
            elif s.name in r.stopped:
                stopped[s.name] = r.stopped[s.name]
                progress.stage(i, status="stopped", error=stopped[s.name],
                               skipped=len(bad), seconds=round(time.monotonic() - t0, 1))
            else:
                failed[s.name] = r.failed.get(s.name, "unknown failure")
                progress.stage(i, status="failed", error=failed[s.name], skipped=len(bad))
        else:
            try:
                meta = (finetuner or _default_finetuner(store, size, test_days, log,
                                                        should_stop))(
                    conn, s.base, s.name, root, data_dir / "models")
                publish([(data_dir / "models" / f"{s.name}{suffix}",
                          f"{prefix}models/{s.name}{suffix}") for suffix in (".pt", ".json")])
                finetuned[s.name] = meta
                progress.stage(i, status="done", finished_at=_now(),
                               seconds=round(time.monotonic() - t0, 1),
                               **{k: meta.get(k) for k in FINETUNE_KEYS})
            except Stopped as e:
                stopped[s.name] = str(e)
                log(f"[{s.name}] FINE-TUNING STOPPED: {e}")
                progress.stage(i, status="stopped", error=str(e))
            except Exception as e:  # keep the rest of the run
                failed[s.name] = f"{e.__class__.__name__}: {e}"
                log(f"[{s.name}] FINE-TUNING FAILED: {failed[s.name]}")
                progress.stage(i, status="failed", error=failed[s.name])
    comparison = None
    if stopped:
        progress.data["comparison"] = ("skipped: the run was stopped by its time limit; "
                                       "run mv compare on the laptop after pulling")
    elif embedded:
        progress.data["state"] = "comparing"
        progress.save()
        comparison = evaluate.compare(conn, list(embedded), methods, test_days=test_days,
                                      embeddings_root=root, log=log)
        report = data_dir / "reports" / f"compare-{comparison['comparison_id']}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(evaluate.format_report(comparison), encoding="utf-8")
        publish([(report, f"{prefix}reports/{report.name}")])
    out = {"run_id": run_id, "size": size, "methods": methods, "state":
           "stopped" if stopped else "finished",
           "embedded": embedded, "failed": failed, "stopped": stopped,
           "skipped_photos": skipped_counts,
           "finetuned": {n: {k: m.get(k) for k in FINETUNE_KEYS} for n, m in finetuned.items()},
           "comparison_id": comparison["comparison_id"] if comparison else None,
           "sample_records": kept, "code_version": config.code_version(),
           "stages": progress.data["stages"]}
    snap = snapshot(conn, data_dir / "manifest-out.sqlite")
    upload(snap, f"{prefix}manifest-out.sqlite")
    progress.data.update(state=out["state"], comparison_id=out["comparison_id"],
                         finished_at=_now())
    progress.save()
    # result.json goes last: its presence means everything else is in place.
    res = data_dir / RESULT_FILE
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")
    upload(res, prefix + RESULT_FILE)
    return out


FINETUNE_KEYS = ("base", "trained_through", "records", "photos", "species", "steps",
                 "minutes", "final_loss")


def _default_finetuner(store, size, test_days, log, should_stop=None):
    def run(conn, base, name, embeddings_root, models_dir):
        from .finetune import finetune
        return finetune(conn, base, store, size, name, test_days=test_days,
                        embeddings_root=embeddings_root, out_dir=models_dir, log=log,
                        should_stop=should_stop)
    return run


# --- bringing it home -----------------------------------------------------------

def run_summary(doc: dict) -> dict:
    """What can come home from a run, from its result.json (complete) or its
    progress.json (still running, stopped, or killed): the backbones it embedded
    completely, the models it fine-tuned, and the stages that are missing, with why."""
    if "stages" not in doc:                 # a result.json from before stages were recorded
        return {**doc, "complete": True, "missing": []}
    embedded, finetuned, missing = {}, {}, []
    for s in doc["stages"]:
        if s.get("status") == "done":
            if s["kind"] == "embed":
                embedded[s["name"]] = {k: s.get(k) for k in ("photos", "skipped",
                                                              "per_second", "seconds")}
            else:
                finetuned[s["name"]] = {k: s.get(k) for k in FINETUNE_KEYS}
        else:
            missing.append({"stage": f"{s['kind']} {s['name']}", "status": s.get("status"),
                            "why": s.get("error")})
    return {"run_id": doc["run_id"], "state": doc.get("state"),
            "complete": doc.get("state") == "finished" and not missing,
            "code_version": doc.get("code_version"), "embedded": embedded,
            "finetuned": finetuned, "missing": missing,
            "comparison_id": doc.get("comparison_id"),
            "sample_records": doc.get("sample_records")}


def _remote_is_finetune(conn: sqlite3.Connection, name: str) -> bool:
    if not conn.execute("select 1 from remote.sqlite_master where type = 'table' and "
                        "name = 'finetunes'").fetchone():
        return False
    return conn.execute("select 1 from remote.finetunes where name = ?", (name,)
                        ).fetchone() is not None


def _staged_problem(conn: sqlite3.Connection, name: str, src: Path, expected: int | None
                    ) -> str | None:
    """Why the staged copy of a backbone is not a complete set (None if it is)."""
    if not src.is_dir():
        return "no embeddings staged"
    rows, shards = conn.execute("select count(*), count(distinct shard) from remote.embeddings "
                                "where backbone = ?", (name,)).fetchone()
    if not rows:
        return "the run's index has no rows for it"
    if expected is not None and rows != expected:
        return f"the index has {rows:,} rows, the run reported {expected:,}"
    for (shard,) in conn.execute("select distinct shard from remote.embeddings "
                                 "where backbone = ?", (name,)):
        for f in (f"shard-{shard:05d}.npy", f"shard-{shard:05d}.ids.npy"):
            if not (src / f).is_file():
                return f"{f} is missing"
    return None


def merge_results(conn: sqlite3.Connection, remote_path: Path, staged: Path, result: dict,
                  data_dir: Path | None = None) -> dict:
    """Bring a run home, whole or the stages it finished (see run_summary).

    `remote_path` is the run's index.sqlite (or its manifest-out.sqlite) and `staged`
    holds its embeddings as <backbone>/shard-*.npy. For each backbone the run embedded
    completely, and whose staged shards and index rows are all there and agree, the local
    folder moves to embeddings-archive/<backbone>-before-<run> and the run's shards and
    rows take its place. Nothing else is touched: a backbone the run didn't finish, or
    whose copy is incomplete, never replaces good local embeddings. The run's embedding
    speeds and its comparison's scoreboard rows are added once, so pulling the same run
    twice changes nothing.
    """
    data_dir = data_dir or config.DATA_DIR
    summary = run_summary(result)
    run_id = summary["run_id"]
    if summary.get("sample_records"):
        raise ValueError(f"run {run_id} was a rehearsal on {summary['sample_records']:,} "
                         "records; its results are not merged (they would replace real "
                         "embeddings)")
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    replaced, archived, refused = [], [], {}
    finetuned = []
    # Fine-tuned weights first: their embeddings are no use without them.
    for name in summary.get("finetuned", {}):
        srcs = [staged.parent / "models" / f"{name}{suffix}" for suffix in (".pt", ".json")]
        if not all(p.is_file() for p in srcs):
            refused[name] = "its weights are not staged"
            continue
        dest = data_dir / "models"
        dest.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            shutil.copyfile(src, dest / src.name)
        finetuned.append(name)
    marker = json.dumps({"run_id": run_id, "code_version": summary.get("code_version"),
                         "pulled_at": _now()})
    conn.execute("attach database ? as remote", (str(remote_path),))
    try:
        if finetuned:
            with conn:
                conn.execute("insert or replace into finetunes select * from remote.finetunes "
                             f"where name in ({','.join('?' * len(finetuned))})", finetuned)
        for name, info in summary.get("embedded", {}).items():
            if name in refused:
                continue
            if (name not in finetuned and _remote_is_finetune(conn, name)
                    and not (data_dir / "models" / f"{name}.pt").is_file()):
                refused[name] = "its fine-tuned weights are not here"
                continue
            dest = data_dir / "embeddings" / name
            if dest.is_dir() and (dest / f".run-{run_id}").exists():
                continue
            src = staged / name
            problem = _staged_problem(conn, name, src, (info or {}).get("photos"))
            if problem:
                refused[name] = problem
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
            (dest / f".run-{run_id}").write_text(marker, encoding="utf-8")
            replaced.append(name)
        cid = summary.get("comparison_id")
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
    return {"run_id": run_id, "state": summary.get("state"), "complete": summary["complete"],
            "code_version": summary.get("code_version"), "replaced": replaced,
            "archived": archived, "refused": refused, "missing": summary["missing"],
            "scoreboard_rows_added": scored, "comparison_id": cid}
