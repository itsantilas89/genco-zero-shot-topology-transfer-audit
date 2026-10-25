"""Run the full evaluation matrix (README section 7).

Iterates manifest.csv x targets x normaliser variants and calls `gridfm_graphkit evaluate`
once per run:
  A  source stats re-keyed to the target (checkpoints that ship a stats file)
  B  target stats from inputs only (all checkpoints)
  C  graphkit's default refit (diagonal pairs and the subgrid-pretrained model only)

Resumable: a run whose results/per_run/<run_name>.json exists is skipped. Failures go to
results/failures.csv with the error tail. After every successful run the metrics are
extracted (postprocess.process_run) and the bulky outputs are deleted.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import (CONFIG_DIR, DATA_DIR, MANIFEST_PATH, RESULTS_DIR, ROOT, RUNS_DIR,
                    SUBGRID_ID, env_for_runs, graphkit_cmd, load_settings, pair_id,
                    run_name, target_names)
from postprocess import PER_RUN_DIR, process_run
from recalibrate_normalizer import variant_a_path, variant_b_path, write_variant_b

FAIL_PATH = RESULTS_DIR / "failures.csv"
_fail_lock = threading.Lock()


def variants_for(row, target: str) -> list[str]:
    v = []
    if row.normalizer_stats_path:
        v.append("A")
    v.append("B")
    if row.ckpt_id == SUBGRID_ID or row.source_grid == target:
        v.append("C")
    return v


def build_jobs(manifest: pd.DataFrame, targets: list[str], variants: set[str],
               ckpt_filter: list[str] | None) -> list[dict]:
    jobs = []
    for target in targets:
        for row in manifest.itertuples():
            if ckpt_filter and not any(s in row.ckpt_id for s in ckpt_filter):
                continue
            for v in variants_for(row, target):
                if variants and v not in variants:
                    continue
                jobs.append({"row": row, "target": target, "variant": v,
                             "run_name": run_name(row.ckpt_id, target, v)})
    return jobs


def command(job: dict, num_workers: int, batch_size: int | None, exp_name: str) -> list[str]:
    row, target, v, rn = job["row"], job["target"], job["variant"], job["run_name"]
    cmd = graphkit_cmd() + [
        "evaluate", "--config_conversion", "auto_inline",
        "--config", str(CONFIG_DIR / f"{pair_id(row.ckpt_id, target)}.yaml"),
        "--model_path", str(ROOT / row.weights_path),
        "--data_path", str(DATA_DIR),
        "--compute_dc_ac_metrics", "--save_output",
        "--exp_name", exp_name, "--run_name", rn,
        "--log_dir", str(RUNS_DIR / rn / "mlruns"),
        "--num_workers", str(num_workers),
    ]
    if batch_size:
        cmd += ["--batch_size", str(batch_size)]
    if v == "A":
        cmd += ["--normalizer_stats", str(variant_a_path(row, target))]
    elif v == "B":
        if not variant_b_path(target).exists():
            write_variant_b(target)
        cmd += ["--normalizer_stats", str(variant_b_path(target))]
    return cmd


def log_failure(job: dict, returncode, tail: str) -> None:
    with _fail_lock:
        RESULTS_DIR.mkdir(exist_ok=True)
        new = not FAIL_PATH.exists()
        with open(FAIL_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["run_name", "ckpt_id", "target", "variant", "returncode", "error"])
            w.writerow([job["run_name"], job["row"].ckpt_id, job["target"], job["variant"],
                        returncode, tail.replace("\n", " | ")[-1500:]])


def execute(job: dict, gpu: str | None, args, exp_name: str) -> tuple[str, bool, str]:
    rn = job["run_name"]
    run_dir = RUNS_DIR / rn
    if run_dir.exists():                      # leftover of an interrupted run
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    env = env_for_runs()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    t0 = time.time()
    try:
        proc = subprocess.run(command(job, args.num_workers, args.batch_size, exp_name),
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        log_failure(job, "timeout", str(e.stdout or "")[-1500:])
        return rn, False, "timeout"
    (run_dir / "log.txt").write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        log_failure(job, proc.returncode, tail)
        return rn, False, tail.strip().splitlines()[-1] if tail.strip() else "failed"
    row = job["row"]
    meta = {"ckpt_id": row.ckpt_id, "source_grid": row.source_grid, "size": row.size,
            "seed": int(row.seed), "hidden_size": int(row.hidden_size),
            "target": job["target"], "variant": job["variant"]}
    try:
        process_run(rn, meta, time.time() - t0, keep_predictions=args.keep_predictions)
    except Exception as e:                     # noqa: BLE001
        log_failure(job, "postprocess", repr(e))
        return rn, False, repr(e)
    if not args.keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return rn, True, f"{time.time() - t0:.0f}s"


def gpu_list(arg: str) -> list[str | None]:
    if arg != "auto":
        return arg.split(",") if arg else [None]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:                          # noqa: BLE001
        n = 0
    return [str(i) for i in range(n)] or [None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--ckpts", nargs="*", help="keep checkpoints whose id contains any of these")
    ap.add_argument("--variants", nargs="*", choices=["A", "B", "C"])
    ap.add_argument("--jobs", type=int, default=1, help="parallel evaluation processes")
    ap.add_argument("--gpus", default="auto", help="comma list of GPU ids, 'auto', or '' for none")
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--limit", type=int, help="run at most this many pending jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="forget previous failures.csv")
    ap.add_argument("--keep-predictions", action="store_true")
    ap.add_argument("--keep-runs", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    if args.num_workers is None:
        args.num_workers = settings["run"]["num_workers"]
    exp_name = settings["run"]["exp_name"]
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    targets = args.targets or target_names(settings)
    jobs = build_jobs(manifest, targets, set(args.variants or []), args.ckpts)
    done = {p.stem for p in PER_RUN_DIR.glob("*.json")}
    pending = [j for j in jobs if j["run_name"] not in done]
    if args.retry_failed and FAIL_PATH.exists():
        FAIL_PATH.unlink()
    n_pending_all = len(pending)
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(jobs)} runs in scope, {len(jobs) - n_pending_all} already done, "
          f"{len(pending)} to run now")
    if args.dry_run or not pending:
        for j in pending[:20]:
            print("  ", j["run_name"])
        return 0

    gpus = gpu_list(args.gpus)
    ok = bad = 0
    by_target: dict[str, list[dict]] = {}
    for j in pending:
        by_target.setdefault(j["target"], []).append(j)

    for target, tjobs in by_target.items():
        # first job alone: it builds the processed dataset cache that the rest then reuse
        first, rest = tjobs[0], tjobs[1:]
        print(f"[{target}] warm-up run {first['run_name']}", flush=True)
        rn, good, msg = execute(first, gpus[0], args, exp_name)
        ok, bad = ok + good, bad + (not good)
        print(f"  {'ok' if good else 'FAILED'} {rn}: {msg}", flush=True)
        if not rest:
            continue
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futs = [pool.submit(execute, j, gpus[i % len(gpus)], args, exp_name)
                    for i, j in enumerate(rest)]
            for n, fut in enumerate(as_completed(futs), 1):
                rn, good, msg = fut.result()
                ok, bad = ok + good, bad + (not good)
                print(f"  [{n}/{len(rest)}] {'ok' if good else 'FAILED'} {rn}: {msg}", flush=True)

    print(f"finished: {ok} ok, {bad} failed (see {FAIL_PATH.name})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
