"""Run CSTR seeds 21--30 (20260791--20260800) entirely on CPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20260791, 20260801))
METHODS = {
    "supervised": {"branch": "S-u", "flags": ()},
    "unprocessed": {"branch": "S-u+K", "flags": ()},
    "linear_cosine": {
        "branch": "S-u+K", "flags": ("--cosine-adaptive-kkt-conflict-projection",),
    },
    "standard_pcgrad": {
        "branch": "S-u+K", "flags": ("--project-conflicting-kkt-gradient",),
    },
}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "kkt_collocation/results/multiseed_cstr_k10_cpu_seeds21_30_20260803_v1",
    )
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--hds-workers-per-job", type=int, default=2)
    parser.add_argument("--torch-threads-per-job", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    manifest_path = output_root / "manifest.json"
    lock = threading.Lock()
    manifest = {
        "status": "running",
        "protocol": "CSTR N=100/RK10, CPU-only, seeds 20260791--20260800, 200+10",
        "seeds": SEEDS,
        "methods": METHODS,
        "python": sys.executable,
        "max_concurrent": args.max_concurrent,
        "hds_workers_per_job": args.hds_workers_per_job,
        "torch_threads_per_job": args.torch_threads_per_job,
        "started_unix": time.time(),
        "jobs": {},
    }
    write_json(manifest_path, manifest)
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "-1"
    child_env["OMP_NUM_THREADS"] = str(args.torch_threads_per_job)
    child_env["MKL_NUM_THREADS"] = str(args.torch_threads_per_job)

    def run_job(method: str, seed: int) -> tuple[str, int]:
        name = f"{method}_seed{seed}"
        cfg = METHODS[method]
        method_root = output_root / method
        train_dir = method_root / "cstr" / f"seed{seed}"
        train_command = [
            sys.executable,
            str(ROOT / "kkt_collocation/train_unified_economou_cstr_n100_ablation.py"),
            "--seed", str(seed), "--output-root", str(method_root),
            "--methods", cfg["branch"], "--supervised-epochs", "200",
            "--continuation-epochs", "10", *cfg["flags"],
        ]
        eval_command = [
            sys.executable,
            str(ROOT / "kkt_collocation/evaluate_unified_economou_cstr_n100_hds.py"),
            "--train", str(train_dir), "--methods", cfg["branch"],
            "--workers", str(args.hds_workers_per_job),
        ]
        started = time.time()
        with lock:
            manifest["jobs"][name] = {"status": "running", "started_unix": started}
            write_json(manifest_path, manifest)
        returncode = 0
        log_path = logs / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            for phase, command in (("train", train_command), ("evaluate", eval_command)):
                print(f"--- {phase} ---", file=log, flush=True)
                completed = subprocess.run(
                    command, cwd=ROOT, env=child_env, stdout=log,
                    stderr=subprocess.STDOUT, text=True, check=False,
                )
                if completed.returncode:
                    returncode = completed.returncode
                    break
        if returncode == 0:
            config = json.loads((train_dir / "config.json").read_text(encoding="utf-8"))
            if config["reproducibility"]["device"] != "cpu":
                returncode = 97
        with lock:
            manifest["jobs"][name].update({
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            })
            write_json(manifest_path, manifest)
        return name, returncode

    failures = []
    jobs = [(method, seed) for seed in SEEDS for method in METHODS]
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = [executor.submit(run_job, method, seed) for method, seed in jobs]
        for future in as_completed(futures):
            name, returncode = future.result()
            print(f"{name}: returncode={returncode}", flush=True)
            if returncode:
                failures.append(name)
    manifest["status"] = "completed" if not failures else "completed_with_failures"
    manifest["failures"] = failures
    manifest["finished_unix"] = time.time()
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


if __name__ == "__main__":
    main()
