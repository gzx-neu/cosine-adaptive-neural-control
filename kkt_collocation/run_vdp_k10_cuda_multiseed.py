"""Run the frozen 10-seed VDP 200+10 projection comparison on CUDA.

This launcher keeps every method/seed in its own directory, runs at most three
jobs concurrently on the single GPU, and preserves one log per job.  The
underlying experiment driver performs the matched-400 HDS evaluation.
"""
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

import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20260771, 20260781))
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


def _write_json(path: Path, value: object) -> None:
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
    parser.add_argument("--benchmark", choices=("vdp", "penicillin"), default="vdp")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "kkt_collocation/results/multiseed10_vdp_k10_cuda_20260803_v1",
    )
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--hds-workers-per-job", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--jobs", nargs="+", choices=METHODS, default=tuple(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "auto" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this launcher")
    if args.max_concurrent < 1 or args.hds_workers_per_job < 1 or args.threads_per_job < 1:
        raise ValueError("concurrency and worker counts must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=args.resume)
    log_root = output_root / "logs"
    log_root.mkdir(exist_ok=args.resume)
    manifest_path = output_root / "manifest.json"
    lock = threading.Lock()
    new_manifest: dict[str, object] = {
        "status": "running",
        "protocol": f"{args.benchmark} S-u/S-u+K, 200 supervised epochs + 10 continuation epochs",
        "seeds": args.seeds,
        "methods": METHODS,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "max_concurrent": args.max_concurrent,
        "hds_workers_per_job": args.hds_workers_per_job,
        "requested_device": args.device,
        "threads_per_job": args.threads_per_job,
        "phase": args.phase,
        "started_unix": time.time(),
        "jobs": {},
    }
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest["methods"] = METHODS
        manifest["seeds"] = sorted(set(manifest.get("seeds", ())) | set(args.seeds))
        manifest["requested_seeds_this_invocation"] = args.seeds
        manifest["phase"] = args.phase
        manifest["resumed_unix"] = time.time()
    else:
        manifest = new_manifest
    _write_json(manifest_path, manifest)

    def run_job(method: str, seed: int) -> tuple[str, int]:
        name = f"{method}_seed{seed}"
        method_root = output_root / method
        method_config = METHODS[method]
        command = [
            sys.executable,
            str(ROOT / "kkt_collocation/run_unified_su_suj_sk_konly_ablation.py"),
            "--benchmark", args.benchmark,
            "--seed", str(seed),
            "--output-root", str(method_root),
            "--phase", args.phase,
            "--methods", method_config["branch"],
            "--supervised-epochs", "200",
            "--continuation-epochs", "10",
            "--workers", str(args.hds_workers_per_job),
            *method_config["flags"],
        ]
        started = time.time()
        with lock:
            manifest["jobs"][name] = {
                "status": "running",
                "command": command,
                "started_unix": started,
            }
            _write_json(manifest_path, manifest)
        log_path = log_root / f"{name}.log"
        environment = os.environ.copy()
        if args.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = str(args.threads_per_job)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        returncode = completed.returncode
        if returncode == 0:
            seed_dir = method_root / args.benchmark / f"seed{seed}"
            config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
            expected_device = "cpu" if args.device == "cpu" else "cuda"
            if config["device"] != expected_device:
                returncode = 97
        with lock:
            manifest["jobs"][name].update({
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
            })
            _write_json(manifest_path, manifest)
        return name, returncode

    failures: list[str] = []
    jobs = [(method, seed) for seed in args.seeds for method in args.jobs]
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {executor.submit(run_job, method, seed): (method, seed) for method, seed in jobs}
        for future in as_completed(futures):
            name, returncode = future.result()
            print(f"{name}: returncode={returncode}", flush=True)
            if returncode != 0:
                failures.append(name)

    manifest["status"] = "completed" if not failures else "completed_with_failures"
    manifest["failures"] = failures
    manifest["finished_unix"] = time.time()
    _write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"Failed jobs: {failures}")


if __name__ == "__main__":
    main()
