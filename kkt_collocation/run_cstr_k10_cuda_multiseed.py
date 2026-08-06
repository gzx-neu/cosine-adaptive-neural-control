"""Run or extend the CSTR 200+10 four-way comparison on CUDA."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20260771, 20260821))
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
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "kkt_collocation/results/multiseed10_cstr_k10_cuda_20260803_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--hds-workers-per-job", type=int, default=4)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError("Seeds must be selected from 20260771--20260820")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=args.resume)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=args.resume)
    manifest_path = output_root / "manifest.json"
    lock = threading.Lock()
    new_manifest = {
        "status": "running",
        "protocol": "CSTR N=100/RK10, S-u or 200 S-u + 10 KKT continuation",
        "seeds": args.seeds,
        "methods": METHODS,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "max_concurrent": args.max_concurrent,
        "phase": args.phase,
        "started_unix": time.time(),
        "jobs": {},
    }
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest["seeds"] = sorted(set(manifest.get("seeds", ())) | set(args.seeds))
        manifest["requested_seeds_this_invocation"] = args.seeds
        manifest["phase"] = args.phase
        manifest["resumed_unix"] = time.time()
    else:
        manifest = new_manifest
    write_json(manifest_path, manifest)

    def run_job(method: str, seed: int) -> tuple[str, int]:
        name = f"{method}_seed{seed}"
        cfg = METHODS[method]
        method_root = output_root / method
        train_dir = method_root / "cstr" / f"seed{seed}"
        train_command = [
            sys.executable,
            str(ROOT / "kkt_collocation/train_unified_economou_cstr_n100_ablation.py"),
            "--seed", str(seed),
            "--output-root", str(method_root),
            "--methods", cfg["branch"],
            "--supervised-epochs", "200",
            "--continuation-epochs", "10",
            *cfg["flags"],
        ]
        eval_command = [
            sys.executable,
            str(ROOT / "kkt_collocation/evaluate_unified_economou_cstr_n100_hds.py"),
            "--train", str(train_dir),
            "--methods", cfg["branch"],
            "--workers", str(args.hds_workers_per_job),
        ]
        started = time.time()
        with lock:
            manifest["jobs"][name] = {"status": "running", "started_unix": started}
            write_json(manifest_path, manifest)
        returncode = 0
        log_path = logs / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            phases = (
                (("train", train_command),) if args.phase == "train"
                else (("evaluate", eval_command),) if args.phase == "evaluate"
                else (("train", train_command), ("evaluate", eval_command))
            )
            for phase, command in phases:
                print(f"--- {phase} ---", file=log, flush=True)
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    text=True, check=False,
                )
                if completed.returncode != 0:
                    returncode = completed.returncode
                    break
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
    jobs = [(method, seed) for seed in args.seeds for method in METHODS]
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
