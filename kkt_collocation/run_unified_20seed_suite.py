"""Sequential runner for the frozen 20-seed unified ablation suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20260771, 20260791))
BENCHMARKS = ("vdp", "penicillin", "cstr")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_job(name: str, command: list[str], log_path: Path, manifest: dict, manifest_path: Path) -> None:
    started = time.time()
    manifest["jobs"][name] = {
        "status": "running",
        "command": command,
        "started_unix": started,
        "log": str(log_path),
    }
    _write_json(manifest_path, manifest)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status = "completed" if completed.returncode == 0 else "failed"
    manifest["jobs"][name].update({
        "status": status,
        "returncode": completed.returncode,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    })
    _write_json(manifest_path, manifest)
    if completed.returncode != 0:
        raise RuntimeError(f"Job {name} failed; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "kkt_collocation/results/unified_su_suj_sk_konly_20seeds_v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"Formal seeds must be selected from the preregistered set {SEEDS}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(exist_ok=True)
    manifest_path = output_root / "suite_manifest.json"
    try:
        output_argument = str(output_root.relative_to(ROOT))
    except ValueError:
        output_argument = str(output_root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["preregistered_seeds"]) != SEEDS:
            raise ValueError("Existing manifest has a different preregistered seed set")
        manifest["requested_seeds_this_invocation"] = seeds
        manifest["benchmarks_this_invocation"] = tuple(args.benchmarks)
        manifest.setdefault("invocations", []).append({
            "started_unix": time.time(),
            "seeds": seeds,
            "benchmarks": tuple(args.benchmarks),
        })
        manifest["status"] = "running_requested_jobs"
        _write_json(manifest_path, manifest)
    else:
        manifest = {
            "protocol": "frozen unified S-u/S-uJ/S+K(200+10)/K-only(210) ablation",
            "preregistered_seeds": SEEDS,
            "requested_seeds_this_invocation": seeds,
            "benchmarks_this_invocation": tuple(args.benchmarks),
            "sequential_pytorch_training": True,
            "test_set_tuning": False,
            "invocations": [{
                "started_unix": time.time(),
                "seeds": seeds,
                "benchmarks": tuple(args.benchmarks),
            }],
            "jobs": {},
        }
        _write_json(manifest_path, manifest)

    for seed in seeds:
        for benchmark in args.benchmarks:
            if benchmark in ("vdp", "penicillin"):
                seed_dir = output_root / benchmark / f"seed{seed}"
                completion = seed_dir / "summary.json"
                name = f"{benchmark}_seed{seed}_train_and_hds"
                if completion.exists():
                    manifest["jobs"].setdefault(name, {"status": "skipped_complete", "completion": str(completion)})
                    _write_json(manifest_path, manifest)
                    continue
                command = [
                    sys.executable,
                    "kkt_collocation/run_unified_su_suj_sk_konly_ablation.py",
                    "--benchmark", benchmark,
                    "--seed", str(seed),
                    "--output-root", output_argument,
                    "--phase", "all",
                    "--workers", str(args.workers),
                ]
                _run_job(name, command, log_root / f"{name}.log", manifest, manifest_path)
            else:
                seed_dir = output_root / "cstr" / f"seed{seed}"
                training_completion = seed_dir / "training_summary.json"
                train_name = f"cstr_seed{seed}_train"
                if training_completion.exists():
                    manifest["jobs"].setdefault(train_name, {"status": "skipped_complete", "completion": str(training_completion)})
                    _write_json(manifest_path, manifest)
                else:
                    command = [
                        sys.executable,
                        "kkt_collocation/train_unified_economou_cstr_n100_ablation.py",
                        "--seed", str(seed),
                        "--output-root", output_argument,
                    ]
                    _run_job(train_name, command, log_root / f"{train_name}.log", manifest, manifest_path)

                evaluation_completion = seed_dir / "hds_test400/summary.json"
                eval_name = f"cstr_seed{seed}_hds"
                if evaluation_completion.exists():
                    manifest["jobs"].setdefault(eval_name, {"status": "skipped_complete", "completion": str(evaluation_completion)})
                    _write_json(manifest_path, manifest)
                else:
                    command = [
                        sys.executable,
                        "kkt_collocation/evaluate_unified_economou_cstr_n100_hds.py",
                        "--train", str(seed_dir.relative_to(ROOT)),
                        "--workers", str(args.workers),
                    ]
                    _run_job(eval_name, command, log_root / f"{eval_name}.log", manifest, manifest_path)

    manifest["status"] = "completed_requested_jobs"
    manifest["finished_unix"] = time.time()
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
