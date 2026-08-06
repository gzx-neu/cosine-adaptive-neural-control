"""Unified command-line entry point for the CA-KKT paper experiments."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEFAULT_SEEDS = tuple(range(20260771, 20260801))
RUN_ROOT = ROOT / "reproduced_results"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def check_environment() -> None:
    import matplotlib
    import numpy
    import pandas
    import scipy
    import torch

    try:
        import casadi
        casadi_version = casadi.__version__
    except ImportError:
        casadi_version = "missing (needed only for deterministic label generation)"

    protocol = json.loads((ROOT / "configs/paper_30seed.json").read_text(encoding="utf-8"))
    print(f"repository: {ROOT}")
    print(f"protocol:   {protocol['protocol_id']}")
    print(f"python:     {sys.version.splitlines()[0]}")
    print(f"numpy:      {numpy.__version__}")
    print(f"scipy:      {scipy.__version__}")
    print(f"pandas:     {pandas.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"torch:      {torch.__version__}")
    print(f"casadi:     {casadi_version}")
    print(f"cuda:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu:        {torch.cuda.get_device_name(0)}")

    required = (
        "kkt_collocation/data/vdp_kkt_20x20.pkl",
        "kkt_collocation/data/penicillin_kkt_400_true_duals.pkl",
        "kkt_collocation/results/final_multiseed_vdp900_penalty_seed20260751/validation_states.npy",
        "kkt_collocation/results/final_multiseed_vdp900_penalty_seed20260751/test_states.npy",
        "kkt_collocation/results/final_multiseed_penicillin400_penalty_seed20260761/validation_x2.npy",
        "kkt_collocation/results/final_multiseed_penicillin400_penalty_seed20260761/test_x2.npy",
        "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl",
        "kkt_collocation/results/jiang_fu_matched400_comparison/per_point_seed.csv",
        "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0/records.jsonl",
        "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0/summary.json",
        "kkt_collocation/results/economou_cstr_n100_test400_hds_s200_sk10_fair/cold_reference_high_fidelity.csv",
        "kkt_collocation/results/ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1/aggregate_summary.json",
        "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1/aggregate_summary.json",
        "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1/cohorts/selected_50x2_points.csv",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required frozen inputs:\n" + "\n".join(missing))
    print("required frozen inputs: OK")


def run_training(benchmarks: list[str], seeds: list[int], max_concurrent: int, workers: int, resume: bool) -> None:
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Seeds outside the frozen 30-seed protocol: {unknown}")

    if "vdp" in benchmarks:
        command = [
            str(PYTHON), "kkt_collocation/run_vdp_k10_cuda_multiseed.py",
            "--benchmark", "vdp", "--device", "auto",
            "--output-root", str(RUN_ROOT / "vdp"),
            "--max-concurrent", str(max_concurrent),
            "--hds-workers-per-job", str(workers),
            "--seeds", *map(str, seeds),
        ]
        run(command + (["--resume"] if resume else []))
    if "penicillin" in benchmarks:
        command = [
            str(PYTHON), "kkt_collocation/run_vdp_k10_cuda_multiseed.py",
            "--benchmark", "penicillin", "--device", "cpu",
            "--output-root", str(RUN_ROOT / "penicillin"),
            "--max-concurrent", str(max_concurrent),
            "--hds-workers-per-job", str(workers),
            "--seeds", *map(str, seeds),
        ]
        run(command + (["--resume"] if resume else []))
    if "cstr" in benchmarks:
        command = [
            str(PYTHON), "kkt_collocation/run_cstr_k10_cuda_multiseed.py",
            "--output-root", str(RUN_ROOT / "cstr"),
            "--max-concurrent", str(max_concurrent),
            "--hds-workers-per-job", str(workers),
            "--seeds", *map(str, seeds),
        ]
        run(command + (["--resume"] if resume else []))


def aggregate(seeds: list[int]) -> None:
    aggregate_root = RUN_ROOT / "aggregates"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    for benchmark in ("vdp", "penicillin"):
        source = RUN_ROOT / benchmark
        if source.exists():
            run([
                str(PYTHON), "kkt_collocation/aggregate_vdp_k10_projection_comparison.py",
                "--benchmark", benchmark,
                "--output", str(source),
                "--seeds", *map(str, seeds),
                "--aggregate-output", str(aggregate_root / benchmark),
            ])
    source = RUN_ROOT / "cstr"
    if source.exists():
        run([
            str(PYTHON), "kkt_collocation/aggregate_cstr_k10_projection_comparison.py",
            "--output", str(source),
            "--expected-device", "cuda",
            "--seeds", *map(str, seeds),
        ])
        if tuple(seeds) == DEFAULT_SEEDS:
            env = os.environ.copy()
            env["CAKKT_CSTR_SOURCE_ROOT"] = str(source.resolve())
            env["CAKKT_CSTR_MARGIN_OUT"] = str((RUN_ROOT / "cstr_margin1e8").resolve())
            run([
                str(PYTHON), "kkt_collocation/reevaluate_cstr_margin1e6_30seeds.py",
                "--workers", "6", "--force",
            ], env=env)


def regenerate_figures() -> None:
    figure_out = ROOT / "paper_assets" / "generated"
    figure_out.mkdir(parents=True, exist_ok=True)

    run([str(PYTHON), "paper_assets/objective_gap_30seeds/plot_objective_gap_30seeds.py"])
    run([str(PYTHON), "kkt_collocation/plot_gradient_conflict_and_timing.py"])
    run([str(PYTHON), "kkt_collocation/plot_linear_cosine_constraint_population_30seeds.py", "--render-only"])
    run([str(PYTHON), "kkt_collocation/plot_linear_cosine_control_corrections_30seeds.py", "--render-only"])

    candidates = (
        ROOT / "paper_assets/objective_gap_30seeds/objective_gap_30seeds_EAAI_final.pdf",
        ROOT / "kkt_collocation/results/paper30_gradient_conflict_and_timing_20260803_v1/gradient_cosine_projection_distribution_30seeds.pdf",
        ROOT / "论文写作/figures/linear_cosine_constraint_population_30seeds_all12000.pdf",
        ROOT / "论文写作/figures/linear_cosine_control_corrections_30seeds_all12000.pdf",
    )
    for source in candidates:
        if source.exists():
            shutil.copy2(source, figure_out / source.name)
    print(f"generated figures: {figure_out}")


def verify_frozen() -> None:
    run([str(PYTHON), "scripts/verify_bundle.py"])


def rebuild_ood_table() -> None:
    """Re-audit frozen cold references and rebuild the 30-seed OOD table."""
    run([
        str(PYTHON),
        "kkt_collocation/run_ood_matched_reference_50x2.py",
        "--stage", "compile",
    ])


def reevaluate_cached_hds(workers: int, resume: bool, checkpoint_root: Path | None) -> None:
    """Re-evaluate reproduced checkpoints with discrete-31 segment-cache HDS."""
    command = [
        str(PYTHON),
        "-m", "kkt_collocation.reevaluate_multiseed30_discrete31_cached",
        "--output", str(RUN_ROOT / "formal_multiseed30_discrete31_cached_margin1e8"),
        "--workers", str(workers),
    ]
    if checkpoint_root is not None:
        command += ["--checkpoint-root", str(checkpoint_root.resolve())]
    run(command + (["--resume"] if resume else []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="show environment and validate required inputs")
    sub.add_parser("frozen", help="verify archived paper summaries and source data")
    sub.add_parser("figures", help="regenerate paper figures from frozen source data")
    sub.add_parser("ood", help="rebuild the frozen 30-seed matched OOD cold-start+HDS table")

    hds = sub.add_parser("hds-cached31", help="re-evaluate all reproduced checkpoints with cached 31-point-grid HDS")
    hds.add_argument("--workers", type=int, default=8)
    hds.add_argument("--resume", action="store_true")
    hds.add_argument("--checkpoint-root", type=Path, default=None)

    train = sub.add_parser("train", help="rerun one or more 30-seed benchmarks")
    train.add_argument("--benchmarks", nargs="+", choices=("vdp", "penicillin", "cstr"), default=("vdp", "penicillin", "cstr"))
    train.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    train.add_argument("--max-concurrent", type=int, default=2)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--resume", action="store_true", help="resume an interrupted output root")

    agg = sub.add_parser("aggregate", help="aggregate fresh reruns under reproduced_results")
    agg.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check":
        check_environment()
    elif args.command == "frozen":
        verify_frozen()
    elif args.command == "figures":
        regenerate_figures()
    elif args.command == "ood":
        rebuild_ood_table()
    elif args.command == "hds-cached31":
        reevaluate_cached_hds(args.workers, args.resume, args.checkpoint_root)
    elif args.command == "train":
        run_training(args.benchmarks, list(args.seeds), args.max_concurrent, args.workers, args.resume)
    elif args.command == "aggregate":
        aggregate(list(args.seeds))


if __name__ == "__main__":
    main()
