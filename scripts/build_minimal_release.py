"""Build a clean, staged GitHub release from the full working package.

The working package intentionally retains historical diagnostics and rendered
assets. This script copies only source code, frozen inputs, compact aggregate
tables, and the staged documentation needed for the headline 30-seed result.
It never deletes or modifies the source package.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = (
    "README.md", "REPRODUCIBILITY.md", "MINIMAL_REPRODUCTION.md",
    "CODE_MAP.md", "LICENSE", "THIRD_PARTY_NOTICE.md",
    "environment.yml", "requirements.txt", ".gitignore",
)
CODE_DIRS = ("kkt_collocation", "offline_safe_control", "scripts", "tests")
INPUT_FILES = (
    "configs/paper_30seed.json",
    "configs/minimal_release_manifest.json",
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
    "kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e8_20260806_v1/aggregate_summary.json",
    "kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e8_20260806_v1/aggregate_table.md",
    "kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e8_20260806_v1/per_seed_summary.csv",
    "kkt_collocation/results/ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1/aggregate_summary.json",
    "kkt_collocation/results/ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1/protocol.json",
    "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1/aggregate_summary.json",
    "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1/cohorts/selected_50x2_points.csv",
    "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1/per_seed_matched_gap.csv",
)


def copy_file(source: Path, destination_root: Path, relative: str, missing: list[str]) -> None:
    if not source.is_file():
        missing.append(relative)
        return
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(destination: Path, *, dry_run: bool = False) -> dict[str, int]:
    copied = 0
    missing: list[str] = []

    def add(relative: str) -> None:
        nonlocal copied
        source = ROOT / relative
        if dry_run:
            if source.is_file():
                copied += 1
            else:
                missing.append(relative)
            return
        before = destination / relative
        copy_file(source, destination, relative, missing)
        if before.is_file():
            copied += 1

    for relative in ROOT_FILES:
        add(relative)
    for directory in CODE_DIRS:
        for source in (ROOT / directory).rglob("*.py"):
            add(str(source.relative_to(ROOT)))
    for relative in INPUT_FILES:
        add(relative)

    return {"copied_files": copied, "missing_files": len(missing), "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "ca_kkt_reproducibility_minimal")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = build(args.output, dry_run=args.dry_run)
    print(f"copied_files={result['copied_files']}")
    print(f"missing_files={result['missing_files']}")
    for path in result["missing"]:
        print(f"MISSING {path}")
    if result["missing_files"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
