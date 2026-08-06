"""Verify frozen inputs and headline 30-seed result invariants."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(20260771, 20260801))


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 5e-12) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def verify_results() -> None:
    cached31 = read_json(
        "kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e8_20260806_v1/aggregate_summary.json"
    )
    if [int(seed) for seed in cached31["seeds"]] != SEEDS:
        raise AssertionError("Seed sequence differs from the frozen protocol")

    if cached31["grid_size"] != 31 or cached31["bisection"]:
        raise AssertionError("Cached HDS candidate protocol mismatch")
    if not cached31["segment_cache_reused"] or cached31["final_reaudit_performed"]:
        raise AssertionError("Cached HDS propagation-reuse invariant failed")
    if cached31["acceptance_threshold"] != -1e-8 or not cached31["formal_protocol"]:
        raise AssertionError("Formal HDS threshold must be -1e-8")
    expected_cached = {
        "vdp": 0.1655908761042304,
        "penicillin": 0.3174297829038171,
        "cstr": 0.24398521332539502,
    }
    for benchmark, expected in expected_cached.items():
        close(cached31["benchmarks"][benchmark]["linear_cosine"]["hds_gap_percent"]["mean"], expected)
        for method in ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad"):
            item = cached31["benchmarks"][benchmark][method]
            if item["accepted_total"] != 12000 or item["fallback_total"] != 0:
                raise AssertionError(f"Cached HDS acceptance invariant failed for {benchmark}/{method}")
            close(item["post_hds_violation_rate_percent"]["mean"], 0.0)

    print("headline 30-seed cached-HDS result invariants: OK")


def verify_required_inputs() -> None:
    required = (
        "configs/paper_30seed.json",
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
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing required release inputs:\n" + "\n".join(missing))
    print(f"required training/reference inputs: {len(required)} files OK")


def verify_ood_results() -> None:
    root = ROOT / "kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1"
    report = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    if report["protocol"]["points_per_layer"] != 50 or report["protocol"]["training_seeds"] != SEEDS:
        raise AssertionError("OOD point count or seed sequence differs from the frozen protocol")
    expected = {
        ("vdp", "near_ood_0_10pct"): 0.3255940875261959,
        ("vdp", "far_ood_10_20pct"): 1.8153517573531512,
        ("penicillin", "near_ood_0_10pct"): 1.3352984019717098,
        ("penicillin", "far_ood_10_20pct"): 4.230290550312939,
        ("cstr", "near_ood_0_10pct"): 0.29447685865243745,
        ("cstr", "far_ood_10_20pct"): 0.5596562452219253,
    }
    if len(report["rows"]) != 6:
        raise AssertionError("Expected six benchmark/layer OOD rows")
    for row in report["rows"]:
        key = (row["problem"], row["layer"])
        if row["reference_solver_success"] != 50 or row["qualified_references"] != 50:
            raise AssertionError(f"OOD deterministic-reference coverage failed for {key}")
        close(row["gap_qualified_percent"]["mean"], expected[key])
    with (root / "cohorts/selected_50x2_points.csv").open(encoding="utf-8-sig", newline="") as handle:
        selected = list(csv.DictReader(handle))
    with (root / "per_seed_matched_gap.csv").open(encoding="utf-8-sig", newline="") as handle:
        per_seed = list(csv.DictReader(handle))
    if len(selected) != 300 or len(per_seed) != 180:
        raise AssertionError("Expected 300 selected OOD points and 180 seed/layer summaries")
    stress = read_json("kkt_collocation/results/ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1/aggregate_summary.json")
    stress_protocol = read_json("kkt_collocation/results/ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1/protocol.json")
    if stress["training_seeds"] != SEEDS or stress["samples_per_layer"] != 100:
        raise AssertionError("OOD stress protocol mismatch")
    if stress_protocol.get("threshold") != -1e-8:
        raise AssertionError("OOD stress threshold must be -1e-8")
    print("30-seed matched OOD 50+50 reference invariants: OK")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path) -> None:
    files = sorted(
        item for item in ROOT.rglob("*")
        if item.is_file() and item != path
        and ".git" not in item.parts
        and "reproduced_results" not in item.parts and "scratch" not in item.parts
        and "__pycache__" not in item.parts and ".pytest_cache" not in item.parts
        and not ("matlab" in item.parts and "jiang_fu" in item.parts)
        and "generated" not in item.parts and item.suffix.lower() not in {".tiff", ".tif"}
    )
    lines = [f"{sha256(item)}  {item.relative_to(ROOT).as_posix()}" for item in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} checksums to {path.relative_to(ROOT)}")


def verify_manifest(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {path}")
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = ROOT / relative
        if not target.is_file() or sha256(target) != expected:
            raise AssertionError(f"Checksum mismatch: {relative}")
        checked += 1
    print(f"checksum manifest: {checked} files OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    verify_required_inputs()
    verify_results()
    verify_ood_results()
    manifest = ROOT / "MANIFEST.sha256"
    if args.write_manifest:
        write_manifest(manifest)
    else:
        print("checksum manifest: omitted from compact source release")


if __name__ == "__main__":
    main()
