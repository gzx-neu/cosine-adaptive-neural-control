"""Merge the frozen 1e-8 VDP/penicillin and CSTR summaries into one formal bundle."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_ROOT = ROOT.parent
RESULTS = ROOT / "kkt_collocation" / "results"
SOURCE_RESULTS = FULL_ROOT / "kkt_collocation" / "results"
SOURCE_VP = SOURCE_RESULTS / "nonformal_vdp_penicillin_threshold1e8_discrete31_cached_30seeds_20260806_v1"
SOURCE_CSTR = SOURCE_RESULTS / "nonformal_cstr_threshold1e8_discrete31_cached_30seeds_20260806_v2"
OUT = RESULTS / "formal_multiseed30_discrete31_cached_margin1e8_20260806_v1"


def main() -> None:
    vp = json.loads((SOURCE_VP / "aggregate_summary.json").read_text(encoding="utf-8"))
    cstr = json.loads((SOURCE_CSTR / "aggregate_summary.json").read_text(encoding="utf-8"))
    if vp["seeds"] != cstr["seeds"]:
        raise ValueError("VDP/penicillin and CSTR seed protocols differ")
    merged = dict(vp)
    merged["formal_protocol"] = True
    merged["protocol"] = "30 frozen seeds, four methods, matched 400 points, formal discrete-only 31-candidate cached HDS"
    merged["acceptance_threshold"] = -1e-8
    merged["benchmarks"] = dict(vp["benchmarks"])
    merged["benchmarks"]["cstr"] = cstr["benchmarks"]["cstr"]
    merged["paired"] = dict(vp.get("paired", {}))
    merged["paired"]["cstr"] = cstr.get("paired", {}).get("cstr", {})
    merged["timing_note"] = vp.get("timing_note", cstr.get("timing_note", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    rows = []
    for source in (SOURCE_VP / "per_seed_summary.csv", SOURCE_CSTR / "per_seed_summary.csv"):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    fields = sorted({key for row in rows for key in row})
    with (OUT / "per_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Formal cached discrete-31 HDS: 30-seed matched-400 results",
        "",
        "All values use 31 closest-to-one candidates without bisection, adaptive DOP853 event audits, segment propagation reuse, and $g_{\\max}\\leq-10^{-8}$.",
        "",
        "| Benchmark | Method | HDS gap (%; mean +/- seed SD) | Corrected segments | Nominal violation | Post-HDS violation | Accepted |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in ("vdp", "penicillin", "cstr"):
        for method, item in merged["benchmarks"][benchmark].items():
            lines.append(
                f"| {benchmark} | {method} | {item['hds_gap_percent']['mean']:.5f} +/- {item['hds_gap_percent']['sample_sd']:.5f} | "
                f"{item['mean_corrected_segments']['mean']:.4f} | {item['nominal_violation_rate_percent']['mean']:.2f}% | "
                f"{item['post_hds_violation_rate_percent']['mean']:.2f}% | {item['accepted_total']}/{item['accepted_total'] + item['fallback_total']} |"
            )
    lines += ["", "CSTR qualified-286 gap:", "", "| Method | Gap (%; mean +/- seed SD) |", "|---|---:|"]
    for method, item in merged["benchmarks"]["cstr"].items():
        q = item["hds_gap_qualified286_percent"]
        lines.append(f"| {method} | {q['mean']:.5f} +/- {q['sample_sd']:.5f} |")
    lines += ["", merged.get("timing_note", "")]
    (OUT / "aggregate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote formal bundle: {OUT}")


if __name__ == "__main__":
    main()
