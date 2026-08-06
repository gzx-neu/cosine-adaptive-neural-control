"""Recompute method summaries from preserved per-sample CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kkt_collocation.run_penicillin_ablation import summary as penicillin_summary
from kkt_collocation.run_vdp_ablation import summarise as vdp_summary


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key == "method":
                continue
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--directories", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    for directory in args.directories:
        summary_path = directory / "summary.json"
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _rows(directory / "per_sample.csv")
        report["methods"] = vdp_summary(rows) if args.problem == "vdp" else penicillin_summary(rows)
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rebuilt {summary_path}")


if __name__ == "__main__":
    main()
