"""Merge non-overlapping matched400 Jiang--Fu continuation shards safely."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def completed(row: dict[str, str]) -> bool:
    return row.get("success", "").strip() in {"1", "true", "True"} or bool(row.get("error_identifier", "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.directory / "jiang_fu_matched400_raw.csv"
    with base_path.open(encoding="utf-8-sig", newline="") as handle:
        base = list(csv.DictReader(handle))
        fields = handle.seek(0) or list(csv.DictReader(handle).fieldnames or [])
    if len(base) != 800:
        raise RuntimeError(f"Expected 800 matched rows in canonical checkpoint, found {len(base)}.")
    lookup = {(row["problem"], row["point_id"]): index for index, row in enumerate(base)}
    shard_paths = sorted(args.directory.glob("jiang_fu_matched400_raw_shard_*_of_*.csv"))
    if len(shard_paths) != 4:
        raise RuntimeError(f"Expected four shard CSVs, found {len(shard_paths)}.")

    for path in shard_paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("problem") or not completed(row):
                    continue
                key = (row["problem"], row["point_id"])
                if key in lookup:
                    index = lookup[key]
                elif row["problem"] == "Penicillin":
                    match = re.fullmatch(r"pen_(\d{3})", row["point_id"])
                    if match is None or not 1 <= int(match.group(1)) <= 400:
                        raise RuntimeError(f"Unknown shard row {key} in {path.name}.")
                    # The interrupted canonical CSV leaves future records blank,
                    # but matched400 has a fixed VDP-then-penicillin ordering.
                    index = 400 + int(match.group(1)) - 1
                else:
                    raise RuntimeError(f"Unknown shard row {key} in {path.name}.")
                base[index] = row

    vdp = [row for row in base if row["problem"] == "VDP"]
    penicillin = [row for row in base if row["problem"] == "Penicillin"]
    if len(vdp) != 400 or len(penicillin) != 400 or not all(completed(row) for row in base):
        raise RuntimeError("Merged checkpoint is incomplete.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(base)
    print(f"Wrote {args.output} (VDP={len(vdp)}, Penicillin={len(penicillin)}).")


if __name__ == "__main__":
    main()
