"""Export HDS-corrected Penicillin CVP20 policy controls for Jiang--Fu warm starts.

This reads frozen checkpoints and the same 400 test initial conditions; it
does not retrain a policy.  The exported rows are candidates only: the
subsequent Jiang--Fu NLP remains the optimization authority.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_event_hds import AdaptiveEventHDSConfig, AdaptiveEventHDSCorrector  # noqa: E402
from kkt_collocation.run_penicillin_cvp20_jiang_su_k import (  # noqa: E402
    HORIZON, N, Policy, UMAX, _load_rows, g, gdot, ode,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    test = _load_rows(args.data_dir, "test")
    corrector = AdaptiveEventHDSCorrector(ode, g, gdot, (0., UMAX), AdaptiveEventHDSConfig(grid_size=31))
    fieldnames = ["method", "index", "x2_0", "accepted", "corrected_segments"] + [f"u_{k:02d}" for k in range(N)]
    rows: list[dict[str, float | int | str | bool]] = []
    for method in ("S-u", "S-u+Jiang-KKT"):
        saved = torch.load(args.checkpoint_dir / f"{method}.pth", map_location="cpu", weights_only=False)
        model = Policy(); model.load_state_dict(saved["model"]); model.eval()
        norm = saved["normalization"]
        features = torch.tensor(((test["x2"] - norm["mean"]) / norm["std"])[:, None], dtype=torch.float32)
        with torch.no_grad():
            nominal = model(features).numpy()
        for index, (x2, control) in enumerate(zip(test["x2"], nominal)):
            result = corrector.correct(np.array([1., x2, .001, 250.]), control, HORIZON / N)
            if not result.accepted:
                raise RuntimeError(f"HDS unexpectedly rejected {method} index {index}")
            row: dict[str, float | int | str | bool] = {
                "method": method, "index": index, "x2_0": float(x2), "accepted": True,
                "corrected_segments": int(sum(part.corrected for part in result.segments)),
            }
            row.update({f"u_{k:02d}": float(value) for k, value in enumerate(result.controls)})
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} HDS-corrected warm starts to {args.output}")


if __name__ == "__main__":
    main()
