"""Write the frozen penicillin training/test x2 cohorts for CVP20 Jiang--Fu solves."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_column(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x2_0"])
        writer.writerows([[float(value)] for value in values])


def main() -> None:
    output = ROOT / "kkt_collocation/results/exploratory_penicillin_cvp20_jiang_v1/cohorts"
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / "kkt_collocation/data/penicillin_kkt_400_true_duals.pkl").open("rb") as handle:
        train = np.asarray(pickle.load(handle)["initial_state"], dtype=float)[:, 1]
    test = np.load(ROOT / "kkt_collocation/results/final_multiseed_penicillin400_penalty_seed20260761/test_x2.npy")
    if train.shape != (400,) or test.shape != (400,):
        raise ValueError("Expected fixed 400-point training and test cohorts")
    write_column(output / "train_x2_400.csv", train)
    write_column(output / "test_x2_400.csv", test)
    np.save(output / "train_x2_400.npy", train)
    np.save(output / "test_x2_400.npy", test)


if __name__ == "__main__":
    main()
