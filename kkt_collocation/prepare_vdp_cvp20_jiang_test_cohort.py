"""Export the fixed VDP 400-point test cohort for the Jiang--Fu CVP20 MATLAB driver."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "kkt_collocation/results/exploratory_vdp_cvp20_jiang_strict_v1/cohorts"
    output.mkdir(parents=True, exist_ok=False)
    states = np.load(ROOT / "kkt_collocation/results/final_multiseed_vdp900_penalty_seed20260751/test_states.npy")
    if states.shape != (400, 3):
        raise ValueError(f"Expected frozen VDP test_states (400, 3), got {states.shape}")
    # No header: MATLAB readmatrix must receive exactly the three numeric columns.
    np.savetxt(output / "test_states_400.csv", states, delimiter=",", fmt="%.17g")
    np.save(output / "test_states_400.npy", states)


if __name__ == "__main__":
    main()
