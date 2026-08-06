"""Export the pre-registered first 50 final-test states for MATLAB baselines."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
vdp = np.load(ROOT/"kkt_collocation"/"results"/"final_multiseed_vdp900_penalty_seed20260751"/"test_states.npy")[:50, :2]
pen = np.load(ROOT/"kkt_collocation"/"results"/"final_multiseed_penicillin400_penalty_seed20260761"/"test_x2.npy")[:50, None]
np.savetxt(ROOT/"原版VDP"/"vdp_teacher_states_final50.csv", vdp, delimiter=",", fmt="%.17g")
# Same stratified 50-point subset used by the already generated NLP-reference table.
vdp_nlp_indices = np.linspace(0, 399, 50, dtype=int)
vdp_nlp = np.load(ROOT/"kkt_collocation"/"results"/"final_multiseed_vdp900_penalty_seed20260751"/"test_states.npy")[vdp_nlp_indices, :2]
np.savetxt(ROOT/"原版VDP"/"vdp_teacher_states_nlp50.csv", vdp_nlp, delimiter=",", fmt="%.17g")
np.savetxt(ROOT/"精简_修改好的青霉素"/"penicillin_teacher_states_final50.csv", pen, delimiter=",", fmt="%.17g")
fixed_pen = np.asarray([.12, .20, .28, .30])
np.savetxt(ROOT/"精简_修改好的青霉素"/"penicillin_teacher_fixed4.csv", fixed_pen[:, None], delimiter=",", fmt="%.17g")
print("Exported fixed final-test subsets:", vdp.shape, pen.shape)
