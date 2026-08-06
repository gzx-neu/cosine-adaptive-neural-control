"""Audit the saved S-u/KKT/PCGrad policies against frozen Jiang--Fu 400-point references.

This is evaluation only: it neither retrains a policy nor solves a new NLP.
The three checkpoints are the paired seed-20260771 exploratory policies:

* S-u: control supervision only;
* S-u+K: control supervision plus the unprojected discrete-KKT continuation;
* S-u+K-PCGrad: the same continuation with only conflicting KKT-gradient
  components projected away.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.run_two_stage_vs_kkt_only_ablation import (  # noqa: E402
    ExperimentConfig,
    Problem,
    dump_json,
    evaluate,
    set_seed,
)


SEED = 20260771
SOURCES = {
    "vdp": {
        "S-u": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_pcgrad_validation_v1/vdp/seed20260771/S-u.pth",
        "S-u+K": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_v2/vdp/seed20260771/S-u+K.pth",
        "S-u+K-PCGrad": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_pcgrad_validation_v1/vdp/seed20260771/S-u+K.pth",
    },
    "penicillin": {
        "S-u": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_pcgrad_validation_v1/penicillin/seed20260771/S-u.pth",
        "S-u+K": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_v2/penicillin/seed20260771/S-u+K.pth",
        "S-u+K-PCGrad": ROOT / "kkt_collocation/results/exploratory_vdp_penicillin_su_k10_pcgrad_validation_v1/penicillin/seed20260771/S-u+K.pth",
    },
}


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in SOURCES[args.benchmark].values():
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    # These match the saved 200+10 policies.  The model architecture and
    # normalisation come from the established fixed training labels.
    cfg = ExperimentConfig(continuation_epochs=10, kkt_weight_vdp=1e-2, kkt_weight_penicillin=1e-2)
    problem = Problem(args.benchmark, device, SEED, cfg)
    if len(problem.test_states) != 400 or len(problem.reference_records) != 400:
        raise RuntimeError("Expected the frozen matched 400-point test/reference pair")

    methods: dict[str, dict] = {}
    for method, checkpoint_path in SOURCES[args.benchmark].items():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = problem.make_model()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print(f"[{args.benchmark}] auditing {method} on 400 frozen points", flush=True)
        rows, deployment = evaluate(problem, method, model, 0.0, workers=args.workers)
        write_rows(args.output / f"per_sample_{method}.csv", rows)
        methods[method] = {
            "checkpoint": str(checkpoint_path),
            "deployment": deployment,
        }

    result = {
        "formal_protocol": False,
        "nonformal_exploration_note": "One fixed seed only; this comparison must not be used as a 20-seed formal conclusion.",
        "benchmark": args.benchmark,
        "seed": SEED,
        "test_samples": 400,
        "reference": {
            "source": "kkt_collocation/results/jiang_fu_matched400_comparison/per_point_seed.csv",
            "protocol": "pointwise frozen Jiang--Fu cold-start objective, solve time, and continuous-time audit record",
            "reference_samples": len(problem.reference_records),
        },
        "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.",
        "methods": methods,
    }
    dump_json(args.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
