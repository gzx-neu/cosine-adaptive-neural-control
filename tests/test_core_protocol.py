"""Fast release checks that do not train a network or write result files."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from offline_safe_control.adaptive_event_hds import (
    AdaptiveEventHDSConfig,
    AdaptiveEventHDSCorrector,
)
from kkt_collocation.run_unified_su_suj_sk_konly_ablation import _optimizer_step


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_declaration(self) -> None:
        protocol = json.loads((ROOT / "configs/paper_30seed.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["seeds"]["count"], 30)
        self.assertEqual(protocol["training"]["lambda_base_grid_points"], 31)
        self.assertTrue(protocol["audit"]["nominal_lambda_audited_separately"])
        self.assertEqual(protocol["audit"]["acceptance_threshold"], -1e-6)
        self.assertEqual(protocol["audit"]["solver"], "DOP853")

    def test_cosine_adaptive_projection(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
        model = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        base = 0.1 * parameter[0]
        kkt = 0.1 * (-0.5 * parameter[0] + math.sqrt(0.75) * parameter[1])
        result = _optimizer_step(
            model,
            optimizer,
            base + kkt,
            base_loss=base,
            kkt_loss=kkt,
            cosine_adaptive_kkt_conflict_projection=True,
        )
        self.assertAlmostEqual(result[1], 0.5, places=12)
        self.assertAlmostEqual(result[3], -0.5, places=12)
        np.testing.assert_allclose(
            parameter.grad.detach().numpy(),
            np.asarray([0.075, 0.1 * math.sqrt(0.75)]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_adaptive_event_hds_negative_margin_and_grid(self) -> None:
        def ode(_time: float, state: np.ndarray, control: float) -> np.ndarray:
            return np.asarray([control], dtype=float)

        def constraint(state: np.ndarray) -> float:
            return float(state[0])

        def derivative(_state: np.ndarray, control: float) -> float:
            return float(control)

        config = AdaptiveEventHDSConfig(grid_size=31, safety_margin=1e-6)
        corrector = AdaptiveEventHDSCorrector(ode, constraint, derivative, (0.0, 0.2), config)
        candidates = corrector._candidate_lambdas(0.17, exclude_nominal=True)
        self.assertLessEqual(len(candidates), 31)
        self.assertFalse(np.any(np.isclose(candidates, 1.0)))

        result = corrector.correct(np.asarray([-0.1]), np.asarray([0.2]), 1.0)
        self.assertTrue(result.accepted)
        self.assertFalse(result.requires_reoptimization)
        self.assertLessEqual(corrector.audit(np.asarray([-0.1]), result.controls, 1.0), -1e-6)

    def test_cstr_modules_import_as_package(self) -> None:
        import kkt_collocation.economou_cstr_hds_fast  # noqa: F401
        import kkt_collocation.evaluate_unified_economou_cstr_n100_hds  # noqa: F401
        import kkt_collocation.train_unified_economou_cstr_n100_ablation  # noqa: F401


if __name__ == "__main__":
    unittest.main()
