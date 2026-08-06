"""Two-dimensional JCB VALC experiment with Jiang--Fu Algorithm-1 labels.

The teacher-label file is produced by ``run_JCB_2D_VALC_labels.m`` from the
authors' published MATLAB implementation.  This script never substitutes a
Python NLP for those labels.  It trains a supervised policy and an optional
KKT-refined copy, selects between them using a disjoint raw-HDS validation
gate, and applies the event-located HDS correction on a disjoint test cohort.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import nnls
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual


@dataclass(frozen=True)
class Config:
    horizon: float = 1.0
    zoh_steps: int = 20
    subnodes_per_zoh: int = 10
    u_min: float = -15.0
    u_max: float = 15.0
    x1_bounds: tuple[float, float] = (-0.1, 0.1)
    x2_bounds: tuple[float, float] = (-1.1, -0.9)
    validation_points: int = 60
    test_points: int = 400
    epochs: int = 1500
    kkt_epochs: int = 50
    seed: int = 20260781
    margin: float = 1e-6
    lambda_grid: int = 101
    value_weight: float = 0.1
    kkt_weight: float = 1e-3
    @property
    def dt(self) -> float: return self.horizon / self.zoh_steps


@dataclass(frozen=True)
class DirectGateAudit:
    """The manuscript's gate in the native physical units of ``g``."""
    sample_count: int
    severe_magnitude_threshold: float
    allowed_severe_rate: float
    R_sev: float
    M_sev: float
    triggered_by_rate: bool
    triggered_by_peak: bool
    kkt_refinement_required: bool


def direct_hds_gate(hds_peaks: list[float], *, epsilon_g: float = 1e-2,
                    rho_g: float = 5e-2) -> DirectGateAudit:
    """Apply Eq. (9) directly, without a constraint-normalization scale."""
    v = np.maximum(np.asarray(hds_peaks, dtype=float), 0.0)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("HDS peaks must be a nonempty finite vector.")
    rate = float(np.mean(v > epsilon_g)); maximum = float(v.max())
    by_rate, by_peak = rate > rho_g, maximum > epsilon_g
    return DirectGateAudit(len(v), epsilon_g, rho_g, rate, maximum,
                           by_rate, by_peak, by_rate or by_peak)


def ode(_t: float, z: np.ndarray, u: float) -> np.ndarray:
    return np.array([z[1], -z[1] + u, 1.0, z[0] ** 2 + z[1] ** 2 + .005 * u ** 2])


def g(z: np.ndarray) -> float:
    return float(z[1] - 8.0 * (z[2] - .5) ** 2 + .5)


def gdot(z: np.ndarray, u: float) -> float:
    return float(-z[1] + u - 16.0 * (z[2] - .5))


def initial(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], p[1], 0.0, 0.0], dtype=float)


def objective_np(p: np.ndarray, controls: np.ndarray, cfg: Config) -> float:
    x1, x2 = float(p[0]), float(p[1]); cost = 0.0
    xi, wi = np.polynomial.legendre.leggauss(5)
    q = .5 * cfg.dt * (xi + 1.0)
    for u in np.asarray(controls, dtype=float):
        e = np.exp(-q)
        qx2 = u + (x2 - u) * e
        qx1 = x1 + u * q + (x2 - u) * (1.0 - e)
        cost += .5 * cfg.dt * np.sum(wi * (qx1 * qx1 + qx2 * qx2 + .005 * u * u))
        e1 = np.exp(-cfg.dt)
        x1, x2 = x1 + u * cfg.dt + (x2 - u) * (1.0 - e1), u + (x2 - u) * e1
    return float(cost)


def path_nodes_np(p: np.ndarray, controls: np.ndarray, cfg: Config) -> np.ndarray:
    x1, x2 = float(p[0]), float(p[1]); values: list[float] = []
    for k, u in enumerate(np.asarray(controls, dtype=float)):
        tau = np.linspace(0.0, cfg.dt, cfg.subnodes_per_zoh + 1)
        e = np.exp(-tau)
        loc_x2 = u + (x2 - u) * e
        t = k * cfg.dt + tau
        local = loc_x2 - 8.0 * (t - .5) ** 2 + .5
        values.extend(local if k == 0 else local[1:])
        e1 = np.exp(-cfg.dt)
        x1, x2 = x1 + u * cfg.dt + (x2 - u) * (1.0 - e1), u + (x2 - u) * e1
    return np.asarray(values, dtype=float)


def reconstruct_path_duals(p: np.ndarray, controls: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """Fixed-grid, reduced-NLP multiplier reconstruction for KKT fine tuning.

    The teacher controls themselves remain Jiang--Fu Algorithm-1 solutions.
    These finite-dimensional duals are only optional numerical labels for the
    refinement loss; they are not continuous-time multiplier functions.
    """
    u = np.asarray(controls, dtype=float); h = 1e-5; n = len(u)
    grad_j = np.empty(n); jac_g = np.empty((cfg.zoh_steps * cfg.subnodes_per_zoh + 1, n))
    for j in range(n):
        d = np.zeros(n); d[j] = h
        grad_j[j] = (objective_np(p, u + d, cfg) - objective_np(p, u - d, cfg)) / (2.0 * h)
        jac_g[:, j] = (path_nodes_np(p, u + d, cfg) - path_nodes_np(p, u - d, cfg)) / (2.0 * h)
    gv = path_nodes_np(p, u, cfg); active = np.flatnonzero(gv > -1e-3)
    dual = np.zeros_like(gv)
    if len(active):
        dual[active], _ = nnls(jac_g[active].T, -grad_j)
    residual = float(np.linalg.norm(grad_j + jac_g.T @ dual, ord=np.inf))
    return dual, residual


class Policy(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, 96), nn.Tanh(), nn.Linear(96, 192), nn.Tanh(), nn.Linear(192, 96), nn.Tanh())
        self.value = nn.Linear(96, 1)
        self.control = nn.Linear(96, n)
    def forward(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(p)
        return self.value(h), 15.0 * torch.tanh(self.control(h))


def torch_rollout(p: torch.Tensor, controls: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    x1, x2 = p[:, 0], p[:, 1]; cost = torch.zeros_like(x1); gs: list[torch.Tensor] = []
    xi, wi = np.polynomial.legendre.leggauss(5)
    q = torch.tensor(.5 * cfg.dt * (xi + 1.0), dtype=p.dtype, device=p.device)[None, :]
    w = torch.tensor(wi, dtype=p.dtype, device=p.device)[None, :]
    local_tau = torch.linspace(0.0, cfg.dt, cfg.subnodes_per_zoh + 1, dtype=p.dtype, device=p.device)[None, :]
    for k in range(cfg.zoh_steps):
        u = controls[:, k]
        e = torch.exp(-q); qx2 = u[:, None] + (x2[:, None] - u[:, None]) * e
        qx1 = x1[:, None] + u[:, None] * q + (x2[:, None] - u[:, None]) * (1.0 - e)
        cost = cost + .5 * cfg.dt * (w * (qx1.square() + qx2.square() + .005 * u[:, None].square())).sum(1)
        e_local = torch.exp(-local_tau)
        loc_x2 = u[:, None] + (x2[:, None] - u[:, None]) * e_local
        t = k * cfg.dt + local_tau
        local_g = loc_x2 - 8.0 * (t - .5).square() + .5
        gs.append(local_g if k == 0 else local_g[:, 1:])
        e1 = np.exp(-cfg.dt)
        x1, x2 = x1 + u * cfg.dt + (x2 - u) * (1.0 - e1), u + (x2 - u) * e1
    return cost, torch.cat(gs, 1)


def lhs(n: int, cfg: Config, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); out = np.empty((n, 2))
    for j, bounds in enumerate((cfg.x1_bounds, cfg.x2_bounds)):
        out[:, j] = bounds[0] + (rng.permutation(n) + rng.random(n)) / n * (bounds[1] - bounds[0])
    return out


def load_teacher(path: Path, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = loadmat(path)
    if not np.all(np.asarray(d['completed']).reshape(-1)):
        raise RuntimeError('Jiang--Fu label generation has not completed.')
    states = np.asarray(d['initialStates'], dtype=np.float64)
    controls = np.asarray(d['controls'], dtype=np.float64)
    objective = np.asarray(d['objectives'], dtype=np.float64).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 2 or controls.shape != (len(states), cfg.zoh_steps):
        raise ValueError(f'Unexpected teacher shapes: {states.shape}, {controls.shape}')
    duals, residuals = [], []
    for i, (p, u) in enumerate(zip(states, controls), start=1):
        dual, residual = reconstruct_path_duals(p, u, cfg); duals.append(dual); residuals.append(residual)
        if i % 50 == 0: print(f'reconstructed KKT duals {i}/{len(states)}')
    return states, controls, objective, np.asarray(duals), np.asarray(residuals)


def train_branch(states: np.ndarray, controls: np.ndarray, objective: np.ndarray, duals: np.ndarray, cfg: Config, *, kkt: bool, base: Policy | None = None) -> tuple[Policy, np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed + (1 if kkt else 0)); device = torch.device('cpu')
    p = torch.tensor(states, dtype=torch.float32, device=device); uref = torch.tensor(controls, dtype=torch.float32, device=device)
    jref = torch.tensor(objective[:, None], dtype=torch.float32, device=device); dual = torch.tensor(duals, dtype=torch.float32, device=device)
    mean = p.mean(0); std = p.std(0, unbiased=False).clamp_min(1e-6); jm = jref.mean(); js = jref.std(unbiased=False).clamp_min(1e-6)
    model = Policy(cfg.zoh_steps).to(device) if base is None else base.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if base is None else 1e-5)
    anchor = None
    if kkt:
        with torch.no_grad(): anchor = model((p - mean) / std)[1].detach()
    for epoch in range(cfg.kkt_epochs if kkt else cfg.epochs):
        pred_j, u = model((p - mean) / std)
        loss = nn.functional.mse_loss(u, uref) + cfg.value_weight * nn.functional.mse_loss(pred_j, (jref - jm) / js)
        if kkt:
            jroll, nodes = torch_rollout(p, u, cfg)
            residual = augmented_lagrangian_kkt_residual(jroll, u, nodes, dual, 10.0).total
            loss = loss + cfg.kkt_weight * residual / residual.detach().clamp_min(1.0) + .1 * nn.functional.mse_loss(u, anchor)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        report_every = 10 if kkt else 250
        if (epoch + 1) % report_every == 0: print(f"{'KKT' if kkt else 'S'} epoch {epoch + 1}: loss={loss.item():.4e}")
    return model.cpu().eval(), mean.cpu().numpy(), std.cpu().numpy()


def predict(model: Policy, mean: np.ndarray, std: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    x = torch.tensor((p - mean) / std, dtype=torch.float32); start = time.perf_counter()
    with torch.no_grad(): _, u = model(x)
    return u.numpy(), (time.perf_counter() - start) / len(p)


def evaluate(name: str, points: np.ndarray, controls: np.ndarray, inference: float, cfg: Config, corrector: HDSLambdaCorrector) -> list[dict]:
    rows = []
    for i, (p, u) in enumerate(zip(points, controls)):
        raw = corrector.audit(initial(p), u, cfg.dt); start = time.perf_counter(); outcome = corrector.correct(initial(p), u, cfg.dt); elapsed = time.perf_counter() - start
        accepted = bool(outcome.accepted); applied = outcome.controls if accepted else None
        rows.append({'method': name, 'index': i, 'x1_0': p[0], 'x2_0': p[1], 'raw_hds_max_g': raw,
                     'accepted': accepted, 'fallback': not accepted, 'applied_hds_max_g': np.nan if not accepted else corrector.audit(initial(p), applied, cfg.dt),
                     'nominal_objective': objective_np(p, u, cfg), 'applied_objective': np.nan if not accepted else objective_np(p, applied, cfg),
                     'corrected_segments': int(sum(s.corrected for s in outcome.segments)), 'inference_seconds': inference, 'hds_seconds': elapsed})
    return rows


def summary(rows: list[dict]) -> dict:
    raw = np.asarray([r['raw_hds_max_g'] for r in rows]); accepted = np.asarray([r['accepted'] for r in rows], bool); applied = np.asarray([r['applied_hds_max_g'] for r in rows], float)
    return {'points': len(rows), 'raw_violation_rate_percent': float(100 * np.mean(raw > 1e-8)), 'raw_max_g': float(raw.max()),
            'hds_acceptance_rate_percent': float(100 * np.mean(accepted)), 'fallback_rate_percent': float(100 * np.mean(~accepted)),
            'accepted_max_g': float(np.nanmax(applied)), 'mean_corrected_segments': float(np.mean([r['corrected_segments'] for r in rows])),
            'mean_total_seconds': float(np.mean([r['inference_seconds'] + r['hds_seconds'] for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1500); parser.add_argument('--kkt-epochs', type=int, default=50)
    parser.add_argument('--lambda-grid', type=int, default=101)
    parser.add_argument('--zoh-steps', type=int, default=20)
    parser.add_argument('--validation-points', type=int, default=60)
    parser.add_argument('--test-points', type=int, default=400)
    parser.add_argument('--x1-min', type=float, default=-.1)
    parser.add_argument('--x1-max', type=float, default=.1)
    parser.add_argument('--x2-min', type=float, default=-1.1)
    parser.add_argument('--x2-max', type=float, default=-.9)
    parser.add_argument('--supervised-only', action='store_true', help='Train and checkpoint only the supervised branch, then stop.')
    parser.add_argument('--skip-kkt', action='store_true', help='Evaluate the validation gate and supervised branch without running the optional KKT candidate.')
    parser.add_argument('--force-kkt', action='store_true', help='Diagnostic only: deploy the KKT candidate even when the gate retains supervision.')
    args = parser.parse_args(); cfg = Config(zoh_steps=args.zoh_steps, epochs=args.epochs, kkt_epochs=args.kkt_epochs,
        lambda_grid=args.lambda_grid, validation_points=args.validation_points,
        test_points=args.test_points, x1_bounds=(args.x1_min, args.x1_max),
        x2_bounds=(args.x2_min, args.x2_max)); args.output.mkdir(parents=True, exist_ok=True)
    cached_teacher = args.output/'teacher_labels.npz'
    if cached_teacher.exists():
        cached = np.load(cached_teacher)
        states, labels, objectives = cached['initial_state'], cached['controls'], cached['objective']
        duals, residuals = cached['path_duals'], cached['kkt_reconstruction_residual']
        print('Loaded cached Jiang--Fu teacher controls and reconstructed KKT labels.')
    else:
        states, labels, objectives, duals, residuals = load_teacher(args.labels, cfg)
        np.savez_compressed(cached_teacher, initial_state=states, controls=labels, objective=objectives, path_duals=duals, kkt_reconstruction_residual=residuals)
    supervised_path = args.output/'jcb2d_supervised.pth'
    if supervised_path.exists():
        payload = torch.load(supervised_path, map_location='cpu', weights_only=False)
        supervised = Policy(cfg.zoh_steps); supervised.load_state_dict(payload['model']); supervised.eval()
        mean_s, std_s = np.asarray(payload['mean']), np.asarray(payload['std'])
        print('Loaded the completed supervised checkpoint.')
    else:
        supervised, mean_s, std_s = train_branch(states, labels, objectives, duals, cfg, kkt=False)
        torch.save({'model': supervised.state_dict(), 'mean': mean_s, 'std': std_s, 'config': asdict(cfg)}, supervised_path)
    if args.supervised_only:
        print(json.dumps({'teacher_labels': len(states), 'supervised_checkpoint': str(supervised_path), 'epochs': cfg.epochs}, indent=2))
        return
    if args.skip_kkt:
        refined, mean_k, std_k = copy.deepcopy(supervised), mean_s, std_s
    else:
        refined, mean_k, std_k = train_branch(states, labels, objectives, duals, cfg, kkt=True, base=copy.deepcopy(supervised))
    corrector = HDSLambdaCorrector(ode, g, gdot, (cfg.u_min, cfg.u_max), HDSLambdaConfig(grid_size=cfg.lambda_grid, safety_margin=cfg.margin, max_step_fraction=200.0))
    validation = lhs(cfg.validation_points, cfg, cfg.seed + 1); usv, _ = predict(supervised, mean_s, std_s, validation)
    gate = direct_hds_gate([corrector.audit(initial(p), u, cfg.dt) for p, u in zip(validation, usv)])
    selected, model, mean, std = ('KKT-refined', refined, mean_k, std_k) if (args.force_kkt or gate.kkt_refinement_required) else ('Supervised', supervised, mean_s, std_s)
    test = lhs(cfg.test_points, cfg, cfg.seed + 2); u, infer = predict(model, mean, std, test); rows = evaluate(f'Adaptive ({selected}) + HDS', test, u, infer, cfg, corrector)
    with (args.output/'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {'config': asdict(cfg), 'teacher': f'Jiang--Fu Algorithm 1; {cfg.zoh_steps} ZOH controls; see the MATLAB label metadata for the recorded label initialisation.',
              'initial_domain': f'x1(0) in {cfg.x1_bounds}, x2(0) in {cfg.x2_bounds}', 'teacher_labels': len(states),
              'kkt_note': 'Finite-dimensional fixed-node dual reconstruction from Jiang--Fu controls; not a continuous-time multiplier function.',
              'kkt_candidate_trained': not args.skip_kkt,
              'selection_mode': 'forced-KKT diagnostic' if args.force_kkt else 'validation gate',
              'kkt_reconstruction_max_residual': float(residuals.max()), 'gate': asdict(gate), 'selected_branch': selected, 'test': summary(rows)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    torch.save({'supervised': supervised.state_dict(), 'kkt_refined': refined.state_dict(), 'normalization': {'S': [mean_s, std_s], 'KKT': [mean_k, std_k]}, 'config': asdict(cfg)}, args.output/'models.pth')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
