"""Reproducible non-isothermal exothermic-CSTR experiment.

This script adds one representative literature-informed process simulation to the
offline policy / HDS safety-correction workflow.  It deliberately models a
mechanistic CSTR, not plant data.  The manipulated input is a nonnegative
jacket heat-removal rate; hence a larger positive scalar-lambda candidate
means stronger cooling, while HDS remains the only acceptance test.

The protocol is intentionally compact enough to rerun on CPU:
  1. solve a direct-RK4 constrained NLP at a 12 x 12 operating-domain grid;
  2. train a bounded policy-value network and a KKT-refined copy;
  3. select one model from an independent HDS validation set;
  4. evaluate nominal and HDS-corrected policies on a frozen test set;
  5. save raw labels, checkpoints, per-sample CSV, JSON summary and figures.

References for the mechanistic mass/energy balance and direct transcription
are recorded in the manuscript bibliography (Henson--Seborg; Biegler).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import BFGS, NonlinearConstraint, minimize, nnls

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual


@dataclass(frozen=True)
class CSTRConfig:
    """Dimensionally explicit, jacket-cooled exothermic-CSTR protocol (min, K)."""
    # C_A dynamics: dC_A/dt=(C_Af-C_A)/tau-k0 exp(-EoverR/T) C_A.
    feed_concentration: float = 1.0       # mol L^-1
    feed_temperature: float = 350.0       # K
    residence_time: float = 1.0           # min
    pre_exponential: float = 7.2e10       # min^-1
    activation_temperature: float = 8750.0  # E/R, K
    heat_release_scale: float = 50.0      # (-Delta H)/(rho Cp), K L mol^-1
    horizon: float = 1.5                  # min
    zoh_steps: int = 10
    substeps_per_zoh: int = 5
    cooling_min: float = 0.0              # K min^-1, positive heat removal
    cooling_max: float = 100.0            # K min^-1
    temperature_max: float = 365.0        # K
    ca_range: tuple[float, float] = (0.60, 1.00)
    temperature_range: tuple[float, float] = (345.0, 360.0)
    collocation_margin: float = 1e-2      # K; HDS is still decisive
    hds_tolerance: float = 1e-8
    cooling_cost: float = 1e-4
    solver_maxiter: int = 1200
    seed: int = 20260718

    @property
    def total_substeps(self) -> int:
        return self.zoh_steps * self.substeps_per_zoh

    @property
    def node_count(self) -> int:
        return self.total_substeps + 1

    @property
    def dt(self) -> float:
        return self.horizon / self.total_substeps

    @property
    def zoh_duration(self) -> float:
        return self.horizon / self.zoh_steps


def _casadi():
    package = ROOT / "third_party" / "casadi"
    dll = package / "casadi"
    if str(package) not in sys.path:
        sys.path.append(str(package))
    if dll.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll))
    import casadi as ca
    return ca


def cstr_ode(_time: float, x: np.ndarray, cooling: float, cfg: CSTRConfig) -> np.ndarray:
    ca, temperature = float(x[0]), float(x[1])
    rate = cfg.pre_exponential * np.exp(-cfg.activation_temperature / max(temperature, 1.0)) * ca
    return np.array([
        (cfg.feed_concentration - ca) / cfg.residence_time - rate,
        (cfg.feed_temperature - temperature) / cfg.residence_time + cfg.heat_release_scale * rate - cooling,
    ])


def path_constraint(x: np.ndarray, cfg: CSTRConfig) -> float:
    return float(x[1] - cfg.temperature_max)


def path_derivative(x: np.ndarray, cooling: float, cfg: CSTRConfig) -> float:
    return float(cstr_ode(0.0, x, cooling, cfg)[1])


class CSTRTranscription:
    """Direct RK4 transcription with CasADi derivatives and trust-constr duals."""
    def __init__(self, cfg: CSTRConfig) -> None:
        self.cfg, self.ca = cfg, _casadi()
        self.state_size = 2
        self.control_offset = self.state_size * cfg.node_count
        self.decision_size = self.control_offset + cfg.zoh_steps
        self.equality_count = self.state_size * cfg.node_count
        self._build()

    def _f(self, x: Any, cooling: Any):
        c = self.ca
        ca_, temp = x[0], x[1]
        rate = self.cfg.pre_exponential * c.exp(-self.cfg.activation_temperature / temp) * ca_
        return c.vertcat(
            (self.cfg.feed_concentration - ca_) / self.cfg.residence_time - rate,
            (self.cfg.feed_temperature - temp) / self.cfg.residence_time + self.cfg.heat_release_scale * rate - cooling,
        )

    def _rk4(self, x: Any, cooling: Any):
        h = self.cfg.dt
        k1 = self._f(x, cooling); k2 = self._f(x + .5*h*k1, cooling)
        k3 = self._f(x + .5*h*k2, cooling); k4 = self._f(x + h*k3, cooling)
        return x + h*(k1 + 2*k2 + 2*k3 + k4)/6

    def _build(self) -> None:
        c, cfg = self.ca, self.cfg
        z = c.SX.sym("z", self.decision_size); p = c.SX.sym("p", self.state_size)
        X = c.reshape(z[:self.control_offset], self.state_size, cfg.node_count)
        U = z[self.control_offset:]
        eq = [X[:, 0] - p]
        for j in range(cfg.total_substeps):
            eq.append(X[:, j+1] - self._rk4(X[:, j], U[j // cfg.substeps_per_zoh]))
        g = [X[1, j] - (cfg.temperature_max - cfg.collocation_margin) for j in range(cfg.node_count)]
        objective = X[0, -1] + cfg.cooling_cost * cfg.zoh_duration * c.sumsqr(U)
        constraints = c.vertcat(*(eq + g))
        self.objective = c.Function("cstr_objective", [z, p], [objective])
        self.gradient = c.Function("cstr_gradient", [z, p], [c.gradient(objective, z)])
        self.constraints = c.Function("cstr_constraints", [z, p], [constraints])
        self.jacobian = c.Function("cstr_jacobian", [z, p], [c.jacobian(constraints, z)])
        self.lower = np.r_[np.full(self.equality_count, 0.0), np.full(cfg.node_count, -np.inf)]
        self.upper = np.zeros(self.equality_count + cfg.node_count)
        self.bounds_lo = np.full(self.decision_size, -np.inf); self.bounds_hi = np.full(self.decision_size, np.inf)
        self.bounds_lo[self.control_offset:] = cfg.cooling_min; self.bounds_hi[self.control_offset:] = cfg.cooling_max

    def initial_guess(self, state0: np.ndarray, controls: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        if controls is None:
            # A deliberately conservative thermal-balance estimate helps the NLP start feasible.
            rate = cfg.pre_exponential*np.exp(-cfg.activation_temperature/state0[1])*state0[0]
            base = np.clip((cfg.feed_temperature-state0[1])/cfg.residence_time + cfg.heat_release_scale*rate + 5.0,
                           cfg.cooling_min, cfg.cooling_max)
            controls = np.full(cfg.zoh_steps, base)
        x = np.asarray(state0, dtype=float).copy(); nodes = [x.copy()]
        for j in range(cfg.total_substeps):
            u = float(controls[j // cfg.substeps_per_zoh]); h = cfg.dt
            k1=cstr_ode(0,x,u,cfg); k2=cstr_ode(0,x+.5*h*k1,u,cfg)
            k3=cstr_ode(0,x+.5*h*k2,u,cfg); k4=cstr_ode(0,x+h*k3,u,cfg)
            x=x+h*(k1+2*k2+2*k3+k4)/6; nodes.append(x.copy())
        return np.r_[np.asarray(nodes).T.reshape(-1, order="F"), controls]

    def solve(self, state0: np.ndarray, controls: np.ndarray | None = None) -> dict[str, Any]:
        p = np.asarray(state0, dtype=float)
        fun=lambda z: float(self.objective(z,p))
        jac=lambda z: np.asarray(self.gradient(z,p),float).reshape(-1)
        con=lambda z: np.asarray(self.constraints(z,p),float).reshape(-1)
        cjac=lambda z: np.asarray(self.jacobian(z,p),float)
        # SLSQP is markedly more reliable than trust-constr on this small but
        # stiff shooting transcription.  CasADi supplies its exact first
        # derivatives; path duals below are reconstructed from the resulting
        # discretized stationarity system (and are kept separate from any
        # continuous-time multiplier interpretation).
        def equalities(z): return con(z)[:self.equality_count]
        def equalities_jac(z): return cjac(z)[:self.equality_count]
        def inequalities(z): return -con(z)[self.equality_count:]
        def inequalities_jac(z): return -cjac(z)[self.equality_count:]
        start=time.perf_counter()
        result=minimize(fun, self.initial_guess(p,controls), method="SLSQP", jac=jac,
            bounds=list(zip(self.bounds_lo,self.bounds_hi)),
            constraints=[{"type":"eq","fun":equalities,"jac":equalities_jac},
                         {"type":"ineq","fun":inequalities,"jac":inequalities_jac}],
            options={"maxiter":self.cfg.solver_maxiter,"ftol":1e-9,"disp":False})
        if not result.success:
            raise RuntimeError(result.message)
        z=np.asarray(result.x); u=z[self.control_offset:].copy()
        # Multipliers are reconstructed at active discretized path nodes by
        # nonnegative least squares after eliminating unrestricted equality
        # multipliers.  They are only optional reduced NLP labels for the
        # KKT-refinement branch, never continuous-time multipliers.
        jacobian = cjac(z); grad = jac(z); path_values = con(z)[self.equality_count:]
        active = np.flatnonzero(path_values > -1e-3)
        duals = np.zeros(self.cfg.node_count)
        if active.size:
            e = jacobian[:self.equality_count].T
            a = jacobian[self.equality_count + active].T
            design = np.c_[e, -e, a]
            coeff, _ = nnls(design, -grad)
            duals[active] = coeff[2*self.equality_count:]
        corrector=make_corrector(self.cfg)
        peak=corrector.audit(p,u,self.cfg.zoh_duration)
        label_hds_corrected = False
        if peak>self.cfg.hds_tolerance:
            # The discretization is intentionally not used as a continuous-time
            # certificate.  A rare between-node excursion is repaired by the
            # same minimum-intervention HDS procedure used at deployment; an
            # empty candidate set remains a hard data-generation failure.
            repaired = corrector.correct(p, u, self.cfg.zoh_duration)
            if not repaired.accepted:
                raise RuntimeError(f"HDS rejection after NLP and correction: peak={peak:.3e}")
            u = repaired.controls
            peak = corrector.audit(p, u, self.cfg.zoh_duration)
            label_hds_corrected = True
        return {"initial_state":p,"optimal_controls":u,"objective":objective(p,u,self.cfg),"path_duals":duals,
                "hds_max_g":float(peak),"label_hds_corrected":label_hds_corrected,"solve_seconds":time.perf_counter()-start}


def make_corrector(cfg: CSTRConfig) -> HDSLambdaCorrector:
    return HDSLambdaCorrector(lambda t,x,u: cstr_ode(t,x,u,cfg), lambda x:path_constraint(x,cfg),
        lambda x,u:path_derivative(x,u,cfg), (cfg.cooling_min,cfg.cooling_max),
        HDSLambdaConfig(grid_size=31, safety_margin=cfg.hds_tolerance, max_step_fraction=75.0))


def grid_states(cfg: CSTRConfig, size: int) -> np.ndarray:
    ca=np.linspace(*cfg.ca_range,size); temp=np.linspace(*cfg.temperature_range,size)
    return np.array([[a,t] for a in ca for t in temp],float)


def lhs_states(cfg: CSTRConfig, n: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed); out=np.empty((n,2))
    for j,bounds in enumerate((cfg.ca_range,cfg.temperature_range)):
        out[:,j]=bounds[0]+(rng.permutation(n)+rng.random(n))/n*(bounds[1]-bounds[0])
    return out


class Policy(nn.Module):
    def __init__(self,cfg:CSTRConfig):
        super().__init__(); self.cfg=cfg
        self.body=nn.Sequential(nn.Linear(2,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.value=nn.Linear(128,1); self.control=nn.Linear(128,cfg.zoh_steps)
    def forward(self,x):
        z=self.body(x); return self.value(z), self.cfg.cooling_max*torch.sigmoid(self.control(z))


def torch_rollout(states:torch.Tensor, controls:torch.Tensor, cfg:CSTRConfig):
    x=states; gs=[x[:,1]-cfg.temperature_max]; h=cfg.dt
    for j in range(cfg.total_substeps):
        u=controls[:,j//cfg.substeps_per_zoh]
        def f(q):
            rate=cfg.pre_exponential*torch.exp(-cfg.activation_temperature/q[:,1])*q[:,0]
            return torch.stack(((cfg.feed_concentration-q[:,0])/cfg.residence_time-rate,
              (cfg.feed_temperature-q[:,1])/cfg.residence_time+cfg.heat_release_scale*rate-u),1)
        k1=f(x); k2=f(x+.5*h*k1); k3=f(x+.5*h*k2); k4=f(x+h*k3); x=x+h*(k1+2*k2+2*k3+k4)/6; gs.append(x[:,1]-cfg.temperature_max)
    j=x[:,0]+cfg.cooling_cost*cfg.zoh_duration*controls.square().sum(1)
    return j,torch.stack(gs,1)


def train_branch(data:dict[str,np.ndarray], cfg:CSTRConfig, *, kkt:bool, epochs:int, seed:int,
                 base:Policy|None=None, rollout_weight:float=0.0, value_weight:float=0.1):
    torch.manual_seed(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state=torch.tensor(data["initial_state"],dtype=torch.float32,device=device); refu=torch.tensor(data["optimal_controls"],dtype=torch.float32,device=device)
    refj=torch.tensor(data["objective"],dtype=torch.float32,device=device)[:,None]; dual=torch.tensor(data["path_duals"],dtype=torch.float32,device=device)
    mean,std=state.mean(0),state.std(0, unbiased=False).clamp_min(1e-6); jm,js=refj.mean(),refj.std(unbiased=False).clamp_min(1e-6)
    model=Policy(cfg).to(device) if base is None else base.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3 if base is None else 1e-5)
    norm=(state-mean)/std
    for _ in range(epochs):
        predj,u=model(norm); sup=nn.functional.mse_loss(u,refu)+value_weight*nn.functional.mse_loss(predj,(refj-jm)/js); loss=sup
        if rollout_weight:
            rollout_j,_=torch_rollout(state,u,cfg)
            consistency=nn.functional.mse_loss(predj*js+jm,rollout_j[:,None])
            loss=loss+rollout_weight*consistency
        if kkt:
            j,g=torch_rollout(state,u,cfg); residual=augmented_lagrangian_kkt_residual(j,u,g,dual,10.0).total
            loss=loss+1e-3*residual/residual.detach().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model.cpu(), mean.cpu().numpy(),std.cpu().numpy(),device


def predict(model:Policy, mean:np.ndarray,std:np.ndarray, states:np.ndarray):
    device=torch.device("cpu"); model=model.to(device).eval(); x=torch.tensor((states-mean)/std,dtype=torch.float32)
    start=time.perf_counter()
    with torch.no_grad(): _,u=model(x)
    return u.numpy(),(time.perf_counter()-start)/len(states)


def evaluate(name:str, states:np.ndarray, nominal:np.ndarray, cfg:CSTRConfig, inference:float, correct:bool=True):
    cor=make_corrector(cfg); rows=[]
    for i,(x,u) in enumerate(zip(states,nominal)):
        raw=cor.audit(x,u,cfg.zoh_duration); jraw=objective(x,u,cfg)
        start=time.perf_counter(); outcome=cor.correct(x,u,cfg.zoh_duration) if correct else None; elapsed=time.perf_counter()-start if correct else 0.
        accepted=True if outcome is None else outcome.accepted; applied=u if outcome is None or not accepted else outcome.controls
        peak=np.nan if not accepted else cor.audit(x,applied,cfg.zoh_duration); ja=np.nan if not accepted else objective(x,applied,cfg)
        seg=[] if outcome is None else list(outcome.segments); lambdas=np.array([s.lambda_value for s in seg if s.lambda_value is not None],float)
        rows.append({"method":name,"sample_index":i,"CA0":x[0],"T0":x[1],"nominal_hds_max_g":raw,"applied_hds_max_g":peak,
          "nominal_objective":jraw,"applied_objective":ja,"objective_change":ja-jraw if accepted else np.nan,
          "accepted":accepted,"fallback":not accepted,"corrected_segments":int(sum(s.corrected for s in seg)),
          "mean_abs_lambda_minus_one":float(np.mean(np.abs(lambdas-1))) if len(lambdas) else 0.,"inference_seconds":inference,"filter_seconds":elapsed})
    return rows


def objective(x:np.ndarray,u:np.ndarray,cfg:CSTRConfig)->float:
    from scipy.integrate import solve_ivp
    state=np.asarray(x,float)
    for control in u:
        sol=solve_ivp(lambda t,z:cstr_ode(t,z,float(control),cfg),(0,cfg.zoh_duration),state,method="DOP853",rtol=1e-10,atol=1e-12,max_step=cfg.zoh_duration/150)
        state=sol.y[:,-1]
    return float(state[0]+cfg.cooling_cost*cfg.zoh_duration*np.square(u).sum())


def summarise(rows:list[dict[str,Any]]):
    raw=np.asarray([r["nominal_hds_max_g"] for r in rows]); accepted=np.asarray([r["accepted"] for r in rows],bool)
    applied=np.asarray([r["applied_hds_max_g"] for r in rows],float); change=np.asarray([r["objective_change"] for r in rows],float)
    return {"samples":len(rows),"raw_violation_rate_percent":float(100*np.mean(raw>1e-8)),"raw_peak_max_K":float(raw.max()),
      "accepted_rate_percent":float(100*np.mean(accepted)),"fallback_rate_percent":float(100*np.mean(~accepted)),
      "corrected_segments_mean":float(np.mean([r["corrected_segments"] for r in rows])),"accepted_peak_max_K":float(np.nanmax(applied)),
      "mean_objective_change":float(np.nanmean(change)),"mean_inference_ms":float(1e3*np.mean([r["inference_seconds"] for r in rows])),
      "mean_hds_ms":float(1e3*np.mean([r["filter_seconds"] for r in rows]))}


def plot_results(rows:list[dict[str,Any]], states:np.ndarray, nominal:np.ndarray,cfg:CSTRConfig,out:Path):
    import matplotlib as mpl
    mpl.rcParams.update({"font.family":"Arial","font.size":7,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False})
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    cor=make_corrector(cfg); time_grid=np.linspace(0,cfg.horizon,301); rawT=[]; fixedT=[]
    for x,u in zip(states,nominal):
        def trajectory(control):
            state=x.copy(); ts=[]; ys=[]; offset=0.
            for v in control:
                local=np.linspace(0,cfg.zoh_duration,31); sol=solve_ivp(lambda t,z:cstr_ode(t,z,float(v),cfg),(0,cfg.zoh_duration),state,t_eval=local,rtol=1e-10,atol=1e-12,method="DOP853")
                ts.extend(offset+local if not ts else offset+local[1:]); ys.extend(sol.y[1] if not ys else sol.y[1,1:]); state=sol.y[:,-1]; offset+=cfg.zoh_duration
            return np.asarray(ts),np.asarray(ys)
        rr=cor.correct(x,u,cfg.zoh_duration); t,y=trajectory(u); rawT.append(np.interp(time_grid,t,y))
        if rr.accepted:
            t,y=trajectory(rr.controls); fixedT.append(np.interp(time_grid,t,y))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.3),sharey=True)
    violating=[i for i,r in enumerate(rows) if r["nominal_hds_max_g"]>cfg.hds_tolerance]
    for y in rawT: axes[0].plot(time_grid,y,color="#d9b8b8",lw=.5,alpha=.65)
    for y in fixedT: axes[1].plot(time_grid,y,color="#b8cbe0",lw=.5,alpha=.65)
    # The few corrected trajectories carry the visual evidence of the HDS
    # intervention without replacing the all-trajectory population display.
    for i in violating:
        axes[0].plot(time_grid,rawT[i],color="#b13a3a",lw=1.0,alpha=.9)
        if i < len(fixedT): axes[1].plot(time_grid,fixedT[i],color="#2f6f9f",lw=1.0,alpha=.95)
    for ax,title in zip(axes,("Nominal neural policy","After HDS--$\\lambda$ correction")):
        ax.axhline(cfg.temperature_max,color="#b13a3a",lw=.9,ls="--"); ax.set(xlabel="Time (min)",title=title,xlim=(0,cfg.horizon)); ax.grid(axis="y",color=".9",lw=.5)
    axes[0].set_ylabel("Reactor temperature $T$ (K)"); fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix(".svg"),bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(out.with_suffix(".png"),dpi=300,bbox_inches="tight"); plt.close(fig)


def plot_controls(rows:list[dict[str,Any]], states:np.ndarray, nominal:np.ndarray,cfg:CSTRConfig,out:Path):
    """Population ZOH-control evidence, preserving every frozen test input."""
    import matplotlib as mpl
    mpl.rcParams.update({"font.family":"Arial","font.size":7,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False})
    import matplotlib.pyplot as plt
    cor=make_corrector(cfg); t=np.linspace(0,cfg.horizon,cfg.zoh_steps+1)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.3),sharey=True)
    for x,u in zip(states,nominal):
        axes[0].step(t,np.r_[u,u[-1]],where="post",color="#d9b8b8",lw=.45,alpha=.6)
        result=cor.correct(x,u,cfg.zoh_duration)
        if result.accepted:
            v=result.controls; axes[1].step(t,np.r_[v,v[-1]],where="post",color="#b8cbe0",lw=.45,alpha=.6)
    for ax,title in zip(axes,("Nominal neural policy","After HDS--$\\lambda$ correction")):
        ax.axhline(cfg.cooling_min,color=".45",lw=.65,ls=":"); ax.axhline(cfg.cooling_max,color=".45",lw=.65,ls=":")
        ax.set(xlabel="Time (min)",title=title,xlim=(0,cfg.horizon)); ax.grid(axis="y",color=".9",lw=.5)
    axes[0].set_ylabel("Heat-removal rate $q_c$ (K min$^{-1}$)"); fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out.with_suffix(".svg"),bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(out.with_suffix(".png"),dpi=300,bbox_inches="tight"); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--grid-size",type=int,default=12); p.add_argument("--test-samples",type=int,default=120); p.add_argument("--validation-samples",type=int,default=60); p.add_argument("--epochs",type=int,default=300); p.add_argument("--kkt-epochs",type=int,default=30); p.add_argument("--value-weight",type=float,default=0.1); p.add_argument("--label-source",type=Path,default=None); p.add_argument("--output",type=Path,default=ROOT/"kkt_collocation"/"results"/"cstr_full_simulation"); p.add_argument("--resume",action="store_true"); p.add_argument("--write-manuscript-figures",action="store_true")
    # These options make a safety-limited operating domain explicit without
    # changing the mechanistic balance equations or the optimal-control setup.
    p.add_argument("--temperature-max", type=float, default=CSTRConfig.temperature_max)
    p.add_argument("--temperature-initial-min", type=float, default=CSTRConfig.temperature_range[0])
    p.add_argument("--temperature-initial-max", type=float, default=CSTRConfig.temperature_range[1])
    p.add_argument("--ca-initial-min", type=float, default=CSTRConfig.ca_range[0])
    p.add_argument("--ca-initial-max", type=float, default=CSTRConfig.ca_range[1])
    args=p.parse_args()
    if args.temperature_initial_max > args.temperature_max:
        raise ValueError("The declared initial-temperature domain must be initially safe.")
    cfg=CSTRConfig(
        temperature_max=args.temperature_max,
        temperature_range=(args.temperature_initial_min, args.temperature_initial_max),
        ca_range=(args.ca_initial_min, args.ca_initial_max),
    )
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    labels_path=out/"cstr_labels.pkl"; states=grid_states(cfg,args.grid_size); records=[]
    source_path=args.label_source if args.label_source is not None else labels_path
    if (args.resume or args.label_source is not None) and source_path.exists():
        with source_path.open("rb") as f: data=pickle.load(f); records=[{k:data[k][i] for k in ("initial_state","optimal_controls","objective","path_duals","hds_max_g","solve_seconds")} for i in range(len(data["initial_state"]))]
    problem=CSTRTranscription(cfg); previous=None
    for i,x in enumerate(states[len(records):],start=len(records)):
        rec=problem.solve(x,previous); records.append(rec); previous=rec["optimal_controls"]
        payload={k:np.asarray([r[k] for r in records]) for k in records[0]}; payload["config"]=asdict(cfg)
        with labels_path.open("wb") as f: pickle.dump(payload,f)
        print(f"label {i+1}/{len(states)} J={rec['objective']:.5f} peak={rec['hds_max_g']:.2e} sec={rec['solve_seconds']:.2f}")
    data={k:np.asarray([r[k] for r in records]) for k in records[0]}
    supervised,mean,std,_=train_branch(data,cfg,kkt=False,epochs=args.epochs,seed=cfg.seed,value_weight=args.value_weight)
    refined,_,_,_=train_branch(data,cfg,kkt=True,epochs=args.kkt_epochs,seed=cfg.seed+1,base=copy.deepcopy(supervised),value_weight=args.value_weight)
    torch.save({"model":supervised.state_dict(),"mean":mean,"std":std,"config":asdict(cfg)},out/"cstr_supervised.pth"); torch.save({"model":refined.state_dict(),"mean":mean,"std":std,"config":asdict(cfg)},out/"cstr_kkt_refined.pth")
    val=lhs_states(cfg,args.validation_samples,cfg.seed+1); uval,_=predict(supervised,mean,std,val); cor=make_corrector(cfg)
    gate=audit_raw_hds_peaks(np.array([cor.audit(x,u,cfg.zoh_duration) for x,u in zip(val,uval)]),AdaptiveKKTThresholds(allowed_violation_rate=.05,rate_normalized_violation=.025,allowed_normalized_peak_violation=.03,engineering_constraint_scale=5.,numerical_violation_tolerance=1e-8))
    selected="KKT-refined" if gate.kkt_refinement_required else "Supervised"; model=refined if gate.kkt_refinement_required else supervised
    test=lhs_states(cfg,args.test_samples,cfg.seed+2); u,inference=predict(model,mean,std,test); rows=evaluate("Adaptive ("+selected+") + HDS",test,u,cfg,inference,True)
    with (out/"per_sample.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    summary={"model":"representative literature-informed mechanistic non-isothermal CSTR; not plant data","config":asdict(cfg),"labels":len(data["initial_state"]),"value_weight":args.value_weight,"gate":asdict(gate),"selected_branch":selected,"test":summarise(rows)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plot_results(rows,test,u,cfg,out/"cstr_temperature_population"); plot_controls(rows,test,u,cfg,out/"cstr_controls_population")
    manuscript_figures=ROOT/"论文写作"/"figures"
    if args.write_manuscript_figures and manuscript_figures.exists():
        plot_results(rows,test,u,cfg,manuscript_figures/"cstr_temperature_population")
        plot_controls(rows,test,u,cfg,manuscript_figures/"cstr_controls_population")
    print(json.dumps(summary,indent=2)); print("saved",out)

if __name__=="__main__": main()
