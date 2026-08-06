"""Generate HDS-audited penicillin labels and path multipliers for KKT learning.

The problem definition is fixed independently of the legacy scripts:
``x2(0) in [0.1,0.3]``, 10 four-hour ZOH feed rates in ``[0,2]``,
``x2(t)<=0.5``, and ``J=-x3(tf)``.  Direct RK4 transcription supplies
discretized path multipliers; an event-located continuous-time HDS audit is
the final label acceptance criterion.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import BFGS, NonlinearConstraint, minimize, nnls, root

ROOT = Path(__file__).resolve().parents[1]
_WORKER: "PenicillinTranscription | None" = None


def import_casadi():
    package_root = ROOT / "third_party" / "casadi"
    dll_dir = package_root / "casadi"
    if package_root.exists() and str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    if dll_dir.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
    import casadi as ca
    return ca


@dataclass(frozen=True)
class PenicillinConfig:
    horizon: float = 40.0
    zoh_steps: int = 10
    substeps_per_zoh: int = 5
    x2_limit: float = 0.5
    u_min: float = 0.0
    u_max: float = 2.0
    x2_range: tuple[float, float] = (0.1, 0.3)
    # A 1e-2 interior margin is required by a pilot HDS audit: this fed-batch
    # model can form a sizeable x2 peak between transcription nodes.
    safety_margin: float = 1e-3
    solver_maxiter: int = 1500
    gtol: float = 1e-6
    hds_tolerance: float = 1e-7
    active_tolerance: float = 1e-7

    @property
    def total_substeps(self): return self.zoh_steps * self.substeps_per_zoh
    @property
    def node_count(self): return self.total_substeps + 1
    @property
    def dt(self): return self.horizon / self.total_substeps
    @property
    def segment_duration(self): return self.horizon / self.zoh_steps


class PenicillinTranscription:
    """Direct multiple shooting with exact CasADi derivatives and SciPy NLP."""
    def __init__(self, config: PenicillinConfig, seed_controls: np.ndarray | None = None):
        self.config, self.ca, self.seed_controls = config, import_casadi(), seed_controls
        self.state_dim = 4
        self.control_offset = self.state_dim * config.node_count
        self.decision_size = self.control_offset + config.zoh_steps
        self.eq_count = self.state_dim * (config.total_substeps + 1)
        self._build()

    @staticmethod
    def _f_numpy(x: np.ndarray, u: float) -> np.ndarray:
        x1, x2, x3, x4 = np.maximum(np.asarray(x[:3], dtype=float), 0.0).tolist() + [max(float(x[3]), 1e-10)]
        kl, mu, yxs, theta, yp, ki, mx, kxp, kp, s0 = .006, .11, .47, .004, 1.2, .1, .029, .01, .0001, 400.
        d1, d2 = kl*x1+x2, x2+kp+x2*x2/ki
        return np.array([mu*x1*x2/d1-u*x1/x4,
            -mu*x1*x2/(yxs*d1)-theta*x1*x2/(yp*d2)-mx*x1+u*(s0-x2)/x4,
            theta*x1*x2/d2-kxp*x3-u*x3/x4, u])

    def _f(self, x, u):
        ca = self.ca
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        kl, mu, yxs, theta, yp, ki, mx, kxp, kp, s0 = .006, .11, .47, .004, 1.2, .1, .029, .01, .0001, 400.
        d1, d2 = kl*x1+x2, x2+kp+x2*x2/ki
        return ca.vertcat(mu*x1*x2/d1-u*x1/x4,
            -mu*x1*x2/(yxs*d1)-theta*x1*x2/(yp*d2)-mx*x1+u*(s0-x2)/x4,
            theta*x1*x2/d2-kxp*x3-u*x3/x4, u)

    def _rk4(self, x, u):
        h = self.config.dt
        k1 = self._f(x,u); k2 = self._f(x+.5*h*k1,u); k3 = self._f(x+.5*h*k2,u); k4 = self._f(x+h*k3,u)
        return x + h*(k1+2*k2+2*k3+k4)/6

    def _build(self):
        ca, c = self.ca, self.config
        z, p = ca.SX.sym("z", self.decision_size), ca.SX.sym("p", 1)
        X = ca.reshape(z[:self.control_offset], self.state_dim, c.node_count)
        U = z[self.control_offset:]
        initial = ca.vertcat(1.0, p[0], .001, 250.0)
        equations = [X[:,0]-initial]
        for j in range(c.total_substeps):
            equations.append(X[:,j+1]-self._rk4(X[:,j], U[j//c.substeps_per_zoh]))
        # g=x2-limit <=0, with a small transcription interior margin.
        paths = [X[1,j]-(c.x2_limit-c.safety_margin) for j in range(c.node_count)]
        constraints = ca.vertcat(*(equations+paths))
        objective = -X[2,-1]
        self.obj = ca.Function("pen_obj", [z,p], [objective])
        self.obj_grad = ca.Function("pen_obj_grad", [z,p], [ca.gradient(objective,z)])
        self.cons = ca.Function("pen_cons", [z,p], [constraints])
        self.cons_jac = ca.Function("pen_cons_jac", [z,p], [ca.jacobian(constraints,z)])
        self.lower = np.concatenate((np.zeros(self.eq_count), np.full(c.node_count, -np.inf)))
        self.upper = np.zeros(self.eq_count+c.node_count)
        self.bounds_lo, self.bounds_hi = np.full(self.decision_size,-np.inf), np.full(self.decision_size,np.inf)
        self.bounds_lo[self.control_offset:], self.bounds_hi[self.control_offset:] = c.u_min, c.u_max

    def _nodes(self, x2: float, u: np.ndarray):
        nodes = np.empty((self.config.node_count,4)); nodes[0] = [1.,x2,.001,250.]
        x = nodes[0].copy(); h=self.config.dt
        for j in range(self.config.total_substeps):
            control=float(u[j//self.config.substeps_per_zoh]); f=self._f_numpy
            k1=f(x,control); k2=f(x+.5*h*k1,control); k3=f(x+.5*h*k2,control); k4=f(x+h*k3,control)
            x=x+h*(k1+2*k2+2*k3+k4)/6; nodes[j+1]=x
        return nodes

    def initial_guess(self, x2: float):
        u = np.full(self.config.zoh_steps, .8) if self.seed_controls is None else self.seed_controls[np.argmin(abs(self.seed_controls[:,0]*0 + np.linspace(*self.config.x2_range, len(self.seed_controls))-x2))]
        return np.concatenate((self._nodes(x2,u).T.reshape(-1,order="F"),u))

    def solve(self, x2: float):
        p=np.array([x2]); start=time.perf_counter()
        result=minimize(lambda z:float(self.obj(z,p)), self.initial_guess(x2), method="trust-constr",
            jac=lambda z:np.asarray(self.obj_grad(z,p),dtype=float).ravel(), hess=BFGS(),
            bounds=list(zip(self.bounds_lo,self.bounds_hi)),
            constraints=[NonlinearConstraint(lambda z:np.asarray(self.cons(z,p),dtype=float).ravel(),self.lower,self.upper,jac=lambda z:np.asarray(self.cons_jac(z,p),dtype=float))],
            options={"maxiter":self.config.solver_maxiter,"gtol":self.config.gtol,"xtol":1e-10,"barrier_tol":1e-10,"verbose":0})
        if not result.success:
            raise RuntimeError(f"{result.message}; optimality={getattr(result, 'optimality', np.nan):.2e}; constraint={getattr(result, 'constr_violation', np.nan):.2e}")
        z=np.asarray(result.x); controls=z[self.control_offset:].copy(); peak,time_peak=hds_peak(x2,controls,self.config)
        dual=np.maximum(np.asarray(result.v[0],dtype=float).ravel()[self.eq_count:],0.0)
        return {"initial_state":np.array([1.,x2,.001,250.]),"optimal_controls":controls,"objective":float(result.fun),"path_duals":dual,"hds_max_g":peak,"hds_peak_time":time_peak,"solve_seconds":time.perf_counter()-start}


def hds_peak(x2: float, controls: np.ndarray, config: PenicillinConfig):
    state=np.array([1.,x2,.001,250.]); greatest=state[1]-config.x2_limit; when=0.; offset=0.
    for u in controls:
        def ode(_t,x): return PenicillinTranscription._f_numpy(x,float(u))
        def stationary(_t,x): return ode(_t,x)[1]
        stationary.direction=0; stationary.terminal=False
        sol=solve_ivp(ode,(0.,config.segment_duration),state,method="DOP853",rtol=1e-10,atol=1e-12,max_step=config.segment_duration/100,dense_output=True,events=stationary)
        ts=np.r_[0.,config.segment_duration,sol.t_events[0]]; values=sol.sol(ts)[1]-config.x2_limit; i=int(values.argmax())
        if values[i]>greatest: greatest=float(values[i]); when=offset+float(ts[i])
        state=sol.y[:,-1]; offset+=config.segment_duration
    return float(greatest),float(when)


class ReducedPenicillinProblem:
    """Control-only transcription used for the penicillin KKT experiment.

    Eliminating the states avoids a poorly conditioned 800-variable multiple
    shooting problem while retaining a direct, differentiable RK4 map from
    the 10 ZOH controls to all path nodes.  Its multipliers therefore match
    the reduced-space KKT residual used during policy training.
    """
    def __init__(self, config: PenicillinConfig, seed_x2: np.ndarray, seed_controls: np.ndarray,
                 use_warm_start: bool = True):
        self.config, self.ca = config, import_casadi()
        self.seed_x2, self.seed_controls = np.asarray(seed_x2), np.asarray(seed_controls)
        self.use_warm_start = use_warm_start
        self._build()

    def _f(self, x, u):
        ca=self.ca; x1,x2,x3,x4=x[0],x[1],x[2],x[3]
        kl,mu,yxs,theta,yp,ki,mx,kxp,kp,s0=.006,.11,.47,.004,1.2,.1,.029,.01,.0001,400.
        d1,d2=kl*x1+x2,x2+kp+x2*x2/ki
        return ca.vertcat(mu*x1*x2/d1-u*x1/x4,
            -mu*x1*x2/(yxs*d1)-theta*x1*x2/(yp*d2)-mx*x1+u*(s0-x2)/x4,
            theta*x1*x2/d2-kxp*x3-u*x3/x4,u)

    def _rk4(self,x,u):
        h=self.config.dt; k1=self._f(x,u); k2=self._f(x+.5*h*k1,u); k3=self._f(x+.5*h*k2,u); k4=self._f(x+h*k3,u)
        return x+h*(k1+2*k2+2*k3+k4)/6

    def _build(self):
        ca,c=self.ca,self.config; u=ca.SX.sym("u",c.zoh_steps); p=ca.SX.sym("p",1)
        x=ca.vertcat(1.,p[0],.001,250.); paths=[x[1]-(c.x2_limit-c.safety_margin)]
        for j in range(c.total_substeps):
            x=self._rk4(x,u[j//c.substeps_per_zoh]); paths.append(x[1]-(c.x2_limit-c.safety_margin))
        g=ca.vertcat(*paths); obj=-x[2]; multiplier=ca.SX.sym("multiplier",g.numel())
        obj_hessian,_=ca.hessian(obj,u)
        lagrangian_hessian,_=ca.hessian(ca.dot(multiplier,g),u)
        self.obj=ca.Function("pen_reduced_obj",[u,p],[obj]); self.obj_grad=ca.Function("pen_reduced_grad",[u,p],[ca.gradient(obj,u)])
        self.g=ca.Function("pen_reduced_g",[u,p],[g]); self.g_jac=ca.Function("pen_reduced_gjac",[u,p],[ca.jacobian(g,u)])
        self.obj_hess=ca.Function("pen_reduced_obj_hess",[u,p],[obj_hessian])
        self.g_lagrangian_hess=ca.Function("pen_reduced_g_lagrangian_hess",[u,p,multiplier],[lagrangian_hessian])

    def solve(self,x2:float, dual_mode: str = "kkt-root"):
        p=np.array([x2]); start=time.perf_counter()
        # Cold-start timing uses the fixed neutral input, independent of all
        # offline labels.  The default remains available for label generation.
        guess = (self.seed_controls[np.argmin(abs(self.seed_x2-x2))]
                 if self.use_warm_start else np.full(self.config.zoh_steps, 0.8))
        # First obtain a feasible reduced control sequence with SLSQP, then
        # solve the active-set KKT equations to recover path duals.
        feasibility=minimize(lambda u:float(self.obj(u,p)),guess,method="SLSQP",jac=lambda u:np.asarray(self.obj_grad(u,p),dtype=float).ravel(),
            bounds=[(self.config.u_min,self.config.u_max)]*self.config.zoh_steps,
            constraints=[{"type":"ineq","fun":lambda u:-np.asarray(self.g(u,p),dtype=float).ravel(),"jac":lambda u:-np.asarray(self.g_jac(u,p),dtype=float)}],
            options={"maxiter":1000,"ftol":1e-10,"disp":False})
        if not feasibility.success:
            raise RuntimeError(f"SLSQP failed: {feasibility.message}")
        controls=np.asarray(feasibility.x)
        peak,tpeak=hds_peak(x2,controls,self.config)
        if peak>self.config.hds_tolerance:
            raise RuntimeError(f"SLSQP node-feasible but HDS-infeasible: {peak:.3e}")
        if dual_mode == "nnls":
            # Diagnostic-only fallback: not a solver dual.  This is retained
            # solely to reproduce the previous exploratory surrogate results.
            path=np.asarray(self.g(controls,p),dtype=float).ravel(); jac=np.asarray(self.g_jac(controls,p),dtype=float)
            grad=np.asarray(self.obj_grad(controls,p),dtype=float).ravel(); active=path>=-2e-3; dual=np.zeros_like(path)
            if active.any(): dual[active]=nnls(jac[active].T,-grad,maxiter=10000)[0]
            return {"initial_state":np.array([1.,x2,.001,250.]),"optimal_controls":controls,"objective":float(self.obj(controls,p)),"path_duals":dual,"hds_max_g":peak,"hds_peak_time":tpeak,"solve_seconds":time.perf_counter()-start,"dual_source":"nnls_reconstruction","solver_success":bool(feasibility.success),"optimality":float(np.linalg.norm(grad+jac.T@dual))}
        if dual_mode != "kkt-root":
            raise ValueError(f"Unsupported dual mode: {dual_mode}")

        # The primal SLSQP solution identifies the active node set.  We then
        # solve its KKT equations directly with exact CasADi derivatives.
        # This avoids the ill-conditioned 801-inequality trust-constr polish
        # while recovering a numerical dual of the same discretized NLP.
        path_before=np.asarray(self.g(controls,p),dtype=float).ravel()
        active=np.flatnonzero(path_before >= -self.config.active_tolerance)
        if not len(active):
            raise RuntimeError("SLSQP reported no active path node; cannot recover a path multiplier")
        # Remove numerically redundant active nodes with negative multipliers
        # and re-polish.  This is the usual active-set dual-feasibility step.
        for _ in range(8):
            jacobian=np.asarray(self.g_jac(controls,p),dtype=float)
            gradient=np.asarray(self.obj_grad(controls,p),dtype=float).ravel()
            initial_dual=np.linalg.lstsq(jacobian[active].T,-gradient,rcond=None)[0]
            def kkt_equations(z):
                u, dual=z[:self.config.zoh_steps],z[self.config.zoh_steps:]
                jac=np.asarray(self.g_jac(u,p),dtype=float)
                grad=np.asarray(self.obj_grad(u,p),dtype=float).ravel()
                return np.r_[grad+jac[active].T@dual,np.asarray(self.g(u,p),dtype=float).ravel()[active]]
            def kkt_jacobian(z):
                u, dual=z[:self.config.zoh_steps],z[self.config.zoh_steps:]
                jac=np.asarray(self.g_jac(u,p),dtype=float)
                full_dual=np.bincount(active,weights=dual,minlength=self.config.node_count)
                hessian=np.asarray(self.obj_hess(u,p),dtype=float)+np.asarray(self.g_lagrangian_hess(u,p,full_dual),dtype=float)
                return np.block([[hessian,jac[active].T],[jac[active],np.zeros((len(active),len(active)))]] )
            result=root(kkt_equations,np.r_[controls,initial_dual],jac=kkt_jacobian,method="lm",
                        options={"ftol":1e-12,"xtol":1e-12,"gtol":1e-12,"maxiter":10000})
            controls=np.asarray(result.x[:self.config.zoh_steps]); raw_active_dual=np.asarray(result.x[self.config.zoh_steps:])
            kkt_residual=kkt_equations(result.x)
            if raw_active_dual.min() >= -1e-7:
                break
            active=active[raw_active_dual >= -1e-7]
            if not len(active):
                raise RuntimeError("Active-set cleanup removed every path node")
        else:
            raise RuntimeError("Active-set cleanup did not reach dual feasibility")
        converged=(np.linalg.norm(kkt_residual)<=1e-7 and
                   controls.min()>=self.config.u_min-1e-7 and controls.max()<=self.config.u_max+1e-7)
        if not converged:
            raise RuntimeError(f"active-set KKT root failed: {result.message}; residual={np.linalg.norm(kkt_residual):.2e}")
        peak,tpeak=hds_peak(x2,controls,self.config)
        if peak>self.config.hds_tolerance:
            raise RuntimeError(f"KKT-polished node-feasible but HDS-infeasible: {peak:.3e}")
        path_after=np.asarray(self.g(controls,p),dtype=float).ravel()
        if path_after.max() > 1e-7 or raw_active_dual.min() < -1e-7:
            raise RuntimeError("Active-set KKT certificate failed at x2=%.12g: max_g=%.3e, min_dual=%.3e, active=%s" %
                               (x2,path_after.max(),raw_active_dual.min(),active.tolist()))
        dual=np.zeros_like(path_after); dual[active]=np.maximum(raw_active_dual,0.)
        stationarity=np.asarray(self.obj_grad(controls,p),dtype=float).ravel()+np.asarray(self.g_jac(controls,p),dtype=float).T@dual
        return {"initial_state":np.array([1.,x2,.001,250.]),"optimal_controls":controls,"objective":float(self.obj(controls,p)),"path_duals":dual,"raw_path_duals":dual.copy(),"hds_max_g":peak,"hds_peak_time":tpeak,"solve_seconds":time.perf_counter()-start,"dual_source":"active_set_kkt_root","solver_success":bool(result.success),"kkt_certificate_accepted":True,"optimality":float(np.linalg.norm(kkt_residual)),"constr_violation":float(max(0.,path_after.max())),"active_count":int(len(active)),"bound_duals":np.zeros(self.config.zoh_steps),"stationarity_norm":float(np.linalg.norm(stationarity))}

def _worker_init(config, seed_x2, seeds, dual_mode, cold_start):
    global _WORKER, _DUAL_MODE; _WORKER=ReducedPenicillinProblem(config,seed_x2,seeds, use_warm_start=not cold_start); _DUAL_MODE=dual_mode
def _worker_solve(x2):
    if _WORKER is None: raise RuntimeError("worker not initialized")
    return _WORKER.solve(float(x2), _DUAL_MODE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points",type=int,default=100); parser.add_argument("--substeps-per-zoh",type=int,default=5)
    parser.add_argument("--states-npy",type=Path); parser.add_argument("--indices-npy",type=Path,help="Optional indices selecting a shared subset of --states-npy."); parser.add_argument("--max-points",type=int); parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--dual-mode",choices=("kkt-root","nnls"),default="kkt-root")
    parser.add_argument("--cold-start",action="store_true",help="Use a fixed neutral control sequence rather than a nearest-label warm start.")
    parser.add_argument("--seed-lookup",type=Path,default=ROOT/"青霉素绘图"/"fermentation_lookup_table_x2_01_03_100samples_fmincon_init_no_early_stop.pkl")
    parser.add_argument("--output",type=Path,default=ROOT/"kkt_collocation"/"data"/"penicillin_kkt.pkl"); args=parser.parse_args()
    config=PenicillinConfig(substeps_per_zoh=args.substeps_per_zoh)
    states=np.linspace(*config.x2_range,args.points) if args.states_npy is None else np.asarray(np.load(args.states_npy),dtype=float).reshape(-1)
    if args.indices_npy is not None:
        indices=np.asarray(np.load(args.indices_npy),dtype=int).reshape(-1)
        states=states[indices]
    if args.max_points: states=states[:args.max_points]
    with args.seed_lookup.open("rb") as handle:
        lookup=pickle.load(handle); seeds=np.asarray(lookup["optimal_us"],dtype=float); seed_x2=np.asarray(lookup["x2_samples"],dtype=float)
    problem=ReducedPenicillinProblem(config,seed_x2,seeds, use_warm_start=not args.cold_start)
    iterator=(map(lambda x2: problem.solve(x2,args.dual_mode),states) if args.workers==1 else
              ProcessPoolExecutor(max_workers=args.workers,initializer=_worker_init,
                                  initargs=(config,seed_x2,seeds,args.dual_mode,args.cold_start)).map(_worker_solve,states))
    records=[]
    for i,record in enumerate(iterator,1):
        if record["hds_max_g"]>config.hds_tolerance: raise RuntimeError(f"HDS audit failed at {i}: {record['hds_max_g']:.3e}")
        records.append(record); print(f"{i}/{len(states)} J={record['objective']:.6f} g={record['hds_max_g']:.2e} dual={record['path_duals'].max():.2e} t={record['solve_seconds']:.2f}s")
    keys=["initial_state","optimal_controls","objective","path_duals","hds_max_g","hds_peak_time","solve_seconds","dual_source","solver_success","kkt_certificate_accepted","optimality","constr_violation","active_count","stationarity_norm"]
    data={key:np.asarray([r.get(key,np.nan) for r in records]) for key in keys}
    if args.dual_mode == "kkt-root":
        data["raw_path_duals"]=np.asarray([r["raw_path_duals"] for r in records])
        data["bound_duals"]=np.asarray([r["bound_duals"] for r in records])
    data["config"]=asdict(config)
    data["initialization"]="fixed neutral control u=0.8 (cold start)" if args.cold_start else "nearest offline reference control (warm start)"
    data["description"]=("Penicillin reduced direct-RK4 NLP labels with active-set KKT-root path multipliers and independent HDS audit. "
                         "The multipliers satisfy the discretized NLP KKT equations to the stored numerical tolerance, not the continuous-time problem." if args.dual_mode == "kkt-root" else
                         "Penicillin reduced direct-RK4 labels with NNLS-reconstructed multiplier surrogates; not solver dual variables.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("wb") as handle: pickle.dump(data,handle)
    print(f"Saved {len(records)} labels to {args.output}")

if __name__=="__main__": main()
