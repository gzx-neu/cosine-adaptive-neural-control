"""VALC experiment for Jiang--Fu Example 2 (JCB).

The original benchmark has x(0)=(0,-1).  For the learned family used here,
only x1(0)=p varies over [-0.1,0.1], while x2(0)=-1 is retained.  Time is
augmented as a state so the event-located audit evaluates the original
time-varying path constraint exactly as written in the source problem.
"""
from __future__ import annotations

import argparse, json, os, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector

@dataclass(frozen=True)
class Config:
    horizon: float = 1.0
    zoh_steps: int = 20                 # Jiang--Fu's original CVP setting
    u_min: float = -15.0
    u_max: float = 15.0
    train_points: int = 400
    validation_points: int = 60
    test_points: int = 400
    epochs: int = 500
    seed: int = 20260771
    margin: float = 1e-6
    lambda_grid: int = 101
    @property
    def dt(self): return self.horizon / self.zoh_steps

def ode(_t: float, z: np.ndarray, u: float) -> np.ndarray:
    # z=(x1,x2,absolute_time,cost); the third state makes the constraint autonomous.
    return np.array([z[1], -z[1] + u, 1.0, z[0]**2 + z[1]**2 + .005*u**2])

def g(z: np.ndarray) -> float:
    return float(z[1] - 8.0*(z[2] - .5)**2 + .5)

def gdot(z: np.ndarray, u: float) -> float:
    return float(-z[1] + u - 16.0*(z[2] - .5))

def initial(p: float) -> np.ndarray: return np.array([p, -1.0, 0.0, 0.0])

def rollout(p: float, controls: np.ndarray, cfg: Config, grid: np.ndarray | None = None):
    # Exact ZOH propagation for x1' = x2, x2' = -x2+u.  The running cost is
    # integrated by five-point Gauss--Legendre quadrature on each segment.
    z = initial(p); ts=[]; zs=[]; xi,wi=np.polynomial.legendre.leggauss(5)
    for k,u in enumerate(np.asarray(controls,float)):
        a,b=k*cfg.dt,(k+1)*cfg.dt; x10,x20,t0,J0=z
        local=np.linspace(a,b,11) if grid is None else grid[(grid>=a-1e-12)&(grid<=b+1e-12)]
        if k: local=local[1:]
        tau=local-a; e=np.exp(-tau); x2=u+(x20-u)*e; x1=x10+u*tau+(x20-u)*(1-e)
        ts.extend(local); zs.extend(np.column_stack((x1,x2,local,np.full_like(local,J0))))
        q=.5*cfg.dt*(xi+1); eq=np.exp(-q); qx2=u+(x20-u)*eq; qx1=x10+u*q+(x20-u)*(1-eq)
        J1=J0+.5*cfg.dt*np.sum(wi*(qx1*qx1+qx2*qx2+.005*u*u))
        e1=np.exp(-cfg.dt); z=np.array([x10+u*cfg.dt+(x20-u)*(1-e1),u+(x20-u)*e1,b,J1])
    return np.asarray(ts),np.asarray(zs),z

def solve_reference(p: float, cfg: Config, start: np.ndarray | None = None) -> tuple[np.ndarray,float]:
    nodes=np.linspace(0,cfg.horizon,201)
    def fun(u): return float(rollout(p,u,cfg)[2][3])
    def path(u): return -np.asarray([g(z) for z in rollout(p,u,cfg,nodes)[1]])
    result=minimize(fun, np.zeros(cfg.zoh_steps) if start is None else start, method='SLSQP',
        bounds=[(cfg.u_min,cfg.u_max)]*cfg.zoh_steps,
        constraints={'type':'ineq','fun':path}, options={'maxiter':1000,'ftol':1e-10,'disp':False})
    if not result.success: raise RuntimeError(f'JCB reference failed at p={p}: {result.message}')
    return np.asarray(result.x),float(result.fun)

class Policy(nn.Module):
    def __init__(self,n:int):
        super().__init__(); self.body=nn.Sequential(nn.Linear(1,64),nn.Tanh(),nn.Linear(64,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh()); self.u=nn.Linear(64,n); self.v=nn.Linear(64,1)
    def forward(self,p):
        h=self.body(p); return self.v(h),15.*torch.tanh(self.u(h))

def train(ps, us, js, cfg:Config):
    torch.manual_seed(cfg.seed); model=Policy(cfg.zoh_steps); p=torch.tensor(ps[:,None],dtype=torch.float32); u=torch.tensor(us,dtype=torch.float32); j=torch.tensor(js[:,None],dtype=torch.float32)
    pm,psd=p.mean(0),p.std(0).clamp_min(1e-6); jm,jsd=j.mean(0),j.std(0).clamp_min(1e-6); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for _ in range(cfg.epochs):
        pv,pu=model((p-pm)/psd); loss=nn.functional.mse_loss(pu,u)+.1*nn.functional.mse_loss(pv,(j-jm)/jsd); opt.zero_grad();loss.backward();opt.step()
    return model.eval(),float(pm),float(psd)

def predict(model,pm,psd,points):
    with torch.no_grad(): return model(torch.tensor(((points-pm)/psd)[:,None],dtype=torch.float32))[1].numpy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'kkt_collocation'/'results'/'jcb_valc'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--train-points',type=int); ap.add_argument('--validation-points',type=int); ap.add_argument('--test-points',type=int); ap.add_argument('--epochs',type=int); ap.add_argument('--lambda-grid',type=int); args=ap.parse_args(); cfg=Config(train_points=40,test_points=40,validation_points=12,epochs=80,lambda_grid=31) if args.smoke else Config()
    cfg=Config(train_points=cfg.train_points if args.train_points is None else args.train_points, validation_points=cfg.validation_points if args.validation_points is None else args.validation_points, test_points=cfg.test_points if args.test_points is None else args.test_points, epochs=cfg.epochs if args.epochs is None else args.epochs, lambda_grid=cfg.lambda_grid if args.lambda_grid is None else args.lambda_grid)
    args.output.mkdir(parents=True,exist_ok=True)
    train_p=np.linspace(-.1,.1,cfg.train_points); labels=[]; warm=None
    for i,p in enumerate(train_p):
        warm,j=solve_reference(float(p),cfg,warm); labels.append(warm.copy()); print(f'label {i+1}/{len(train_p)}')
    controls=np.asarray(labels); costs=np.asarray([rollout(float(p),u,cfg)[2][3] for p,u in zip(train_p,controls)])
    model,pm,psd=train(train_p,controls,costs,cfg); torch.save({'model':model.state_dict(),'mean':pm,'std':psd,'config':asdict(cfg)},args.output/'supervised.pth')
    rng=np.random.default_rng(cfg.seed+1); val=rng.uniform(-.1,.1,cfg.validation_points); test=rng.uniform(-.1,.1,cfg.test_points)
    cor=HDSLambdaCorrector(ode,g,gdot,(cfg.u_min,cfg.u_max),HDSLambdaConfig(grid_size=cfg.lambda_grid,safety_margin=cfg.margin,max_step_fraction=200.))
    def assess(points):
        u=predict(model,pm,psd,points); rows=[]
        for p,a in zip(points,u):
            z0=initial(float(p)); raw=cor.audit(z0,a,cfg.dt); out=cor.correct(z0,a,cfg.dt); applied=out.controls if out.accepted else None; J=np.nan if applied is None else rollout(float(p),applied,cfg)[2][3]
            rows.append({'p':float(p),'nominal_peak':raw,'accepted':out.accepted,'corrected_segments':sum(s.corrected for s in out.segments),'applied_cost':J})
        return rows
    vr,tr=assess(val),assess(test)
    summary={'source':'Jiang and Fu (2026), Example 2 / Li et al. (2011); VALC extension varies x1(0) only.', 'config':asdict(cfg),'initial_domain':'x1(0) in [-0.1,0.1], x2(0)=-1','validation':vr,'test':tr,
      'test_summary':{'acceptance_rate':float(np.mean([r['accepted'] for r in tr])),'nominal_violation_rate':float(np.mean([r['nominal_peak']>0 for r in tr])),'mean_corrected_segments':float(np.mean([r['corrected_segments'] for r in tr]))}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary['test_summary'],indent=2))
if __name__=='__main__': main()
