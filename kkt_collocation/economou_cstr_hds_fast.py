"""Cached discrete-grid HDS audit for the Economou two-input CSTR.

The frozen 31-point lambda base grid is unchanged. DOP853 locates roots of
dC_A/dt and dT/dt directly, then the candidate closest to lambda=1 is tested
first and accepted immediately. Terminal states and peaks from accepted
segment propagations are reused, so the corrected sequence is not propagated
a redundant second time.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.run_economou_cstr_supervised_hds import segment_audit
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig,economou_ode

def _events(control,cfg):
 def dca(t,x): return economou_ode(t,x,control,cfg)[0]
 def dtemp(t,x): return economou_ode(t,x,control,cfg)[2]
 dca.direction=0;dca.terminal=False;dtemp.direction=0;dtemp.terminal=False
 return dca,dtemp
def segment_event_audit(state,control,cfg):
 dur=cfg.zoh_duration_s
 sol=solve_ivp(lambda t,x:economou_ode(t,x,control,cfg),(0,dur),state,method='DOP853',dense_output=True,events=_events(control,cfg),rtol=1e-10,atol=1e-12)
 if not sol.success:raise RuntimeError(sol.message)
 ts=np.unique(np.r_[0.,dur,*sol.t_events]);xs=sol.sol(ts);g=np.vstack((xs[0]-cfg.ca_max,xs[2]-cfg.temperature_max_K))
 return g.max(1),xs[:,-1],ts
def candidates(control,cfg,grid):
 low=np.array([cfg.ti_bounds_K[0],cfg.flow_bounds[0]]);span=np.array([cfg.ti_bounds_K[1]-cfg.ti_bounds_K[0],cfg.flow_bounds[1]-cfg.flow_bounds[0]]);n=np.clip((control-low)/span,0,1);pos=n>1e-10;lmax=float(np.min(1/n[pos])) if np.any(pos) else 1.;a=np.unique(np.r_[np.linspace(0,lmax,grid),1.]);a=a[(a>=0)&(a<=lmax+1e-12)];return low,span,n,sorted(a,key=lambda z:(abs(z-1),z))
def fast_correct(state0,controls,cfg,grid=31,threshold=-1e-8):
 state=np.asarray(state0,float).copy();out=np.asarray(controls,float).copy();raw=-np.inf;applied_peak=-np.inf;changed=[];lams=[]
 for k in range(cfg.zoh_steps):
  peak,next_state,_=segment_event_audit(state,out[k],cfg);raw=max(raw,float(peak.max()))
  if peak.max()<=threshold:state=next_state;applied_peak=max(applied_peak,float(peak.max()));lams.append(1.);continue
  low,span,n,order=candidates(out[k],cfg,grid);found=None
  for lam in order:
   if abs(float(lam)-1.)<=1e-14:continue
   cand=low+np.clip(lam*n,0,1)*span;p,ns,_=segment_event_audit(state,cand,cfg)
   if p.max()<=threshold:found=(lam,cand,ns,float(p.max()));break
  if found is None:return {'accepted':False,'controls':out,'raw_peak':raw,'final_peak':np.nan,'corrected_segments':len(changed),'lambdas':lams}
  lam,out[k],state,corrected_peak=found;applied_peak=max(applied_peak,corrected_peak);lams.append(float(lam));changed.append(k)
 return {'accepted':applied_peak<=threshold,'controls':out,'raw_peak':raw,'final_peak':applied_peak,'corrected_segments':len(changed),'lambdas':lams,'segment_cache_reused':True,'final_reaudit_performed':False}
def strict_trace(state0,controls,cfg,grid=31):
 # Original 41-point scan/minimize peak finder, but expose lambdas.
 state=np.asarray(state0,float).copy();out=np.asarray(controls,float).copy();raw=-np.inf;lams=[];changed=0
 for k in range(cfg.zoh_steps):
  p,next_state=segment_audit(state,out[k],cfg);raw=max(raw,float(p.max()))
  if p.max()<=1e-8:state=next_state;lams.append(1.);continue
  low,span,n,order=candidates(out[k],cfg,grid);safe=[]
  for lam in order:
   cand=low+np.clip(lam*n,0,1)*span;q,ns=segment_audit(state,cand,cfg)
   if q.max()<=1e-8:safe.append((lam,cand,ns));break
  if not safe:return {'accepted':False,'controls':out,'raw_peak':raw,'final_peak':np.nan,'corrected_segments':changed,'lambdas':lams}
  lam,out[k],state=safe[0];lams.append(float(lam));changed+=1
 final=np.asarray(state0,float).copy();peak=-np.inf
 for u in out:
  q,final=segment_audit(final,u,cfg);peak=max(peak,float(q.max()))
 return {'accepted':peak<=1e-8,'controls':out,'raw_peak':raw,'final_peak':peak,'corrected_segments':changed,'lambdas':lams}
