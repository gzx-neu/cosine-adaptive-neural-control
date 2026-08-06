"""Four-way penicillin ablation from the legacy 100-point lookup table.

This is deliberately labelled an exploratory KKT experiment: the legacy data
contains controls and terminal products but no solver multipliers.  We thus
reconstruct non-negative *reduced-space multiplier surrogates* from each
reference rollout, and never present them as the teacher solver's true duals.
"""
from __future__ import annotations
import argparse, copy, csv, json, os, pickle, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig,HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual

KL,MU,YXS,THETA,YP,KI,MX,KXP,KP,S0=.006,.11,.47,.004,1.2,.1,.029,.01,.0001,400.
T,N,DT=40.,10,4.; X2_LIMIT=.5; UMAX=2.

@dataclass(frozen=True)
class Config:
    epochs:int=500; learning_rate:float=1e-3; kkt_weight:float=.05; rollout_weight:float=0.
    penalty:float=10.; substeps:int=10; seed:int=20260715; validation_samples:int=100; test_samples:int=400; grid_size:int=31
    kkt_epochs:int=50; kkt_learning_rate:float=1e-5; anchor_weight:float=1.

class Policy(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.j=nn.Linear(128,1); self.u=nn.Linear(128,N)
    def forward(self,x):
        z=self.body(x); return self.j(z),UMAX*torch.sigmoid(self.u(z))

def ode(_t,x,u):
    x1,x2,x3,x4=x; x1,x2,x3=max(x1,0),max(x2,0),max(x3,0); x4=max(x4,1e-10); d1=KL*x1+x2; d2=x2+KP+x2*x2/KI
    return np.array([MU*x1*x2/d1-u*x1/x4,-MU*x1*x2/(YXS*d1)-THETA*x1*x2/(YP*d2)-MX*x1+u*(S0-x2)/x4,THETA*x1*x2/d2-KXP*x3-u*x3/x4,u])
def g(x): return float(x[1]-X2_LIMIT)
def gdot(x,u): return float(ode(0,x,u)[1])

def rollout(x2,u,substeps=10):
    b=u.shape[0]; x=torch.stack((torch.ones(b,device=u.device),x2,torch.full((b,),.001,device=u.device),torch.full((b,),250.,device=u.device)),1); gs=[x[:,1]-X2_LIMIT]; h=DT/substeps
    def f(z,v):
        x1,x2_,x3,x4=z.unbind(1); d1=KL*x1+x2_; d2=x2_+KP+x2_.square()/KI
        return torch.stack((MU*x1*x2_/d1-v*x1/x4,-MU*x1*x2_/(YXS*d1)-THETA*x1*x2_/(YP*d2)-MX*x1+v*(S0-x2_)/x4,THETA*x1*x2_/d2-KXP*x3-v*x3/x4,v),1)
    for k in range(N*substeps):
        v=u[:,k//substeps]; k1=f(x,v); k2=f(x+.5*h*k1,v); k3=f(x+.5*h*k2,v); k4=f(x+h*k3,v); x=x+h*(k1+2*k2+2*k3+k4)/6; gs.append(x[:,1]-X2_LIMIT)
    return -x[:,2],torch.stack(gs,1)

def lhs(n,seed):
    rng=np.random.default_rng(seed); return .1+(rng.permutation(n)+rng.random(n))/n*.2

def reconstruct_duals(x2,controls,device,substeps=10):
    """One-active-peak reduced-space multiplier surrogate per reference.

    This is the least-squares nonnegative multiplier for the largest sampled
    path value.  It avoids pretending that the legacy table carries a full
    continuous multiplier measure, and is computationally tractable on CPU.
    """
    u=torch.tensor(controls,dtype=torch.float32,device=device,requires_grad=True); xi=torch.tensor(x2,dtype=torch.float32,device=device)
    J,G=rollout(xi,u,substeps); gradj=torch.autograd.grad(J.sum(),u,retain_graph=True)[0]
    indices=torch.argmax(G,dim=1); peak=G.gather(1,indices[:,None]).sum(); gradg=torch.autograd.grad(peak,u)[0]
    scalar=torch.relu(-(gradj*gradg).sum(1)/(gradg.square().sum(1)+1e-12)).detach().cpu().numpy()
    output=np.zeros((len(x2),G.shape[1]),dtype=np.float32); output[np.arange(len(x2)),indices.detach().cpu().numpy()]=scalar
    return output

def train(x2,u_ref,j_ref,mu_ref,use_kkt,cfg,device,model=None,epochs=None,learning_rate=None,anchor_controls=None,
          path_penalty_weight=0.0,constraint_scale=1.0):
    torch.manual_seed(cfg.seed); model=Policy().to(device) if model is None else model.to(device); xmean,xstd=x2.mean(),x2.std()+1e-8; jmean,jstd=j_ref.mean(),j_ref.std()+1e-8
    x=torch.tensor(((x2-xmean)/xstd)[:,None],dtype=torch.float32,device=device); xi=torch.tensor(x2,dtype=torch.float32,device=device); ur=torch.tensor(u_ref,dtype=torch.float32,device=device); jr=torch.tensor(((j_ref-jmean)/jstd)[:,None],dtype=torch.float32,device=device); mr=torch.tensor(mu_ref,dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate if learning_rate is None else learning_rate); total_epochs=cfg.epochs if epochs is None else epochs
    anchor=None if anchor_controls is None else torch.tensor(anchor_controls,dtype=torch.float32,device=device)
    for epoch in range(1,total_epochs+1):
        jp,u=model(x); sup=nn.functional.mse_loss(u,ur)+.1*nn.functional.mse_loss(jp,jr); loss=sup
        # A pure-supervision pass has rollout_weight=0 and no KKT term.  Do
        # not unroll the high-fidelity dynamics in that case: it contributes
        # no gradient to the stated objective and is especially costly for
        # the 801-node true-KKT penicillin labels.
        if cfg.rollout_weight > 0 or use_kkt or path_penalty_weight > 0:
            J,G=rollout(xi,u,cfg.substeps)
            if cfg.rollout_weight > 0:
                con=nn.functional.mse_loss(jp*jstd+jmean,J[:,None]); loss=loss+cfg.rollout_weight*con
        if path_penalty_weight > 0:
            loss=loss+path_penalty_weight*torch.relu(G/constraint_scale).square().mean()
        if use_kkt:
            raw_kkt=augmented_lagrangian_kkt_residual(J,u,G,mr,cfg.penalty).total
            # The penicillin states have heterogeneous physical scales
            # (volume≈250 versus x2≈0.5).  Normalize only the magnitude,
            # retaining the KKT gradient direction without overwhelming MSE.
            loss=loss+cfg.kkt_weight*raw_kkt/raw_kkt.detach().clamp_min(1.)
        if anchor is not None: loss=loss+cfg.anchor_weight*nn.functional.mse_loss(u,anchor)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if epoch in (1,total_epochs) or epoch%100==0:
            name='S+KKT' if use_kkt else ('S+P' if path_penalty_weight>0 else 'S')
            print(f"{name} e={epoch} loss={loss.item():.3e}")
    return model.eval(),float(xmean),float(xstd)

def predict(model,mean,std,x2,device):
    x=torch.tensor(((x2-mean)/std)[:,None],dtype=torch.float32,device=device); start=time.perf_counter()
    with torch.no_grad(): _,u=model(x)
    return u.cpu().numpy(),(time.perf_counter()-start)/len(x2)

def terminal_product(x2,u,corrector):
    state=np.array([1.,x2,.001,250.])
    for v in u: _,state=corrector.segment_peak(state,float(v),DT)
    return float(state[2])

def evaluate(name,nominal,x2s,correct,corrector,infer,cfg):
    out=[]
    for i,(x2,u) in enumerate(zip(x2s,nominal)):
        raw=corrector.audit(np.array([1.,x2,.001,250.]),u,DT); product_raw=terminal_product(x2,u,corrector); start=time.perf_counter(); result=corrector.correct(np.array([1.,x2,.001,250.]),u,DT) if correct else None; ft=time.perf_counter()-start if correct else 0.
        accepted=True if result is None else result.accepted; applied=u if result is None or not accepted else result.controls
        peak=corrector.audit(np.array([1.,x2,.001,250.]),applied,DT) if accepted else np.nan; prod=terminal_product(x2,applied,corrector) if accepted else np.nan
        seg=0 if result is None else sum(s.corrected for s in result.segments); lamb=np.ones(N) if result is None else np.asarray([s.lambda_value for s in result.segments if s.lambda_value is not None])
        out.append({'method':name,'sample_index':i,'x2_0':x2,'raw_hds_max_g':raw,'applied_hds_max_g':peak,'accepted':accepted,'fallback':not accepted,'raw_product':product_raw,'applied_product':prod,'product_change':prod-product_raw if accepted else np.nan,'corrected_segments':seg,'mean_abs_lambda_minus_one':float(np.mean(abs(lamb-1))),'inference_seconds':infer,'filter_seconds':ft})
    return out

def summary(rows):
    ans={}
    for name in sorted(set(r['method'] for r in rows)):
        a=[r for r in rows if r['method']==name]; f=lambda k:np.asarray([r[k] for r in a],float); accepted=f('accepted').astype(bool)
        raw=f('raw_hds_max_g')
        ans[name]={'samples':len(a),'raw_violation_rate':float(np.mean(raw>1e-8)),
                   'raw_severe_violation_rate':float(np.mean(raw>0.025*X2_LIMIT)),
                   'mean_positive_raw_violation':float(np.maximum(raw,0.).mean()),
                   'max_raw_hds_g':float(raw.max()),
                   'accepted_rate':float(accepted.mean()),'fallback_rate':float(1-accepted.mean()),
                   'max_applied_hds_g':float(np.nanmax(f('applied_hds_max_g'))),
                   'mean_corrected_segments':float(f('corrected_segments').mean()),
                   'mean_abs_lambda_minus_one':float(np.nanmean(f('mean_abs_lambda_minus_one'))),
                   'mean_product_change':float(np.nanmean(f('product_change'))),
                   'mean_inference_seconds':float(f('inference_seconds').mean()),
                   'mean_filter_seconds':float(f('filter_seconds').mean())}
    return ans

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'青霉素绘图'/'fermentation_lookup_table_x2_01_03_100samples_fmincon_init_no_early_stop.pkl'); p.add_argument('--output',type=Path,default=ROOT/'kkt_collocation'/'results'/'penicillin_legacy100_ablation'); p.add_argument('--epochs',type=int,default=500); p.add_argument('--kkt-epochs',type=int,default=50); p.add_argument('--kkt-weight',type=float,default=.05); p.add_argument('--substeps',type=int,default=5); p.add_argument('--test-samples',type=int,default=400); p.add_argument('--validation-samples',type=int,default=100); p.add_argument('--smoke',action='store_true'); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args(); cfg=Config(epochs=20 if args.smoke else args.epochs,kkt_epochs=10 if args.smoke else args.kkt_epochs,kkt_weight=args.kkt_weight,substeps=args.substeps,test_samples=12 if args.smoke else args.test_samples,validation_samples=8 if args.smoke else args.validation_samples)
    with args.data.open('rb') as h: d=pickle.load(h)
    x2=np.asarray(d['x2_samples'],np.float32); u=np.asarray(d['optimal_us'],np.float32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); args.output.mkdir(parents=True,exist_ok=True)
    # Turn the high-quality but occasionally node-only legacy controls into
    # HDS-safe teacher labels before learning or multiplier reconstruction.
    teacher_corrector=HDSLambdaCorrector(ode,g,gdot,(0.,UMAX),HDSLambdaConfig(grid_size=101,max_step_fraction=100.))
    safe_u=[]; safe_j=[]
    for value,control in zip(x2,u):
        certificate=teacher_corrector.correct(np.array([1.,value,.001,250.]),control,DT)
        if not certificate.accepted: raise RuntimeError(f'No HDS-safe teacher label at x2={value:.6f}')
        safe_u.append(certificate.controls); safe_j.append(-terminal_product(value,certificate.controls,teacher_corrector))
    u=np.asarray(safe_u,np.float32); j=np.asarray(safe_j,np.float32); np.savez(args.output/'hds_safe_teacher_labels.npz',x2=x2,controls=u,objective=j)
    print('Reconstructing reduced-space multiplier surrogates from HDS-safe legacy labels.'); mu=reconstruct_duals(x2,u,device,cfg.substeps); np.save(args.output/'multiplier_surrogates.npy',mu)
    if args.prepare_only:
        print(json.dumps({'labels':len(x2),'nonzero_multiplier_fraction':float(np.mean(mu>1e-10)),'max_multiplier':float(mu.max())},ensure_ascii=False)); return
    s,ms,ss=train(x2,u,j,mu,False,cfg,device); xpre=torch.tensor(((x2-ms)/ss)[:,None],dtype=torch.float32,device=device)
    with torch.no_grad(): _,anchor=s(xpre)
    k,mk,sk=train(x2,u,j,mu,True,cfg,device,model=copy.deepcopy(s),epochs=cfg.kkt_epochs,learning_rate=cfg.kkt_learning_rate,anchor_controls=anchor.cpu().numpy()); val=lhs(cfg.validation_samples,cfg.seed+1); test=lhs(cfg.test_samples,cfg.seed+2); np.save(args.output/'validation_x2.npy',val); np.save(args.output/'test_x2.npy',test)
    corr=HDSLambdaCorrector(ode,g,gdot,(0.,UMAX),HDSLambdaConfig(grid_size=cfg.grid_size,max_step_fraction=100.))
    us,ts=predict(s,ms,ss,test,device); uk,tk=predict(k,mk,sk,test,device); rows=[]; rows+=evaluate('S',us,test,False,corr,ts,cfg); rows+=evaluate('S+KKT-surrogate',uk,test,False,corr,tk,cfg); rows+=evaluate('S+HDS-lambda',us,test,True,corr,ts,cfg); rows+=evaluate('Full: S+KKT-surrogate+HDS-lambda',uk,test,True,corr,tk,cfg)
    with (args.output/'per_sample.csv').open('w',newline='',encoding='utf8') as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report={'config':asdict(cfg),'label_source':str(args.data),'kkt_note':'Multipliers are NNLS-reconstructed reduced-space surrogates, not teacher-solver duals.','methods':summary(rows)}
    with (args.output/'summary.json').open('w',encoding='utf8') as h: json.dump(report,h,ensure_ascii=False,indent=2)
    torch.save({'S':s.state_dict(),'KKT_surrogate':k.state_dict(),'normalization':{'S':[ms,ss],'KKT':[mk,sk]},'config':asdict(cfg)},args.output/'models.pth'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
