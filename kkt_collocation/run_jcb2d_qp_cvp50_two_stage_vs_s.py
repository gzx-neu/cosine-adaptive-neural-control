"""Independent CVP50 JCB S-versus-S+K experiment using reduced-QP labels.

The reference solver is the same cold-start reduced-space CVP50 QP used for
the labels.  Its multipliers are finite-dimensional transcription quantities,
not continuous-time multipliers.
"""
from __future__ import annotations

import argparse, copy, csv, json, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from generate_jcb_reduced_kkt_data import JCBConfig, ReducedJCBQP
from run_jcb2d_jiang_valc import g, gdot, initial, lhs, ode
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    seed: int = 20260811
    split_seed: int = 20260771
    supervised_epochs: int = 200
    continuation_epochs: int = 20
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    value_weight: float = .1
    kkt_weight: float = 1e-3
    augmented_penalty: float = 10.0
    anchor_weight: float = .1
    validation_count: int = 60
    test_count: int = 100
    lambda_grid: int = 31
    gradient_clip_norm: float = 1.0

    @property
    def total_epochs(self): return self.supervised_epochs + self.continuation_epochs


class Policy(nn.Module):
    def __init__(self, n: int, low: float, high: float):
        super().__init__()
        self.low, self.high = low, high
        self.body = nn.Sequential(nn.Linear(2, 96), nn.Tanh(), nn.Linear(96, 192), nn.Tanh(), nn.Linear(192, 96), nn.Tanh())
        self.value, self.control = nn.Linear(96, 1), nn.Linear(96, n)
    def forward(self, p):
        z = self.body(p)
        return self.value(z), self.low + (self.high-self.low) * torch.sigmoid(self.control(z))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_labels(path):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows.sort(key=lambda x: int(x["index"]))
    if len(rows) != 900 or not all(x.get("success") for x in rows): raise ValueError("Expected 900 successful CVP50 labels")
    p = np.asarray([x["initial_state_parameter"] for x in rows], float)
    u = np.asarray([x["controls"] for x in rows], float)
    j = np.asarray([x["objective"] for x in rows], float)
    mu = np.asarray([x["path_duals"] for x in rows], float)
    bd = np.asarray([x["bound_duals"] for x in rows], float)
    r = np.asarray([x["kkt_stationarity_norm"] for x in rows], float)
    expected_nodes = u.shape[1] * (int(json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))["config"]["substeps_per_zoh"]) + 1)
    if u.shape[0] != 900 or mu.shape != (900, expected_nodes) or bd.shape != (900, 2, u.shape[1]):
        raise ValueError(f"Unexpected CVP50 label shapes: {u.shape}, {mu.shape}, {bd.shape}")
    return p, u, j, mu, bd, r


def rollout(p, u, cfg):
    """Exact ZOH flow and the exact same 5-point objective quadrature as the QP."""
    x1, x2 = p[:, 0], p[:, 1]
    h = cfg.zoh_duration; cost = torch.zeros_like(x1); gs = []
    xi, wi = np.polynomial.legendre.leggauss(5)
    q = torch.tensor(.5*h*(xi+1), dtype=p.dtype, device=p.device)[None]
    w = torch.tensor(wi, dtype=p.dtype, device=p.device)[None]
    for k in range(cfg.zoh_steps):
        uk = u[:, k]
        # Match the generator: retain every segment left endpoint and every
        # equally spaced subnode through its right endpoint.
        tau = torch.linspace(0.0, h, cfg.substeps_per_zoh + 1, dtype=p.dtype, device=p.device)[None]
        local_x2 = uk[:, None] + (x2[:, None]-uk[:, None]) * torch.exp(-tau)
        local_t = k*h + tau
        gs.append(local_x2 - 8*(local_t-.5).square() + .5 + cfg.node_margin)
        e = torch.exp(-q); qx2 = uk[:, None] + (x2[:, None]-uk[:, None])*e
        qx1 = x1[:, None] + uk[:, None]*q + (x2[:, None]-uk[:, None])*(1-e)
        cost = cost + .5*h*(w*(qx1.square()+qx2.square()+.005*uk[:, None].square())).sum(1)
        eh = torch.exp(torch.as_tensor(-h, dtype=p.dtype, device=p.device))
        x1, x2 = x1 + uk*h + (x2-uk)*(1-eh), uk + (x2-uk)*eh
    return cost, torch.cat(gs, 1)


def full_kkt(p, u, mu, bound, cfg, rho, *, include_bounds=True):
    j, path = rollout(p, u, cfg); low, high = cfg.control_bounds
    lower, upper = low-u, u-high
    aug = j + (mu*path).sum(1)
    if include_bounds:
        aug = aug + (bound[:,0]*lower + bound[:,1]*upper).sum(1)
    aug = aug + .5*rho*torch.relu(path).square().sum(1)
    if include_bounds:
        aug = aug + .5*rho*(torch.relu(lower).square().sum(1) + torch.relu(upper).square().sum(1))
    grad = torch.autograd.grad(aug.sum(), u, create_graph=True, retain_graph=True)[0]
    stationarity = grad.square().mean()
    primal = torch.relu(path).square().mean()
    comp = (mu*path).square().mean()
    if include_bounds:
        primal = torch.cat((torch.relu(path), torch.relu(lower), torch.relu(upper)), 1).square().mean()
        comp = torch.cat((mu*path, bound[:,0]*lower, bound[:,1]*upper), 1).square().mean()
    return {"total": stationarity+primal+comp, "stationarity": stationarity, "primal": primal, "complementarity": comp}


def actual_objective(p, u, cfg):
    with torch.no_grad():
        j, _ = rollout(torch.tensor(p[None], dtype=torch.float64), torch.tensor(u[None], dtype=torch.float64), cfg)
    return float(j.item())


def cold_references(points, qp, corrector, cfg, output):
    rows=[]
    for i, point in enumerate(points):
        result=qp.solve(point); u=np.asarray(result["controls"], float)
        peak=float(corrector.audit(initial(point),u,cfg.zoh_duration))
        answer=corrector.correct(initial(point),u,cfg.zoh_duration) if peak > 1e-8 else None
        applied=np.asarray(answer.controls if answer is not None and answer.accepted else u, float)
        final_peak=float(corrector.audit(initial(point),applied,cfg.zoh_duration))
        rows.append({"index":i,"x1_0":float(point[0]),"x2_0":float(point[1]),"controls":u.tolist(),"objective":float(result["objective"]),"hds_controls":applied.tolist(),"hds_objective":actual_objective(point,applied,cfg),"solve_seconds":float(result["solve_seconds"]),"nominal_hds_max_g":peak,"hds_max_g":final_peak,"solver_success":True,"audit_accepted":bool(answer.accepted) if answer is not None else bool(peak<=1e-8),"cold_start":"fixed zero CVP50 control; no warm start; same HDS/lambda correction when the discrete-QP sequence fails the continuous audit"})
    dump(output/"cold_start_references.json", rows)
    return rows


def evaluate(name, model, mean, std, points, refs, cfg, exp, output):
    x=torch.tensor((points-mean)/std,dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10): model(x[:10])
        t=time.perf_counter(); _, out=model(x); infer=(time.perf_counter()-t)/len(points)
    corrector=HDSLambdaCorrector(ode,g,gdot,cfg.control_bounds,HDSLambdaConfig(grid_size=exp.lambda_grid,safety_margin=0.0,max_step_fraction=1.0))
    rows=[]
    for i,(p,u,ref) in enumerate(zip(points,out.numpy(),refs)):
        nominal_peak=float(corrector.audit(initial(p),u,cfg.zoh_duration)); nominal_j=actual_objective(p,u,cfg)
        t=time.perf_counter(); ans=corrector.correct(initial(p),u,cfg.zoh_duration); hds=time.perf_counter()-t
        accepted=bool(ans.accepted); applied=np.asarray(ans.controls if accepted else u,float); applied_j=actual_objective(p,applied,cfg) if accepted else np.nan
        ref_ok=bool(ref["solver_success"] and ref["audit_accepted"])
        reference_j=float(ref["hds_objective"])
        gap=100*(applied_j-reference_j)/max(abs(reference_j),1e-12) if accepted and ref_ok else np.nan
        rows.append({"method":name,"index":i,"x1_0":float(p[0]),"x2_0":float(p[1]),"accepted":accepted,"fallback":not accepted,"reference_audited":ref_ok,"nominal_hds_max_g":nominal_peak,"final_hds_max_g":float(corrector.audit(initial(p),applied,cfg.zoh_duration)) if accepted else np.nan,"nominal_objective":nominal_j,"hds_objective":applied_j,"reference_objective":reference_j,"hds_relative_gap_percent":gap,"corrected_segments":int(sum(s.corrected for s in ans.segments)),"inference_seconds":infer,"hds_seconds":hds,"total_predeployment_seconds":infer+hds})
    with (output/f"{name}_test_sample_log.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    a=lambda k:np.asarray([r[k] for r in rows],float); ok=np.isfinite(a("hds_relative_gap_percent")); accepted=a("accepted").astype(bool)
    return {"accepted_network_samples":int(accepted.sum()),"fallback_samples":int((~accepted).sum()),"accepted_reference_samples":int(ok.sum()),"nominal_violation_rate_percent":float(100*np.mean(a("nominal_hds_max_g")>1e-8)),"nominal_max_g":float(np.max(a("nominal_hds_max_g"))),"final_max_g":float(np.nanmax(a("final_hds_max_g"))),"mean_hds_relative_gap_percent":float(np.nanmean(a("hds_relative_gap_percent"))),"std_hds_relative_gap_percent":float(np.nanstd(a("hds_relative_gap_percent"),ddof=1)),"mean_corrected_segments":float(np.mean(a("corrected_segments"))),"mean_hds_objective_change":float(np.nanmean(a("hds_objective")-a("nominal_objective"))),"mean_inference_seconds":infer,"mean_hds_seconds":float(np.mean(a("hds_seconds"))),"mean_total_predeployment_seconds":float(np.mean(a("total_predeployment_seconds")))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--labels",type=Path,default=ROOT/"kkt_collocation/results/jcb2d_qp_cvp50_30x30/records.jsonl"); ap.add_argument("--output",type=Path,default=ROOT/"kkt_collocation/results/jcb2d_qp_cvp50_two_stage_vs_s_single"); ap.add_argument("--seed",type=int,default=20260811); ap.add_argument("--split-seed",type=int,default=20260771); ap.add_argument("--anchor-weight",type=float,default=.1); ap.add_argument("--path-only-kkt",action="store_true"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    exp=Experiment(seed=args.seed,split_seed=args.split_seed,anchor_weight=args.anchor_weight,supervised_epochs=2,continuation_epochs=1,validation_count=4,test_count=4) if args.smoke else Experiment(seed=args.seed,split_seed=args.split_seed,anchor_weight=args.anchor_weight)
    seed_all(exp.seed)
    meta=json.loads((args.labels.parent/"summary.json").read_text(encoding="utf-8")); raw=dict(meta["config"])
    for k in ("control_bounds","x1_initial_range","x2_initial_range"): raw[k]=tuple(raw[k])
    cfg=JCBConfig(**raw); p,uref,jref,mu,bound,recorded=load_labels(args.labels)
    validation=lhs(exp.validation_count, type("B",(),{"x1_bounds":cfg.x1_initial_range,"x2_bounds":cfg.x2_initial_range})(), exp.split_seed+1)
    test=lhs(exp.test_count, type("B",(),{"x1_bounds":cfg.x1_initial_range,"x2_bounds":cfg.x2_initial_range})(), exp.split_seed+2)
    np.save(args.output/"validation_initial_conditions.npy",validation); np.save(args.output/"test_initial_conditions.npy",test)
    # Teacher self-check is deliberately in float64 and checks the actual full reduced QP KKT form.
    pu=torch.tensor(p,dtype=torch.float64); uu=torch.tensor(uref,dtype=torch.float64,requires_grad=True); kk=full_kkt(pu,uu,torch.tensor(mu,dtype=torch.float64),torch.tensor(bound,dtype=torch.float64),cfg,exp.augmented_penalty,include_bounds=not args.path_only_kkt)
    teacher={"recorded_stationarity_norm_mean":float(recorded.mean()),"recorded_stationarity_norm_max":float(recorded.max()),"torch_total_kkt_residual":float(kk["total"].detach()),"interpretation":"finite-dimensional exact-flow CVP50 QP KKT quantities; not continuous-time multipliers"}; dump(args.output/"teacher_kkt_self_check.json",teacher)
    corrector=HDSLambdaCorrector(ode,g,gdot,cfg.control_bounds,HDSLambdaConfig(grid_size=exp.lambda_grid,safety_margin=0.0,max_step_fraction=1.0)); refs=cold_references(test,ReducedJCBQP(cfg),corrector,cfg,args.output)
    pt=torch.tensor(p,dtype=torch.float32); ut=torch.tensor(uref,dtype=torch.float32); jt=torch.tensor(jref[:,None],dtype=torch.float32); mt=torch.tensor(mu,dtype=torch.float32); bt=torch.tensor(bound,dtype=torch.float32); mean=p.mean(0); std=p.std(0).clip(1e-6); nx=torch.tensor((p-mean)/std,dtype=torch.float32); jm,js=jt.mean(),jt.std().clamp_min(1e-6); low,high=cfg.control_bounds
    def train(name):
        seed_all(exp.seed); model=Policy(cfg.zoh_steps,low,high); initial_weights=copy.deepcopy(model.state_dict()); model.load_state_dict(initial_weights); history=[]; failed=None; start=time.perf_counter(); stages=[("supervised",exp.total_epochs,False,None,exp.supervised_lr)] if name=="S" else [("supervised",exp.supervised_epochs,False,None,exp.supervised_lr)]
        for stage,epochs,use_kkt,anchor,lr in stages:
            opt=torch.optim.Adam(model.parameters(),lr=lr)
            for epoch in range(1,epochs+1):
                try:
                    pred,u=model(nx); control=nn.functional.mse_loss((u-low)/(high-low),(ut-low)/(high-low)); value=nn.functional.mse_loss(pred,(jt-jm)/js); loss=control+exp.value_weight*value
                    k=None
                    if use_kkt:
                        k=full_kkt(pt,u,mt,bt,cfg,exp.augmented_penalty,include_bounds=not args.path_only_kkt); loss=loss+exp.kkt_weight*k["total"]/k["total"].detach().clamp_min(1.)
                    if anchor is not None: loss=loss+exp.anchor_weight*nn.functional.mse_loss((u-low)/(high-low),(anchor-low)/(high-low))
                    if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),exp.gradient_clip_norm); opt.step()
                    if epoch==1 or epoch%20==0 or epoch==epochs: history.append({"stage":stage,"epoch":epoch,"loss":float(loss.detach()),"control_mse":float(control.detach()),"objective_mse":float(value.detach()),"kkt":float(k["total"].detach()) if k else None})
                except (RuntimeError,FloatingPointError) as exc: failed=f"{type(exc).__name__}: {exc}"; break
            if failed: break
            if name=="S+K" and stage=="supervised":
                with torch.no_grad(): anchor=model(nx)[1].detach()
                stages.append(("continuation",exp.continuation_epochs,True,anchor,exp.continuation_lr))
        _,fu=model(nx); final=full_kkt(pt,fu,mt,bt,cfg,exp.augmented_penalty,include_bounds=not args.path_only_kkt); record={"completed":failed is None,"failure":failed,"seconds":time.perf_counter()-start,"control_mse_normalized":float(nn.functional.mse_loss((fu-low)/(high-low),(ut-low)/(high-low)).detach()),"objective_mse_normalized":float(nn.functional.mse_loss(model(nx)[0],(jt-jm)/js).detach()),"kkt_residual":float(final["total"].detach()),"kkt_stationarity":float(final["stationarity"].detach())}; torch.save({"model":model.state_dict(),"state_mean":mean,"state_std":std,"training":record},args.output/f"{name}.pth"); dump(args.output/f"{name}_training_log.json",{"training":record,"history":history}); return model,record
    methods={}
    for name in ("S","S+K"):
        model,tr=train(name); methods[name]={"training":tr,"deployment":evaluate(name,model,mean,std,test,refs,cfg,exp,args.output)}
    cold=np.mean([x["solve_seconds"] for x in refs]);
    for m in methods.values(): m["deployment"]["mean_cold_qp_seconds"]=float(cold);m["deployment"]["speedup_vs_cold_qp"]=float(cold/m["deployment"]["mean_total_predeployment_seconds"])
    final={"status":"completed","config":asdict(exp),"kkt_structure":"path-only finite-dimensional transcription KKT residual" if args.path_only_kkt else "path and box-bound finite-dimensional transcription KKT residual","label_source":str(args.labels),"label_cold_start_note":meta["cold_start_protocol"],"teacher_kkt_self_check":teacher,"reference":{"method":"same reduced-space exact-flow CVP50 QP, fixed zero-control cold start","count":len(refs),"audited":int(sum(x["audit_accepted"] for x in refs)),"mean_seconds":float(cold)},"methods":methods,"hds_statement":"Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}; dump(args.output/"summary.json",final)
    rows=["| Method | HDS gap (%) | Nominal violation | Corrected segments | Accepted / fallback | KKT residual | Speedup |","|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in methods.items():
        d,t=m["deployment"],m["training"]; rows.append(f"| {name} | {d['mean_hds_relative_gap_percent']:.3f} | {d['nominal_violation_rate_percent']:.1f}% | {d['mean_corrected_segments']:.2f} | {d['accepted_network_samples']} / {d['fallback_samples']} | {t['kkt_residual']:.3e} | {d['speedup_vs_cold_qp']:.2f}x |")
    (args.output/"summary_table.md").write_text("\n".join(rows)+"\n",encoding="utf-8"); print(json.dumps(final,indent=2,default=str),flush=True)


if __name__=="__main__": main()
