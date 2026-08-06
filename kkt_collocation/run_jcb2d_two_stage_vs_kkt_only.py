"""Fair JCB-2D S / direct-KKT / S+K single-seed ablation.

This is separate from the gate-based JCB driver.  It retains its Jiang--Fu
Example-2 dynamics, 20 ZOH controls, bounded policy, fixed-node dual
reconstruction, and event-located HDS correction, but never selects a branch.
"""
from __future__ import annotations

import argparse, copy, csv, json, platform, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch import nn

from run_jcb2d_jiang_valc import (Config as BaseConfig, Policy, g, gdot, initial, lhs,
                                  objective_np, ode, torch_rollout)
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 20260771
    supervised_epochs: int = 200
    continuation_epochs: int = 20
    learning_rate: float = 1e-3
    continuation_learning_rate: float = 1e-5
    kkt_weight: float = 1e-3
    augmented_penalty: float = 10.0
    anchor_weight: float = 0.1
    value_weight: float = 0.1
    lambda_grid: int = 31
    validation_points: int = 60
    test_points: int = 100
    gradient_clip_norm: float = 1.0

    @property
    def total_epochs(self): return self.supervised_epochs + self.continuation_epochs


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def seed_everything(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(labels: Path, output: Path, cfg: Config):
    base = BaseConfig(epochs=cfg.supervised_epochs, kkt_epochs=cfg.continuation_epochs,
                      lambda_grid=cfg.lambda_grid, validation_points=cfg.validation_points,
                      test_points=cfg.test_points, seed=cfg.seed, kkt_weight=cfg.kkt_weight)
    cache = output / "teacher_labels_and_reconstructed_duals.npz"
    if cache.exists():
        d = np.load(cache); return base, d["states"], d["controls"], d["objective"], d["duals"], d["reconstruction_residual"]
    # The supplied .mat is the authoritative source of all 900 teacher
    # states/controls/objectives.  Reuse only the already reconstructed
    # fixed-node numerical quantities after checking an exact label match;
    # this avoids a 900-point duplicate numerical reconstruction, not a reuse
    # of any trained policy or deployment result.
    raw = loadmat(labels)
    states=np.asarray(raw["initialStates"],float); controls=np.asarray(raw["controls"],float); objective=np.asarray(raw["objectives"],float).reshape(-1)
    reconstruction_source = ROOT / "kkt_collocation/results/jcb2d_jiang_900_warm/teacher_labels.npz"
    prior=np.load(reconstruction_source)
    if not (np.array_equal(states,prior["initial_state"]) and np.array_equal(controls,prior["controls"]) and np.array_equal(objective,prior["objective"])):
        raise ValueError("The cached finite-dimensional dual reconstruction does not exactly match the requested 900 labels")
    duals, residual=prior["path_duals"],prior["kkt_reconstruction_residual"]
    if len(states) != 900: raise ValueError(f"Expected 900 JCB labels, got {len(states)}")
    np.savez_compressed(cache, states=states, controls=controls, objective=objective, duals=duals, reconstruction_residual=residual)
    return base, states, controls, objective, duals, residual


def tensors(states, controls, objective, duals):
    p = torch.tensor(states, dtype=torch.float32); u = torch.tensor(controls, dtype=torch.float32)
    j = torch.tensor(objective[:, None], dtype=torch.float32); d = torch.tensor(duals, dtype=torch.float32)
    pm, ps = p.mean(0), p.std(0, unbiased=False).clamp_min(1e-6)
    jm, js = j.mean(), j.std(unbiased=False).clamp_min(1e-6)
    return p, u, j, d, pm, ps, jm, js


def terms(model, p, uref, jref, duals, pm, ps, jm, js, base, cfg, *, kkt: bool, anchor=None):
    predicted_j, u = model((p - pm) / ps)
    control = nn.functional.mse_loss(u, uref); objective = nn.functional.mse_loss(predicted_j, (jref - jm) / js)
    out = {"control_mse": control, "objective_mse": objective, "supervised": control + cfg.value_weight * objective}
    if kkt:
        jroll, nodes = torch_rollout(p, u, base)
        raw = augmented_lagrangian_kkt_residual(jroll, u, nodes, duals, cfg.augmented_penalty)
        out.update({"kkt_raw": raw.total, "kkt_loss": raw.total / raw.total.detach().clamp_min(1.),
                    "stationarity": raw.stationarity, "primal": raw.primal_feasibility, "complementarity": raw.complementarity})
    if anchor is not None: out["anchor"] = nn.functional.mse_loss(u, anchor)
    return out


def fit(method, data, base, cfg, output: Path):
    states, controls, objective, duals = data
    p, uref, jref, d, pm, ps, jm, js = tensors(states, controls, objective, duals)
    seed_everything(cfg.seed); model = Policy(base.zoh_steps)
    history, failure = [], None; started = time.perf_counter()
    stages = [("S", cfg.total_epochs, False, cfg.learning_rate, None)] if method == "S" else []
    if method == "S+K": stages = [("S", cfg.supervised_epochs, False, cfg.learning_rate, None)]
    if method == "K-only": stages = [("K-only", cfg.total_epochs, True, cfg.learning_rate, None)]
    for stage, epochs, use_kkt, lr, anchor in stages:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            try:
                z = terms(model, p, uref, jref, d, pm, ps, jm, js, base, cfg, kkt=use_kkt, anchor=anchor)
                loss = z["kkt_loss"] if method == "K-only" else z["supervised"]
                if use_kkt and method != "K-only": loss = loss + cfg.kkt_weight * z["kkt_loss"]
                if anchor is not None: loss = loss + cfg.anchor_weight * z["anchor"]
                if not bool(torch.isfinite(loss).all()): raise FloatingPointError("non-finite loss")
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm); opt.step()
                if epoch == 1 or epoch == epochs or epoch % 20 == 0:
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()), **{k: float(v.detach()) for k,v in z.items()}})
            except (FloatingPointError, RuntimeError) as exc:
                failure = f"{type(exc).__name__}: {exc}"; break
        if failure: break
        if method == "S+K" and stage == "S":
            with torch.no_grad(): anchor = model((p-pm)/ps)[1].detach()
            stages.append(("K-continuation", cfg.continuation_epochs, True, cfg.continuation_learning_rate, anchor))
    model.eval()
    with torch.enable_grad(): z = terms(model,p,uref,jref,d,pm,ps,jm,js,base,cfg,kkt=True)
    record = {"completed": failure is None, "numerical_failure": failure is not None, "failure_reason": failure,
              "train_seconds": time.perf_counter()-started, "final_control_mse": float(z["control_mse"].detach()),
              "final_objective_mse": float(z["objective_mse"].detach()), "final_kkt_residual": float(z["kkt_raw"].detach()),
              "kkt_components": {k: float(z[k].detach()) for k in ("stationarity","primal","complementarity")}}
    torch.save({"model":model.state_dict(),"normalization":{"mean":pm,"std":ps},"training":record}, output/f"{method}.pth")
    dump(output/f"training_log_{method}.json", {"training":record,"history":history})
    return model, pm.numpy(), ps.numpy(), record


def evaluate(method, model, mean, std, points, base, cfg, index_offset=0):
    x = torch.tensor((points-mean)/std, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10): model(x[:10]) # warm-up excluded from timing
        start=time.perf_counter(); _, controls=model(x); inference=(time.perf_counter()-start)/len(points)
    u = controls.numpy(); corrector=HDSLambdaCorrector(ode,g,gdot,(base.u_min,base.u_max),HDSLambdaConfig(grid_size=cfg.lambda_grid,safety_margin=base.margin,max_step_fraction=200.0))
    rows=[]
    for index,(point,nominal) in enumerate(zip(points,u), start=index_offset):
        raw=float(corrector.audit(initial(point),nominal,base.dt)); nominal_j=objective_np(point,nominal,base)
        start=time.perf_counter(); outcome=corrector.correct(initial(point),nominal,base.dt); hds=time.perf_counter()-start
        accept=bool(outcome.accepted); applied=outcome.controls if accept else nominal
        rows.append({"method":method,"index":index,"x1_0":point[0],"x2_0":point[1],"nominal_hds_max_g":raw,"accepted":accept,"fallback":not accept,
                     "applied_hds_max_g":float(corrector.audit(initial(point),applied,base.dt)) if accept else np.nan,
                     "nominal_objective":nominal_j,"applied_objective":objective_np(point,applied,base) if accept else np.nan,
                     "objective_change":objective_np(point,applied,base)-nominal_j if accept else np.nan,
                     "corrected_segments":int(sum(s.corrected for s in outcome.segments)),"inference_seconds":inference,"hds_seconds":hds,"total_deployment_seconds":inference+hds})
    a=lambda k:np.asarray([r[k] for r in rows],float); acc=a("accepted").astype(bool)
    return rows,{"points":len(rows),"nominal_objective":float(a("nominal_objective").mean()),"nominal_violation_rate":float((a("nominal_hds_max_g")>1e-8).mean()),"nominal_max_g":float(a("nominal_hds_max_g").max()),"acceptance_rate":float(acc.mean()),"fallback_rate":float(1-acc.mean()),"final_max_g":float(np.nanmax(a("applied_hds_max_g"))),"mean_corrected_segments":float(a("corrected_segments").mean()),"hds_objective":float(np.nanmean(a("applied_objective"))),"mean_hds_objective_change":float(np.nanmean(a("objective_change"))),"mean_inference_seconds":float(a("inference_seconds").mean()),"mean_hds_seconds":float(a("hds_seconds").mean()),"mean_total_deployment_seconds":float(a("total_deployment_seconds").mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--labels",type=Path,default=ROOT/"kkt_collocation/results/jcb2d_jiang_900_warm/JCB_2D_VALC_labels.mat"); ap.add_argument("--output",type=Path,default=ROOT/"kkt_collocation/results/jcb2d_900_two_stage_vs_kkt_only_single"); ap.add_argument("--seed",type=int,default=20260771); ap.add_argument("--anchor-weight",type=float,default=.1); ap.add_argument("--phase",choices=("full","train","evaluate","aggregate"),default="full"); ap.add_argument("--method",choices=("S","K-only","S+K","all"),default="all"); ap.add_argument("--test-start",type=int,default=0); ap.add_argument("--test-stop",type=int); args=ap.parse_args()
    cfg=Config(seed=args.seed,anchor_weight=args.anchor_weight)
    if args.output.exists() and any(args.output.iterdir()) and not (args.output/"config.json").exists(): raise FileExistsError(f"Refusing to overwrite existing result directory: {args.output}")
    args.output.mkdir(parents=True,exist_ok=True)
    raw=loadmat(args.labels); metadata=raw["metadata"][0,0]; initial_guess=str(metadata["initial_guess"].reshape(-1)[0])
    base, states, controls, objective, duals, recon = load_data(args.labels,args.output,cfg)
    validation=lhs(cfg.validation_points,base,cfg.seed+1); test=lhs(cfg.test_points,base,cfg.seed+2); np.save(args.output/"validation_states.npy",validation); np.save(args.output/"test_states.npy",test); savemat(args.output/"coldstart_test_states.mat", {"testStates":test})
    config={"experiment":asdict(cfg),"base_problem":asdict(base),"labels":str(args.labels),"label_count":len(states),"label_initialization":initial_guess,"label_warm_start_note":"The supplied 900 Jiang--Fu labels used a preceding-label warm start. All three branches use exactly these same labels.","konly_note":"K-only uses reconstructed finite-dimensional discretized-NLP multipliers, but no reference control or objective supervision.","hds_note":"HDS is continuous-time numerical audit evidence under the declared model and numerical settings; it is not an absolute safety guarantee."}; dump(args.output/"config.json",config)
    methods={}
    if args.phase=="aggregate":
        methods={}
        for method in ("S","K-only","S+K"):
            training=json.loads((args.output/f"training_log_{method}.json").read_text(encoding="utf-8"))["training"]
            files=sorted(args.output.glob(f"test_sample_log_{method}_part_*.csv"))
            if not files:
                methods[method]={"training":training,"deployment":{"not_evaluable":True,"reason":"no HDS result because training was not deployable"}}; continue
            rows=[]
            for file in files:
                for row in csv.DictReader(file.open(encoding="utf-8")):
                    rows.append({k:(row[k] if k=="method" else ((row[k]=="True") if k in ("accepted","fallback") else float(row[k]))) for k in row})
            indices=[int(r["index"]) for r in rows]
            if len(indices)!=cfg.test_points or set(indices)!=set(range(cfg.test_points)):
                raise ValueError(f"Incomplete or duplicate chunks for {method}: n={len(indices)}, unique={len(set(indices))}, expected={cfg.test_points}")
            a=lambda k:np.asarray([r[k] for r in rows],float); acc=a("accepted").astype(bool)
            methods[method]={"training":training,"deployment":{"points":len(rows),"nominal_objective":float(a("nominal_objective").mean()),"nominal_violation_rate":float((a("nominal_hds_max_g")>1e-8).mean()),"nominal_max_g":float(a("nominal_hds_max_g").max()),"acceptance_rate":float(acc.mean()),"fallback_rate":float(1-acc.mean()),"final_max_g":float(np.nanmax(a("applied_hds_max_g"))),"mean_corrected_segments":float(a("corrected_segments").mean()),"hds_objective":float(np.nanmean(a("applied_objective"))),"mean_hds_objective_change":float(np.nanmean(a("objective_change"))),"mean_inference_seconds":float(a("inference_seconds").mean()),"mean_hds_seconds":float(a("hds_seconds").mean()),"mean_total_deployment_seconds":float(a("total_deployment_seconds").mean())}}
        report={"config":config,"methods":methods,"reconstructed_dual_max_residual":float(recon.max()),"coldstart_reference_status":"Not yet available: frozen test_states.npy is saved for a separate cold-start Jiang--Fu Algorithm 1 reference run. Relative objective gaps are deliberately not reported before that run."}; dump(args.output/"single_seed_summary.json",report); print(json.dumps(report,ensure_ascii=False,indent=2)); return
    requested=("S","K-only","S+K") if args.method=="all" else (args.method,)
    for method in requested:
        checkpoint=args.output/f"{method}.pth"
        if args.phase in ("full","train") and not checkpoint.exists():
            print(f"training {method}",flush=True); model,mean,std,training=fit(method,(states,controls,objective,duals),base,cfg,args.output)
        else:
            payload=torch.load(checkpoint,map_location="cpu",weights_only=False); model=Policy(base.zoh_steps); model.load_state_dict(payload["model"]); model.eval(); mean=payload["normalization"]["mean"].numpy();std=payload["normalization"]["std"].numpy();training=payload["training"]
        if args.phase in ("full","evaluate") and training["completed"]:
            stop=len(test) if args.test_stop is None else args.test_stop; rows,deployment=evaluate(method,model,mean,std,test[args.test_start:stop],base,cfg,args.test_start)
            with (args.output/f"test_sample_log_{method}_part_{args.test_start:03d}_{stop:03d}.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        elif training["completed"]: deployment={"not_yet_evaluated":True}
        else: deployment={"not_evaluable":True,"reason":"training numerical failure before a deployable 220-epoch policy"}
        methods[method]={"training":training,"deployment":deployment}
    dump(args.output/"validation_summary.json",{"points":len(validation),"seed":cfg.seed+1,"note":"Fixed independent LHS validation set; no gate and no model selection."})
    if args.phase=="full" and args.method=="all":
        report={"config":config,"methods":methods,"reconstructed_dual_max_residual":float(recon.max()),"coldstart_reference_status":"Not yet available: frozen test_states.npy is saved for a separate cold-start Jiang--Fu Algorithm 1 reference run. Relative objective gaps are deliberately not reported before that run."}; dump(args.output/"single_seed_summary.json",report); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
