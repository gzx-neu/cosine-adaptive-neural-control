"""Train a JCB policy by distilling HDS-corrected CVP50 QP controls.

This is deliberately not a KKT loss: its sole training target is the control
that the declared deployment HDS/lambda procedure actually accepts.
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

from generate_jcb_reduced_kkt_data import JCBConfig
from run_jcb2d_jiang_valc import g, gdot, initial, ode
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from run_jcb2d_qp_cvp50_two_stage_vs_s import (Experiment, Policy, actual_objective, dump, evaluate, load_labels, seed_all)

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',type=Path,default=ROOT/'kkt_collocation/results/jcb2d_qp_cvp50_nodes10_30x30/records.jsonl'); ap.add_argument('--baseline',type=Path,default=ROOT/'kkt_collocation/results/jcb2d_qp_cvp50_nodes10_two_stage_vs_s_single'); ap.add_argument('--output',type=Path,default=ROOT/'kkt_collocation/results/jcb2d_qp_cvp50_hds_distill_single'); ap.add_argument('--seed',type=int,default=20260811); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True); exp=Experiment(seed=args.seed); seed_all(exp.seed)
    meta=json.loads((args.labels.parent/'summary.json').read_text(encoding='utf-8')); raw=dict(meta['config'])
    for k in ('control_bounds','x1_initial_range','x2_initial_range'): raw[k]=tuple(raw[k])
    cfg=JCBConfig(**raw); p,u,j,_mu,_bound,_res=load_labels(args.labels)
    test=np.load(args.baseline/'test_initial_conditions.npy'); refs=json.loads((args.baseline/'cold_start_references.json').read_text(encoding='utf-8'))
    if len(test)!=exp.test_count or len(refs)!=len(test): raise ValueError('Baseline frozen test/reference set is incomplete')
    corrector=HDSLambdaCorrector(ode,g,gdot,cfg.control_bounds,HDSLambdaConfig(grid_size=exp.lambda_grid,safety_margin=0.,max_step_fraction=1.))
    corrected=[]; target_j=[]; corrected_count=0; started=time.perf_counter()
    for i,(state,control) in enumerate(zip(p,u)):
        peak=float(corrector.audit(initial(state),control,cfg.zoh_duration)); ans=corrector.correct(initial(state),control,cfg.zoh_duration) if peak>1e-8 else None
        if ans is not None and not ans.accepted: raise RuntimeError(f'HDS teacher correction failed at label {i}')
        applied=np.asarray(ans.controls if ans is not None else control,float); corrected.append(applied); target_j.append(actual_objective(state,applied,cfg)); corrected_count+=int(ans is not None and np.any(np.abs(applied-control)>1e-12))
    uh=np.asarray(corrected,np.float32); jh=np.asarray(target_j,np.float32)[:,None]
    np.savez_compressed(args.output/'hds_distilled_teacher.npz',states=p,original_controls=u,hds_controls=uh,hds_objective=jh)
    pt=torch.tensor(p,dtype=torch.float32); ut=torch.tensor(uh,dtype=torch.float32); jt=torch.tensor(jh,dtype=torch.float32); mean=p.mean(0); std=p.std(0).clip(1e-6); nx=torch.tensor((p-mean)/std,dtype=torch.float32); jm,js=jt.mean(),jt.std().clamp_min(1e-6); low,high=cfg.control_bounds
    model=Policy(cfg.zoh_steps,low,high); opt=torch.optim.Adam(model.parameters(),lr=exp.supervised_lr); hist=[]; train_start=time.perf_counter()
    for epoch in range(1,exp.total_epochs+1):
        pred,out=model(nx); control=nn.functional.mse_loss((out-low)/(high-low),(ut-low)/(high-low)); value=nn.functional.mse_loss(pred,(jt-jm)/js); loss=control+exp.value_weight*value
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),exp.gradient_clip_norm); opt.step()
        if epoch==1 or epoch%20==0 or epoch==exp.total_epochs: hist.append({'epoch':epoch,'loss':float(loss.detach()),'control_mse':float(control.detach()),'objective_mse':float(value.detach())})
    record={'completed':True,'seconds':time.perf_counter()-train_start,'control_mse_normalized':float(control.detach()),'objective_mse_normalized':float(value.detach())}
    torch.save({'model':model.state_dict(),'state_mean':mean,'state_std':std,'training':record},args.output/'S+HDS-distill.pth'); dump(args.output/'training_log.json',{'training':record,'history':hist})
    deployment=evaluate('S+HDS-distill',model,mean,std,test,refs,cfg,exp,args.output); cold=float(np.mean([x['solve_seconds'] for x in refs])); deployment['mean_cold_qp_seconds']=cold; deployment['speedup_vs_cold_qp']=cold/deployment['mean_total_predeployment_seconds']
    final={'status':'completed','config':asdict(exp),'label_source':str(args.labels),'teacher_construction':{'method':'same fixed zero-control cold-start CVP50 QP controls corrected offline by the declared event-located HDS/31-lambda procedure','labels':len(p),'teacher_sequences_modified':corrected_count,'preparation_seconds':time.perf_counter()-started},'frozen_test_source':str(args.baseline/'test_initial_conditions.npy'),'cold_reference_source':str(args.baseline/'cold_start_references.json'),'method':{'training':record,'deployment':deployment},'hds_statement':'Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.'}
    dump(args.output/'summary.json',final); (args.output/'summary_table.md').write_text(f"| Method | HDS gap (%) | Nominal violation | Corrected segments | Accepted / fallback | Speedup |\n|---|---:|---:|---:|---:|---:|\n| S+HDS-distill | {deployment['mean_hds_relative_gap_percent']:.3f} | {deployment['nominal_violation_rate_percent']:.1f}% | {deployment['mean_corrected_segments']:.2f} | {deployment['accepted_network_samples']} / {deployment['fallback_samples']} | {deployment['speedup_vs_cold_qp']:.2f}x |\n",encoding='utf-8'); print(json.dumps(final,indent=2,default=str))

if __name__=='__main__': main()
