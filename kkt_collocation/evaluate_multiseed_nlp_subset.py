"""Evaluate final multi-seed models on the existing same-point NLP references."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig,HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import constraint as vg,constraint_derivative as vgdot,predict as vpredict,terminal_cost,vdp_ode
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork,TrainConfig
from kkt_collocation.run_penicillin_ablation import DT,UMAX,Policy,g as pg,gdot as pgdot,ode as pode,predict as ppredict,terminal_product

def _reference(reference_dir:Path):
    rows=list(csv.DictReader((reference_dir/'per_sample.csv').open(encoding='utf-8')))
    # NLP objective is identical across model rows; retain first occurrence.
    values={int(r['sample_index']):float(r['nlp_reference_objective']) for r in rows}
    return np.load(reference_dir/'reference_test_indices.npy'),np.asarray([values[i] for i in range(len(values))])

def _vdp(directory:Path,indices,reference,device,model_keys=None):
    cp=torch.load(directory/'models.pth',map_location=device,weights_only=False)
    all_states=np.load(directory/'test_states.npy'); states=all_states[indices]
    corr=HDSLambdaCorrector(vdp_ode,vg,vgdot,(-.3,1.),HDSLambdaConfig(grid_size=31,max_step_fraction=100.))
    rows=[]
    for key,label in (('S','S'),('Penalty','S+Penalty'),('KKT','S+KKT')):
        if model_keys is not None and key not in model_keys:
            continue
        model=KKTPolicyValueNetwork(TrainConfig()).to(device); model.load_state_dict(cp[key]); model.eval()
        mean,std=cp['normalization'][key]; controls,inference=vpredict(model,np.asarray(mean),np.asarray(std),states,device)
        for i,(state,control,ref) in enumerate(zip(states,controls,reference)):
            started=time.perf_counter(); outcome=corr.correct(state,control,.5); elapsed=time.perf_counter()-started
            if not outcome.accepted: raise RuntimeError(f'VDP fallback at {directory} {label} {i}')
            applied=terminal_cost(state,outcome.controls,corr,.5)
            rows.append({'seed':cp['config']['seed'],'method':label,'sample_index':i,'reference_objective':ref,
                         'applied_objective':applied,'relative_absolute_gap':abs(applied-ref)/max(abs(ref),1e-12),
                         'accepted':True,'fallback':False,'applied_hds_max_g':corr.audit(state,outcome.controls,.5),
                         'corrected_segments':sum(s.corrected for s in outcome.segments),'hds_seconds':elapsed,'inference_seconds':inference})
    return rows

def _pen(directory:Path,indices,reference,device,model_keys=None):
    cp=torch.load(directory/'models.pth',map_location=device,weights_only=False)
    all_x2=np.load(directory/'test_x2.npy'); x2s=all_x2[indices]
    corr=HDSLambdaCorrector(pode,pg,pgdot,(0.,UMAX),HDSLambdaConfig(grid_size=31,max_step_fraction=100.))
    rows=[]
    for key,label in (('S','S'),('Penalty','S+Penalty'),('true_KKT','S+true-KKT')):
        if model_keys is not None and key not in model_keys:
            continue
        model=Policy().to(device); model.load_state_dict(cp[key]); model.eval()
        mean,std=cp['normalization'][key]; controls,inference=ppredict(model,float(mean),float(std),x2s,device)
        for i,(x2,control,ref) in enumerate(zip(x2s,controls,reference)):
            state=np.array([1.,x2,.001,250.]); started=time.perf_counter(); outcome=corr.correct(state,control,DT); elapsed=time.perf_counter()-started
            if not outcome.accepted: raise RuntimeError(f'Penicillin fallback at {directory} {label} {i}')
            applied=-terminal_product(x2,outcome.controls,corr)
            rows.append({'seed':cp['config']['seed'],'method':label,'sample_index':i,'reference_objective':ref,
                         'applied_objective':applied,'relative_absolute_gap':abs(applied-ref)/max(abs(ref),1e-12),
                         'accepted':True,'fallback':False,'applied_hds_max_g':corr.audit(state,outcome.controls,DT),
                         'corrected_segments':sum(s.corrected for s in outcome.segments),'hds_seconds':elapsed,'inference_seconds':inference})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--problem',choices=('vdp','penicillin'),required=True); p.add_argument('--inputs',nargs='+',type=Path,required=True); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--model-keys',nargs='+',default=None,help='Optional checkpoint keys to evaluate; use the selected deployment branch only for a focused comparison.'); a=p.parse_args()
    indices,reference=_reference(a.reference); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for directory in a.inputs: rows.extend(_vdp(directory,indices,reference,device,a.model_keys) if a.problem=='vdp' else _pen(directory,indices,reference,device,a.model_keys))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        per_seed=[]
        for seed in sorted({r['seed'] for r in rows}):
            group=[r for r in rows if r['method']==method and r['seed']==seed]
            per_seed.append({k:float(np.mean([r[k] for r in group])) for k in ('relative_absolute_gap','applied_hds_max_g','corrected_segments','hds_seconds','inference_seconds')})
        summary[method]={k:{'mean':float(np.mean([x[k] for x in per_seed])),'sample_std':float(np.std([x[k] for x in per_seed],ddof=1)),'per_seed':[x[k] for x in per_seed]} for k in per_seed[0]}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'per_sample.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'problem':a.problem,'reference_samples':len(indices),'training_seeds':sorted({r['seed'] for r in rows}),'methods':summary,'note':'All objectives are evaluated after sequential HDS-lambda correction on the same independent NLP-reference subset.'}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
