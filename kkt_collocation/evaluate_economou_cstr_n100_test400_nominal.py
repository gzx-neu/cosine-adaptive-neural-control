"""Fast nominal discrete-objective check on frozen 400 cold-start references."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch

from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import ROOT, rollout_flat
from screen_economou_cstr_30x30 import EconomouScreenConfig

TRAIN = ROOT / "kkt_collocation/results/economou_cstr_n100_all900_s_sk_konly_training_seed20260771"
REF = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0"
OUT = ROOT / "kkt_collocation/results/economou_cstr_n100_test400_nominal_s_sk_while_konly_training"

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    cfg_dict=json.loads((REF/'summary.json').read_text(encoding='utf-8'))['config']
    for key in ('ti_bounds_K','flow_bounds','ca_initial_range','temperature_initial_range_K'): cfg_dict[key]=tuple(cfg_dict[key])
    cfg=EconomouScreenConfig(**cfg_dict)
    rows=[json.loads(x) for x in (REF/'records.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    rows.sort(key=lambda x:int(x['index']))
    if len(rows)!=400 or not all(r.get('success') for r in rows): raise ValueError('Expected 400 successful references')
    x=np.asarray([r['initial_state'] for r in rows],float)
    ref=np.asarray([r['objective'] for r in rows],float)
    result={'reference':'400 independently cold-start N=100 RK10 discrete-NLP objectives; no references used in training.', 'methods':{}}
    for name in ('S','S+K'):
        ck=torch.load(TRAIN/f'{name}.pth',map_location='cpu',weights_only=False)
        model=PolicyValue(cfg).double().eval(); model.load_state_dict(ck['model'])
        mean=torch.as_tensor(ck['state_mean'],dtype=torch.float64); std=torch.as_tensor(ck['state_std'],dtype=torch.float64)
        xx=torch.as_tensor(x,dtype=torch.float64)
        with torch.no_grad(): _,u=model((xx[:,[0,2]]-mean)/std)
        flat=u.reshape(len(x),-1).detach().requires_grad_(False)
        objective,g=rollout_flat(xx,flat,cfg)
        pred=objective.detach().numpy(); maxg=g.detach().numpy().max(1)
        gap=100*(pred-ref)/np.maximum(np.abs(ref),1e-12)
        output=[]
        for i in range(len(x)): output.append({'index':i,'CA0':x[i,0],'T0_K':x[i,2],'reference_discrete_objective':ref[i], 'network_discrete_objective':pred[i], 'relative_nominal_gap_percent':gap[i], 'discrete_max_g':maxg[i]})
        with (OUT/f'{name}_nominal.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
        result['methods'][name]={'mean_relative_nominal_gap_percent':float(gap.mean()),'std_relative_nominal_gap_percent':float(gap.std()),'median_relative_nominal_gap_percent':float(np.median(gap)),'p95_relative_nominal_gap_percent':float(np.quantile(gap,.95)), 'discrete_violation_rate_percent':float(100*np.mean(maxg>1e-8)),'max_discrete_g':float(maxg.max())}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
