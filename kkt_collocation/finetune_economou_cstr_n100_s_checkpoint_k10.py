"""Exploratory ten-epoch KKT continuation from the completed all-label S model."""
from __future__ import annotations
import copy
import json
import time
from dataclasses import asdict
import numpy as np
import torch
from torch import nn

from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import Config, ROOT, dump_json, kkt_terms, load_labels
from screen_economou_cstr_30x30 import EconomouScreenConfig

SOURCE = ROOT / "kkt_collocation/results/economou_cstr_n100_all900_s_sk_konly_training_seed20260771/S.pth"
LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl"
OUT = ROOT / "kkt_collocation/results/economou_cstr_n100_all900_s220_then_k10_exploratory"

def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    cfg=Config(); raw=json.loads((LABELS.parent/'summary.json').read_text(encoding='utf-8'))['config']
    for key in ('ti_bounds_K','flow_bounds','ca_initial_range','temperature_initial_range_K'): raw[key]=tuple(raw[key])
    cstr=EconomouScreenConfig(**raw)
    states,u_ref,j_ref,path,bounds,_=load_labels(LABELS)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(SOURCE,map_location=device,weights_only=False)
    model=PolicyValue(cstr).to(device); model.load_state_dict(ck['model'])
    x=torch.tensor(states,dtype=torch.float32,device=device); ur=torch.tensor(u_ref,dtype=torch.float32,device=device)
    jr=torch.tensor(j_ref[:,None],dtype=torch.float32,device=device); dr=torch.tensor(path,dtype=torch.float32,device=device); br=torch.tensor(bounds,dtype=torch.float32,device=device)
    mean=torch.as_tensor(ck['state_mean'],dtype=torch.float32,device=device); std=torch.as_tensor(ck['state_std'],dtype=torch.float32,device=device)
    inputs=(x[:,[0,2]]-mean)/std; jmean,jstd=jr.mean(),jr.std().clamp_min(1e-6)
    low=torch.tensor([cstr.ti_bounds_K[0],cstr.flow_bounds[0]],dtype=torch.float32,device=device); span=torch.tensor([70.,1.],dtype=torch.float32,device=device)
    with torch.no_grad(): _,anchor=model(inputs); anchor=anchor.detach()
    history=[]; started=time.perf_counter(); optimizer=torch.optim.Adam(model.parameters(),lr=cfg.continuation_lr)
    for epoch in range(1,11):
        pred_j,pred_u=model(inputs)
        control=nn.functional.mse_loss((pred_u-low)/span,(ur-low)/span)
        obj=nn.functional.mse_loss(pred_j,(jr-jmean)/jstd)
        supervised=control+.1*obj
        kkt=kkt_terms(x,pred_u.reshape(len(x),-1),dr,br,cstr,cfg.augmented_penalty)
        anchor_loss=nn.functional.mse_loss((pred_u-low)/span,(anchor-low)/span)
        loss=supervised+cfg.kkt_weight*kkt['total']/kkt['total'].detach().clamp_min(1.)+cfg.anchor_weight*anchor_loss
        if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss at epoch {epoch}')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        record={'epoch':epoch,'loss':float(loss.detach()),'control_mse':float(control.detach()),'objective_mse':float(obj.detach()),'anchor':float(anchor_loss.detach()), **{f'kkt_{k}':float(v.detach()) for k,v in kkt.items() if k not in ('objective','path_g')}}
        history.append(record); print(f"K10 {epoch}/10 loss={record['loss']:.3e} kkt={record['kkt_total']:.3e}",flush=True)
    with torch.enable_grad():
        pred_j,pred_u=model(inputs); control=nn.functional.mse_loss((pred_u-low)/span,(ur-low)/span); obj=nn.functional.mse_loss(pred_j,(jr-jmean)/jstd); final=kkt_terms(x,pred_u.reshape(len(x),-1),dr,br,cstr,cfg.augmented_penalty)
    training={'completed':True,'seconds':time.perf_counter()-started,'control_mse_normalized':float(control.detach()),'objective_mse_normalized':float(obj.detach()),'kkt_residual':float(final['total'].detach()),'kkt_stationarity':float(final['stationarity'].detach()),'kkt_primal':float(final['primal'].detach())}
    torch.save({'model':model.state_dict(),'state_mean':ck['state_mean'],'state_std':ck['state_std'],'config':asdict(cfg),'training':training},OUT/'S+K10_from_S220.pth')
    dump_json(OUT/'training_log.json',{'protocol':'Exploratory: append 10 KKT-continuation updates to the completed 220-epoch S model; 230 total updates, so not an equal-update formal comparison.','source_checkpoint':str(SOURCE),'training':training,'history':history})
    dump_json(OUT/'summary.json',{'status':'completed','training':training,'source_checkpoint':str(SOURCE),'config':asdict(cfg),'continuation_epochs':10,'reference_use':'No reference test point was used.'})
if __name__=='__main__': main()
