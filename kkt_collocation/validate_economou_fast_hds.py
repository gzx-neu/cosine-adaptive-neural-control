"""Pointwise equivalence audit of fast versus legacy strict Economou HDS."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,torch
from screen_economou_cstr_30x30 import EconomouScreenConfig
from run_economou_cstr_supervised_hds import PolicyValue,lhs_states
from economou_cstr_hds_fast import fast_correct,strict_trace
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'kkt_collocation/results/economou_cstr_two_stage_vs_kkt_only_single';c=EconomouScreenConfig();states=lhs_states(c,100,20260772);np.save(out/'hds_equivalence_states.npy',states)
 q=torch.load(out/'S.pth',map_location='cpu',weights_only=False);m=PolicyValue(c);m.load_state_dict(q['model']);m.eval();x=torch.tensor((states[:,[0,2]]-q['mean'].numpy())/q['std'].numpy(),dtype=torch.float32)
 with torch.no_grad():_,u=m(x)
 rows=[]
 for i,(s,v) in enumerate(zip(states,u.numpy())):
  t=time.perf_counter();a=strict_trace(s,v,c);ts=time.perf_counter()-t;t=time.perf_counter();b=fast_correct(s,v,c);tf=time.perf_counter()-t
  rows.append({'index':i,'strict_accepted':a['accepted'],'fast_accepted':b['accepted'],'strict_peak':a['final_peak'],'fast_peak':b['final_peak'],'strict_segments':a['corrected_segments'],'fast_segments':b['corrected_segments'],'lambda_match':np.allclose(a['lambdas'],b['lambdas'],atol=1e-12),'controls_match':np.allclose(a['controls'],b['controls'],atol=1e-8),'strict_seconds':ts,'fast_seconds':tf})
 bad=[r for r in rows if not(r['strict_accepted']==r['fast_accepted'] and r['lambda_match'] and r['controls_match'] and abs(r['strict_peak']-r['fast_peak'])<1e-7 and r['strict_segments']==r['fast_segments'])]
 (out/'hds_fast_equivalence.json').write_text(json.dumps({'samples':100,'mismatches':len(bad),'mean_strict_seconds':float(np.mean([r['strict_seconds'] for r in rows])),'mean_fast_seconds':float(np.mean([r['fast_seconds'] for r in rows])),'rows':rows},indent=2),encoding='utf8')
 if bad:raise RuntimeError(f'{len(bad)} fast-HDS mismatches')
 print('equivalent',len(rows))
if __name__=='__main__':main()
