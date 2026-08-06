"""Same-point cold-start NLP comparison for the JCB VALC experiment."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from run_jcb_valc import Config,Policy,initial,rollout,ode,g,gdot,solve_reference
from offline_safe_control.hds_lambda_corrector import HDSLambdaCorrector,HDSLambdaConfig

def mean_sd(x): return {'mean':float(np.mean(x)),'sample_std':float(np.std(x,ddof=1))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();d=json.loads((a.experiment/'summary.json').read_text());c=Config(**d['config']);q=torch.load(a.experiment/'supervised.pth',map_location='cpu',weights_only=False);m=Policy(c.zoh_steps);m.load_state_dict(q['model']);m.eval();x=np.array([r['p'] for r in d['test']])
 t0=time.perf_counter()
 with torch.no_grad():u=m(torch.tensor(((x-q['mean'])/q['std'])[:,None],dtype=torch.float32))[1].numpy()
 infer=(time.perf_counter()-t0)/len(x);cor=HDSLambdaCorrector(ode,g,gdot,(c.u_min,c.u_max),HDSLambdaConfig(grid_size=c.lambda_grid,safety_margin=c.margin,max_step_fraction=200.));out=a.experiment/'coldstart_comparison.csv';rows=list(csv.DictReader(out.open())) if out.exists() else [];start=len(rows)
 if start and int(rows[-1]['index'])!=start-1: raise RuntimeError('non-contiguous result file')
 fields=['index','p','valc_objective','reference_objective','relative_objective_difference_percent','valc_seconds','reference_seconds','valc_peak','reference_peak','corrected_segments']
 with out.open('a' if start else 'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if not start:w.writeheader()
  for i in range(start,len(x)):
   st=time.perf_counter();z=cor.correct(initial(float(x[i])),u[i],c.dt);vt=time.perf_counter()-st+infer
   if not z.accepted:raise RuntimeError(f'VALC fallback at {i}')
   vj=rollout(float(x[i]),z.controls,c)[2][3];vp=cor.audit(initial(float(x[i])),z.controls,c.dt)
   st=time.perf_counter();ru,rj=solve_reference(float(x[i]),c,None);rt=time.perf_counter()-st;rp=cor.audit(initial(float(x[i])),ru,c.dt)
   w.writerow({'index':i,'p':x[i],'valc_objective':vj,'reference_objective':rj,'relative_objective_difference_percent':abs(vj-rj)/max(abs(rj),1e-12)*100,'valc_seconds':vt,'reference_seconds':rt,'valc_peak':vp,'reference_peak':rp,'corrected_segments':sum(s.corrected for s in z.segments)});f.flush();print(f'{i+1}/{len(x)}')
 rows=list(csv.DictReader(out.open()));v=lambda k:np.array([float(r[k]) for r in rows]);s={'points':len(rows),'hds_acceptance':'%d/%d'%(len(rows),len(rows)),'valc_time_seconds':mean_sd(v('valc_seconds')),'reference_time_seconds':mean_sd(v('reference_seconds')),'relative_objective_difference_percent':mean_sd(v('relative_objective_difference_percent')),'valc_objective':mean_sd(v('valc_objective')),'reference_objective':mean_sd(v('reference_objective')),'valc_max_hds_g':float(v('valc_peak').max()),'reference_max_hds_g':float(v('reference_peak').max()),'mean_corrected_segments':float(v('corrected_segments').mean())};(a.experiment/'coldstart_comparison_summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()
