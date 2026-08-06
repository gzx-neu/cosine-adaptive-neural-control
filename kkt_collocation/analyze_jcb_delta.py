"""Resume-safe HDS objective-change accounting for a completed JCB run."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from run_jcb_valc import Config,Policy,initial,rollout,ode,g,gdot
from offline_safe_control.hds_lambda_corrector import HDSLambdaCorrector,HDSLambdaConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args(); d=json.loads((a.experiment/'summary.json').read_text());c=Config(**d['config']);q=torch.load(a.experiment/'supervised.pth',map_location='cpu',weights_only=False);m=Policy(c.zoh_steps);m.load_state_dict(q['model']);m.eval();x=np.array([r['p'] for r in d['test']])
 with torch.no_grad(): u=m(torch.tensor(((x-q['mean'])/q['std'])[:,None],dtype=torch.float32))[1].numpy()
 out=a.experiment/'correction_delta.csv'; rows=list(csv.DictReader(out.open())) if out.exists() else []; start=len(rows)
 if start and int(rows[-1]['index'])!=start-1: raise RuntimeError('non-contiguous result file')
 cor=HDSLambdaCorrector(ode,g,gdot,(c.u_min,c.u_max),HDSLambdaConfig(grid_size=c.lambda_grid,safety_margin=c.margin,max_step_fraction=200.)); fields=['index','p','nominal_cost','applied_cost','delta_j','relative_delta_percent','corrected_segments']
 with out.open('a' if start else 'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if not start:w.writeheader()
  for i in range(start,len(x)):
   nom=rollout(float(x[i]),u[i],c)[2][3]; z=cor.correct(initial(float(x[i])),u[i],c.dt)
   if not z.accepted: raise RuntimeError(f'fallback at {i}')
   app=rollout(float(x[i]),z.controls,c)[2][3]; w.writerow({'index':i,'p':x[i],'nominal_cost':nom,'applied_cost':app,'delta_j':app-nom,'relative_delta_percent':(app-nom)/nom*100,'corrected_segments':sum(s.corrected for s in z.segments)});f.flush();print(f'{i+1}/{len(x)}')
 rows=list(csv.DictReader(out.open())); v=lambda k:np.array([float(r[k]) for r in rows]); summary={'points':len(rows),'mean_delta_j':float(v('delta_j').mean()),'std_delta_j':float(v('delta_j').std(ddof=1)),'mean_relative_delta_percent':float(v('relative_delta_percent').mean()),'std_relative_delta_percent':float(v('relative_delta_percent').std(ddof=1)),'max_relative_delta_percent':float(v('relative_delta_percent').max()),'mean_corrected_segments':float(v('corrected_segments').mean())};(a.experiment/'correction_delta_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
