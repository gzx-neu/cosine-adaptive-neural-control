"""Same-fixed-point timing/quality comparison for the teacher penicillin code."""
from __future__ import annotations
import csv,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig,HDSLambdaCorrector
from kkt_collocation.generate_penicillin_kkt_data import PenicillinConfig,ReducedPenicillinProblem
from kkt_collocation.run_penicillin_ablation import DT,UMAX,Policy,g,gdot,ode,predict,terminal_product

def main():
    teacher=ROOT/'精简_修改好的青霉素'/'penicillin_teacher_original50_fixed4.csv'
    output=ROOT/'kkt_collocation'/'results'/'final_penicillin_fixed_teacher_original50_comparison'
    labels=ROOT/'kkt_collocation'/'data'/'penicillin_kkt_400_true_duals.pkl'
    model_dir=ROOT/'kkt_collocation'/'results'/'final_multiseed_penicillin400_penalty_seed20260761'
    trows=list(csv.DictReader(teacher.open(encoding='utf-8-sig'))); x2=np.asarray([float(r['x2_values']) for r in trows])
    with labels.open('rb') as h: data=pickle.load(h)
    seed_x2=np.asarray(data['initial_state'])[:,1]; seed_controls=np.asarray(data['optimal_controls'])
    problem=ReducedPenicillinProblem(PenicillinConfig(substeps_per_zoh=80),seed_x2,seed_controls)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); checkpoint=torch.load(model_dir/'models.pth',map_location=device,weights_only=False)
    model=Policy().to(device); model.load_state_dict(checkpoint['true_KKT']); model.eval(); mean,std=checkpoint['normalization']['true_KKT']
    controls,infer=predict(model,float(mean),float(std),x2,device); corrector=HDSLambdaCorrector(ode,g,gdot,(0.,UMAX),HDSLambdaConfig(grid_size=31,max_step_fraction=100.))
    rows=[]
    for i,(value,teacher_row,control) in enumerate(zip(x2,trows,controls)):
        record=problem.solve(float(value),'kkt-root')
        state=np.array([1.,value,.001,250.]); started=time.perf_counter(); safe=corrector.correct(state,control,DT); hds_time=time.perf_counter()-started
        if not safe.accepted: raise RuntimeError(f'Adaptive HDS fallback at x2={value}')
        adaptive_obj=-terminal_product(value,safe.controls,corrector); adaptive_peak=corrector.audit(state,safe.controls,DT)
        rows.append({'x2_0':value,'teacher_objective':float(teacher_row['teacher_objective']),
                     'teacher_hds_g':float(teacher_row['teacher_gmax']),
                     'teacher_seconds':float(teacher_row['teacher_seconds']),
                     'teacher_hds_iterations':float(teacher_row['teacher_hds_iterations']),
                     'teacher_solver_calls':float(teacher_row['teacher_solver_calls']),
                     'teacher_final_nodes':float(teacher_row['teacher_final_nodes']),
                     'teacher_converged':float(teacher_row['teacher_converged']),
                     'same_grid_objective':float(teacher_row['same_grid_objective']),
                     'same_grid_hds_g':float(teacher_row['same_grid_gmax']),
                     'same_grid_seconds':float(teacher_row['same_grid_seconds']),
                     'same_grid_solver_completed':float(teacher_row['same_grid_solver_completed']),
                     'reduced_nlp_objective':record['objective'],'reduced_nlp_hds_g':record['hds_max_g'],
                     'reduced_nlp_seconds':record['solve_seconds'],
                     'adaptive_objective':adaptive_obj,'adaptive_hds_g':adaptive_peak,
                     'adaptive_inference_seconds':infer,'adaptive_hds_seconds':hds_time,'adaptive_total_seconds':infer+hds_time,
                     'adaptive_corrected_segments':sum(s.corrected for s in safe.segments)})
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_point.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='x2_0'}
    report={'fixed_x2_initials':x2.tolist(),'samples':len(rows),'means':means,
            'note':('Teacher columns reproduce the original restriction/HDS/KKT termination logic with at most 50 HDS checks. '
                    'Same-grid NLP is one MATLAB SQP from the same neutral start using exactly the teacher run final nodes, '
                    'bounds, derivatives, restriction epsilon and tolerances. Reduced NLP is separately labelled because it '
                    'uses 801 fixed RK4 nodes, exact CasADi derivatives and nearest-label warm start. Adaptive uses its own HDS audit.')}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
