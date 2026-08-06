"""Post-process frozen ablation logs against matched cold-start references."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'kkt_collocation/results/three_benchmark_corrected_gap_evaluation'

def rows(path): return list(csv.DictReader(path.open(encoding='utf-8')))
def num(x): return float(x)
def write(name, records):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def statistics(records):
    ans=[]
    for method in sorted(set(r['method'] for r in records)):
        a=[r for r in records if r['method']==method]; ok=[r for r in a if r['accepted_network']=='True']
        v=np.array([num(r['hds_corrected_relative_gap_percent']) for r in ok]); n=np.array([num(r['nominal_relative_gap_percent']) for r in ok]);
        ans.append({'Method':method,'Accepted / Fallback':f'{len(ok)} / {len(a)-len(ok)}','Nominal gap (%)':np.mean(n) if len(n) else np.nan,'HDS-corrected gap (%)':np.mean(v) if len(v) else np.nan,'Median':np.median(v) if len(v) else np.nan,'p95':np.percentile(v,95) if len(v) else np.nan,'Mean corrected segments':np.mean([num(r['corrected_segments']) for r in ok]) if ok else np.nan,'t_VALC':np.mean([num(r['total_deployment_seconds']) for r in ok]) if ok else np.nan,'t_cold NLP':np.mean([num(r['cold_reference_seconds']) for r in ok]) if ok else np.nan,'Speedup':np.mean([num(r['cold_reference_seconds'])/num(r['total_deployment_seconds']) for r in ok]) if ok else np.nan})
    return ans
def gap_records(log, references, cold_seconds, benchmark):
    out=[]
    for r in log:
        i=int(float(r.get('sample_index',r.get('index')))); ref=references[i]; accepted=str(r['accepted'])=='True'
        nom=num(r['nominal_objective']); app=num(r['applied_objective']) if accepted else np.nan; den=max(abs(ref),1e-12)
        out.append({'benchmark':benchmark,'method':r['method'],'sample_index':i,'reference_objective_minimization':ref,'cold_reference_seconds':cold_seconds[i],'reference_status':'success','accepted_network':str(accepted),'fallback':str(not accepted),'nominal_objective_minimization':nom,'hds_objective_minimization':app,'nominal_relative_gap_percent':100*(nom-ref)/den,'hds_corrected_relative_gap_percent':100*(app-ref)/den if accepted else np.nan,'corrected_segments':r['corrected_segments'],'total_deployment_seconds':r.get('total_seconds',r.get('total_deployment_seconds'))})
    return out
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 vdir=ROOT/'kkt_collocation/results/two_stage_vs_kkt_only_single/vdp_seed20260751'; pdir=ROOT/'kkt_collocation/results/two_stage_vs_kkt_only_single/penicillin_seed20260761'; jdir=ROOT/'kkt_collocation/results/jcb2d_900_two_stage_vs_kkt_only_single'
 vr=rows(ROOT/'kkt_collocation/results/final_vdp_nlp_reference_50/per_sample.csv'); vi=np.load(ROOT/'kkt_collocation/results/final_vdp_nlp_reference_50/reference_test_indices.npy'); vref={int(i):num(r['nlp_reference_objective']) for i,r in zip(vi,vr)}; vt=json.loads((ROOT/'kkt_collocation/results/final_vdp_nlp_reference_50/summary.json').read_text(encoding='utf-8'))['nlp_reference']['mean_solve_seconds']; vtime={int(i):vt for i in vi}
 pr=rows(ROOT/'kkt_collocation/results/final_penicillin_nlp_reference_50/per_sample.csv'); pi=np.load(ROOT/'kkt_collocation/results/final_penicillin_nlp_reference_50/reference_test_indices.npy'); pref={int(i):num(r['nlp_reference_objective']) for i,r in zip(pi,pr)}; pt=json.loads((ROOT/'kkt_collocation/results/final_penicillin_nlp_reference_50/summary.json').read_text(encoding='utf-8'))['nlp_reference']['mean_solve_seconds']; ptime={int(i):pt for i in pi}
 jl=sum((rows(f) for f in jdir.glob('test_sample_log_*_part_*.csv')),[]); d=loadmat(jdir/'coldstart_jiangfu_references.mat'); assert np.array_equal(np.load(jdir/'test_states.npy'),d['testStates']); jref=d['references'][:,0]; jtime=d['references'][:,1]
 V=gap_records([r for r in rows(vdir/'test_sample_log.csv') if int(float(r['sample_index'])) in vref],vref,vtime,'VDP'); P=gap_records([r for r in rows(pdir/'test_sample_log.csv') if int(float(r['sample_index'])) in pref],pref,ptime,'Penicillin'); J=gap_records(jl,{i:float(x) for i,x in enumerate(jref)},{i:float(x) for i,x in enumerate(jtime)},'JCB-2D')
 write('vdp_reference_and_gap.csv',V);write('penicillin_reference_and_gap.csv',P);write('jcb_reference_and_gap.csv',J)
 table=[]
 for b,x in [('VDP',V),('Penicillin',P),('JCB-2D',J)]:
  for r in statistics(x):table.append({'Benchmark':b,**r})
 write('summary_table.csv',table); report={'coverage_note':'VDP and Penicillin use their existing fixed 50-point matched cold-start reference subsets; JCB uses all frozen 100 test points. No warm-start teacher label was used as a test reference.','tables':table};(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'README.md').write_text('Objectives are minimization objectives. HDS gaps are under the declared model and numerical settings.\n',encoding='utf-8')
if __name__=='__main__':main()
