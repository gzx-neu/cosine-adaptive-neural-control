"""Render paper figures from the frozen final experiment summaries."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'论文写作'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
def load(path): return json.loads(path.read_text(encoding='utf-8'))
vdp_sampling=load(ROOT/'kkt_collocation'/'results'/'final_sampling_vs_hds_vdp_3seeds'/'summary.json')['comparison']['audits']
pen_sampling=load(ROOT/'kkt_collocation'/'results'/'final_sampling_vs_hds_penicillin_3seeds'/'summary.json')['comparison']['audits']
labels=['ZOH endpoints','10 samples/segment','100 samples/segment']
keys=['zoh_endpoints','uniform_10','uniform_100']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(6.4,3.6)); x=np.arange(3); w=.34
ax.bar(x-w/2,[100*vdp_sampling[k]['false_safe_rate_vs_hds'] for k in keys],w,label='VDP')
ax.bar(x+w/2,[100*pen_sampling[k]['false_safe_rate_vs_hds'] for k in keys],w,label='Penicillin')
ax.set_ylabel('False-safe rate relative to HDS (%)'); ax.set_xticks(x,labels,rotation=12,ha='right'); ax.set_ylim(0,100); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'hds_sampling_false_safe_rate.pdf'); fig.savefig(OUT/'hds_sampling_false_safe_rate.png',dpi=300); plt.close(fig)
vdp=load(ROOT/'kkt_collocation'/'results'/'final_multiseed_vdp900_penalty_aggregate'/'summary.json')['methods']
pen=load(ROOT/'kkt_collocation'/'results'/'final_multiseed_penicillin400_penalty_aggregate'/'summary.json')['methods']
vnames=['Never-KKT + HDS-lambda','Constraint-penalty: S+P + HDS-lambda','Always-KKT + HDS-lambda']
pnames=['Never-KKT + HDS-lambda','Constraint-penalty: S+P + HDS-lambda','Always-KKT + HDS-lambda']
labels=['S','S+Penalty','S+KKT']; xv=np.arange(3)
fig,ax=plt.subplots(figsize=(6.4,3.6));
ax.bar(xv-.18,[vdp[k]['mean_corrected_segments']['mean'] for k in vnames],.36,yerr=[vdp[k]['mean_corrected_segments']['sample_std'] for k in vnames],capsize=3,label='VDP')
ax.bar(xv+.18,[pen[k]['mean_corrected_segments']['mean'] for k in pnames],.36,yerr=[pen[k]['mean_corrected_segments']['sample_std'] for k in pnames],capsize=3,label='Penicillin')
ax.set_ylabel('Mean corrected ZOH segments'); ax.set_xticks(xv,labels); ax.legend(frameon=False); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'correction_burden_ablation.pdf'); fig.savefig(OUT/'correction_burden_ablation.png',dpi=300); plt.close(fig)
print(OUT)
