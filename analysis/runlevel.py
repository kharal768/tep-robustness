"""Run-level aggregation of per-sample results."""
import os
import numpy as np, pandas as pd
from scipy.stats import binomtest
rows=[]
for budget in ['raw002','sd025','sd01']:
    def per_run(tag):
        acc={}
        for s in range(5000,5005):
            z=np.load(f'persample/{budget}_{tag}_{s}.npz',allow_pickle=True); ok,y,rid=z['ok'],z['y'],z['rid']
            for c in range(4):
                for r in np.unique(rid[y==c]):
                    m=(y==c)&(rid==r); acc.setdefault((c,r),[]).append(ok[m].mean())
        return {k:np.mean(v) for k,v in acc.items()}
    S=per_run('standard')
    for tag in ['adv_pgd4','adv_pgd10','tuned']:
        try: A=per_run(tag)
        except FileNotFoundError: continue
        for c in range(4):
            ks=[k for k in S if k[0]==c]; d=np.array([A[k]-S[k] for k in ks]); pos=(d>0).sum(); neg=(d<0).sum()
            p=binomtest(pos,pos+neg).pvalue if pos+neg>0 else 1.0
            rows.append(dict(budget=budget,model=tag,cls=c,runs=len(ks),std=np.mean([S[k] for k in ks]),adv=np.mean([A[k] for k in ks]),diff=d.mean(),pos=pos,neg=neg,tie=len(ks)-pos-neg,p=p))
R=pd.DataFrame(rows)
# Holm within each budget across classes x models
R['p_holm']=np.nan
for b,g in R.groupby('budget'):
    o=g.p.sort_values().index; m=len(o); prev=0
    for i,ix in enumerate(o):
        v=min(1,(m-i)*R.loc[ix,'p']); prev=max(prev,v); R.loc[ix,'p_holm']=prev
R.to_csv('results_r5_runlevel.csv',index=False); print(R.round(4).to_string())
