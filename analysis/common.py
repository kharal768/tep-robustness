"""Shared helpers: data loading, model training and attack evaluation used by the other analysis scripts."""
import os
import sys, types, time, argparse
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','core'))
sys.modules["mobo"]=types.ModuleType("mobo"); sys.modules["mobo"].run_mobo=None
import numpy as np, pandas as pd, torch
from run_pipeline import load_csv_dataset, split
from train import Hyperparams, EvalConfig, train_model
from attacks import pgd_attack
D=os.environ.get('TEP_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data'))+'/'
import os
torch.set_num_threads(int(os.environ.get('NT','1')))
EP=int(os.environ.get('EP','50'))

def load():
    X,y,rid,t,_=load_csv_dataset(D+'raw_dataset/tep_multirun_composition_data.csv')
    assert len(X)==19320 and len(set(rid))==120
    Ztr,ytr,Zv,yv=split(X,y,run_id=rid,t=t,val_frac=0.25,seed=0)
    return Ztr,ytr,Zv,yv

def front():
    d=pd.read_csv(D+'results/pareto_front_30runs_seed0.csv')
    r=d[d.clean_accuracy>0.25+1e-6].copy()
    P=r[['clean_accuracy','robust_accuracy']].values
    keep=[not any((P[j]>=P[i]).all() and (P[j]>P[i]).any() for j in range(len(P)) if j!=i) for i in range(len(P))]
    return r[keep].sort_values('clean_accuracy',ascending=False).head(8).reset_index(drop=True)

def rec(pred,y,n=4):
    return np.array([(pred[y==c]==c).mean() for c in range(n)])

class Rot(torch.nn.Module):   # model evaluated in rotated coordinates z' = Q z
    def __init__(s,m,Q): super().__init__(); s.m=m; s.register_buffer('Q',Q)
    def forward(s,zp): return s.m(zp@s.Q)   # z = Q^T z'  (row-vector form: z'@Q)
class Scaled(torch.nn.Module):
    def __init__(s,m,lo,sc): super().__init__(); s.m=m; s.lo=lo; s.sc=sc
    def forward(s,x): return s.m(x.reshape(x.shape[0],-1)*s.sc+s.lo)

def pred(m,z):
    with torch.no_grad(): return m(z).argmax(1).numpy()

def diagnostics(m,Zv,yv,seed=0,aa=True,n_aa=2500):
    m.eval(); z=torch.as_tensor(Zv,dtype=torch.float32); yt=torch.as_tensor(yv); out={}
    pc=pred(m,z); rc=rec(pc,yv); out['clean_macro']=rc.mean(); out['clean_min']=rc.min()
    pa=pred(m,pgd_attack(m,z,yt,0.15,0.03,10,'linf')); ra=rec(pa,yv)
    out['pgd_macro']=ra.mean(); out['pgd_min']=ra.min()
    ok=(pc==yv); out['rob_given_clean_macro']=np.mean([ (pa[(yv==c)&ok]==c).mean() if ((yv==c)&ok).any() else np.nan for c in range(4)])
    g=torch.Generator().manual_seed(seed)
    zn=z+(torch.rand(z.shape,generator=g)*2-1)*0.15; rn=rec(pred(m,zn),yv)
    out['unif_noise_macro']=rn.mean(); out['unif_noise_min']=rn.min()
    zs=z+(torch.randint(0,2,z.shape,generator=g)*2-1).float()*0.15; rs=rec(pred(m,zs),yv)
    out['sign_noise_macro']=rs.mean()
    # basis swap: random orthogonal rotation, l_inf PGD in the rotated coordinates
    A=np.random.default_rng(123).normal(size=(16,16)); Q=torch.as_tensor(np.linalg.qr(A)[0],dtype=torch.float32)
    mr=Rot(m,Q); zp=z@Q.T
    pb=pred(mr,pgd_attack(mr,zp,yt,0.15,0.03,10,'linf')); rb=rec(pb,yv)
    out['basisswap_pgd_macro']=rb.mean(); out['basisswap_pgd_min']=rb.min()
    for r_ in (0.15,0.60):
        p2=pred(m,pgd_attack(m,z,yt,r_,r_/5,10,'l2')); r2=rec(p2,yv)
        out[f'l2_{r_}_macro']=r2.mean(); out[f'l2_{r_}_min']=r2.min()
    pe=pred(m,pgd_attack(m,z,yt,3.0,0.6,20,'linf')); out['eps3_macro']=rec(pe,yv).mean()
    if aa:
        import torchattacks as ta
        lo=float(Zv.min())-1.0; hi=float(Zv.max())+1.0; sc=hi-lo
        sm=Scaled(m,lo,sc).eval(); idx=np.random.default_rng(0).choice(len(yv),size=min(n_aa,len(yv)),replace=False)
        x=((z[idx]-lo)/sc).reshape(-1,16,1,1); yy=yt[idx]; e=0.15/sc
        worst=pred(m,z[idx])==yv[idx]
        for nm,atk in (('apgd_ce',ta.APGD(sm,norm='Linf',eps=e,steps=50,loss='ce',n_restarts=1,seed=0)),
                       ('apgd_dlr',ta.APGD(sm,norm='Linf',eps=e,steps=50,loss='dlr',n_restarts=1,seed=0)),
                       ('square',ta.Square(sm,norm='Linf',eps=e,n_queries=1000,n_restarts=1,seed=0))):
            xa=atk(x,yy); pp=pred(sm,xa); worst&=(pp==yv[idx])
            out[f'{nm}_macro']=rec(pp,yv[idx]).mean()
        # worst case over the three attacks (per-sample), macro recall
        out['worstcase_aa_macro']=np.mean([worst[yv[idx]==c].mean() for c in range(4)])
        out['worstcase_aa_min']=np.min([worst[yv[idx]==c].mean() for c in range(4)])
    return out


def strong(m,Zv,yv,n=2500,eps=0.15,use_sq=False):
    """worst-case-over-attacks per-class recall: PGD(100 steps, step eps/20, 3 restarts), APGD-CE, APGD-DLR (+Square)"""
    import torchattacks as ta
    m.eval(); z=torch.as_tensor(Zv,dtype=torch.float32); yt=torch.as_tensor(yv)
    idx=np.random.default_rng(0).choice(len(yv),size=min(n,len(yv)),replace=False)
    ok=pred(m,z[idx])==yv[idx]; ok0=ok.copy(); per={}
    for rs in range(3):
        za=pgd_attack(m,z[idx],yt[idx],eps,eps/20,100,'linf'); p=pred(m,za)==yv[idx]; ok&=p
    per['pgd_fine']=p
    lo=float(Zv.min())-1.0; hi=float(Zv.max())+1.0; sc=hi-lo
    sm=Scaled(m,lo,sc).eval(); x=((z[idx]-lo)/sc).reshape(-1,16,1,1); e=eps/sc
    atks=[('apgd_ce',ta.APGD(sm,norm='Linf',eps=e,steps=100,loss='ce',n_restarts=1,seed=0)),('apgd_dlr',ta.APGD(sm,norm='Linf',eps=e,steps=100,loss='dlr',n_restarts=1,seed=0))]
    if use_sq: atks.append(('square',ta.Square(sm,norm='Linf',eps=e,n_queries=1000,n_restarts=1,seed=0)))
    for nm,atk in atks:
        xa=atk(x,yt[idx]); p=pred(sm,xa)==yv[idx]; ok&=p; per[nm]=p
    yy=yv[idx]
    out={'clean_pc':[ok0[yy==c].mean() for c in range(4)],'worst_pc':[ok[yy==c].mean() for c in range(4)]}
    for nm,p in per.items(): out[nm+'_macro']=float(np.mean([p[yy==c].mean() for c in range(4)]))
    out['worst_macro']=float(np.mean(out['worst_pc'])); out['worst_min']=float(np.min(out['worst_pc']))
    return out

def hp_of(r): return Hyperparams(lr=r.lr,weight_decay=r.weight_decay,hidden_dim=int(r.hidden_dim),depth=2,lambda_adv=r.lambda_adv,pgd_steps_train=int(r.pgd_steps_train))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--out',default=None); a=ap.parse_args()
    Ztr,ytr,Zv,yv=load(); F=front(); cfg=EvalConfig(eps=0.15,pgd_alpha=0.03,pgd_steps_eval=10)
    rows=[]; out=a.out or f'results_{a.stage}.csv'
    if a.stage=='E':   # epoch sweep, fresh training seeds
        for i,r in F.iterrows():
            for ep in (30,100,300):
                for s in (5000,5001,5002):
                    m,_=train_model(hp_of(r),Ztr,ytr,Zv,yv,4,cfg,epochs=ep,seed=s)
                    d=diagnostics(m,Zv,yv,aa=False); d.update(config=i,lambda_adv=r.lambda_adv,epochs=ep,seed=s,search_clean=r.clean_accuracy,search_robust=r.robust_accuracy)
                    rows.append(d); pd.DataFrame(rows).to_csv(out,index=False)
                    print(i,ep,s,round(d['clean_macro'],3),round(d['pgd_macro'],3),round(d['pgd_min'],3),flush=True)
    elif a.stage=='A':  # attack suite at the search epochs (30), fresh seeds
        for i,r in F.iterrows():
            for s in (5000,):
                m,_=train_model(hp_of(r),Ztr,ytr,Zv,yv,4,cfg,epochs=EP,seed=s)
                d=diagnostics(m,Zv,yv,aa=True); d.update(config=i,lambda_adv=r.lambda_adv,epochs=EP,seed=s)
                rows.append(d); pd.DataFrame(rows).to_csv(out,index=False); print(i,s,{k:round(v,3) for k,v in d.items() if 'macro' in k},flush=True)

    elif a.stage=='E2':   # selection-bias test: selected configs, search epochs (30), fresh seeds
        for i,r in F.iterrows():
            for s in (5000,5001,5002):
                m,_=train_model(hp_of(r),Ztr,ytr,Zv,yv,4,cfg,epochs=EP,seed=s)
                d=diagnostics(m,Zv,yv,aa=False); d.update(config=i,lambda_adv=r.lambda_adv,epochs=EP,seed=s,search_clean=r.clean_accuracy,search_robust=r.robust_accuracy)
                rows.append(d); pd.DataFrame(rows).to_csv(out,index=False); print(i,s,round(d['clean_macro'],3),round(d['pgd_macro'],3),flush=True)
    elif a.stage=='F':    # epoch sweep on the selected wide model (width 384) and at width 128
        b=pd.read_csv(D+'results/pareto_front_minmetric_bigcapacity_seed0.csv').sort_values('robust_accuracy',ascending=False).iloc[0]
        for hd,eps_list in ((384,(50,100,200)),(128,(50,100))):
            hp=Hyperparams(lr=b.lr,weight_decay=b.weight_decay,hidden_dim=hd,depth=2,lambda_adv=b.lambda_adv,pgd_steps_train=int(b.pgd_steps_train))
            for ep in eps_list:
                for s in (5000,5001):
                    m,_=train_model(hp,Ztr,ytr,Zv,yv,4,cfg,epochs=ep,seed=s)
                    d=diagnostics(m,Zv,yv,aa=False); st=strong(m,Zv,yv,n=2000); d.update(hidden=hd,epochs=ep,seed=s,strong_min=st['worst_min'],strong_macro=st['worst_macro'])
                    rows.append(d); pd.DataFrame(rows).to_csv(out,index=False); print(hd,ep,s,round(d['clean_min'],3),round(d['pgd_min'],3),flush=True)

    elif a.stage=='G':   # unbounded-eps sanity check: robust accuracy as eps grows, strong PGD + APGD
        import torchattacks as ta
        for i,r in F.iterrows():
            m,_=train_model(hp_of(r),Ztr,ytr,Zv,yv,4,cfg,epochs=EP,seed=5000); m.eval()
            z=torch.as_tensor(Zv,dtype=torch.float32); yt=torch.as_tensor(yv)
            idx=np.random.default_rng(0).choice(len(yv),size=3000,replace=False)
            for eps in (0.15,0.3,1.0,3.0,10.0):
                ok=pred(m,z[idx])==yv[idx]
                for rs in range(5):
                    za=pgd_attack(m,z[idx],yt[idx],eps,eps/20,100,'linf'); ok&=(pred(m,za)==yv[idx])
                rp=[ok[yv[idx]==c].mean() for c in range(4)]
                lo=float(Zv.min())-eps-1.0; hi=float(Zv.max())+eps+1.0; sc=hi-lo
                sm=Scaled(m,lo,sc).eval(); x=((z[idx]-lo)/sc).reshape(-1,16,1,1)
                xa=ta.APGD(sm,norm='Linf',eps=eps/sc,steps=100,loss='ce',n_restarts=2,seed=0)(x,yt[idx])
                ra=rec(pred(sm,xa),yv[idx])
                rows.append(dict(config=i,lambda_adv=r.lambda_adv,eps=eps,pgd100x5_macro=float(np.mean(rp)),pgd100x5_min=float(np.min(rp)),apgd_macro=float(ra.mean()),apgd_min=float(ra.min())))
                pd.DataFrame(rows).to_csv(out,index=False); print(rows[-1],flush=True)

    elif a.stage=='H':   # selected wide model (width 384, 100 epochs): standard vs strong attacks, fresh seeds
        b=pd.read_csv(D+'results/pareto_front_minmetric_bigcapacity_seed0.csv').sort_values('robust_accuracy',ascending=False).iloc[0]
        hp=Hyperparams(lr=b.lr,weight_decay=b.weight_decay,hidden_dim=int(b.hidden_dim),depth=2,lambda_adv=b.lambda_adv,pgd_steps_train=int(b.pgd_steps_train))
        for s in (5000,5001,5002):
            m,_=train_model(hp,Ztr,ytr,Zv,yv,4,cfg,epochs=100,seed=s)
            d=diagnostics(m,Zv,yv,aa=False); st=strong(m,Zv,yv,n=2500,use_sq=True)
            d.update(seed=s,**{k:v for k,v in st.items() if not k.endswith('_pc')},worst_pc=str(np.round(st['worst_pc'],3).tolist()),clean_pc=str(np.round(st['clean_pc'],3).tolist()))
            rows.append(d); pd.DataFrame(rows).to_csv(out,index=False); print(s,round(d['clean_min'],3),round(d['pgd_min'],3),round(st['worst_min'],3),flush=True)

    elif a.stage=='K':   # smaller budgets: PGD-10 vs strong attacks; standard vs adversarial training at matching eps
        def ev(m,eps,tag,s,lam):
            m.eval(); z=torch.as_tensor(Zv,dtype=torch.float32); yt=torch.as_tensor(yv)
            pc=rec(pred(m,z),yv); pa=rec(pred(m,pgd_attack(m,z,yt,eps,eps/5,10,'linf')),yv)
            g=torch.Generator().manual_seed(0); pn=rec(pred(m,z+(torch.rand(z.shape,generator=g)*2-1)*eps),yv)
            st=strong(m,Zv,yv,n=2500,eps=eps)
            rows.append(dict(model=tag,train_lambda=lam,eps=eps,seed=s,clean_macro=pc.mean(),clean_min=pc.min(),pgd10_macro=pa.mean(),pgd10_min=pa.min(),noise_macro=pn.mean(),strong_macro=st['worst_macro'],strong_min=st['worst_min']))
            pd.DataFrame(rows).to_csv(out,index=False); print(rows[-1],flush=True)
        for s in (5000,5001):
            m,_=train_model(hp_of(F.iloc[1]),Ztr,ytr,Zv,yv,4,cfg,epochs=EP,seed=s)
            for eps in (0.005,0.01,0.02,0.05): ev(m,eps,'standard',s,0.0)
            for eps in (0.01,0.02,0.05):
                c2=EvalConfig(eps=eps,pgd_alpha=eps/5,pgd_steps_eval=10)
                m2,_=train_model(hp_of(F.iloc[3]),Ztr,ytr,Zv,yv,4,c2,epochs=EP,seed=s)
                ev(m2,eps,'adversarial',s,float(F.iloc[3].lambda_adv))

    elif a.stage=='F2':   # epoch sweep at baseline-capacity width 128 (same other hyperparameters as the selected wide model)
        b=pd.read_csv(D+'results/pareto_front_minmetric_bigcapacity_seed0.csv').sort_values('robust_accuracy',ascending=False).iloc[0]
        hp=Hyperparams(lr=b.lr,weight_decay=b.weight_decay,hidden_dim=128,depth=2,lambda_adv=b.lambda_adv,pgd_steps_train=int(b.pgd_steps_train))
        for ep in (50,100):
            for s in (5000,5001):
                m,_=train_model(hp,Ztr,ytr,Zv,yv,4,cfg,epochs=ep,seed=s)
                d=diagnostics(m,Zv,yv,aa=False); st=strong(m,Zv,yv,n=2000); d.update(hidden=128,epochs=ep,seed=s,strong_min=st['worst_min'],strong_macro=st['worst_macro'])
                rows.append(d); pd.DataFrame(rows).to_csv(out,index=False); print(128,ep,s,round(d['clean_min'],3),round(d['pgd_min'],3),flush=True)
    elif a.stage=='B':  # sklearn baselines (clean ceiling)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc=StandardScaler().fit(Ztr)
        for nm,mk in (('rf',lambda:RandomForestClassifier(300,n_jobs=4,random_state=0)),('knn15',lambda:KNeighborsClassifier(15)),('logreg',lambda:LogisticRegression(max_iter=2000))):
            mdl=mk().fit(sc.transform(Ztr),ytr); p=mdl.predict(sc.transform(Zv)); r=rec(p,yv)
            rng=np.random.default_rng(0); pn=mdl.predict(sc.transform(Zv+rng.uniform(-.15,.15,Zv.shape))); rn=rec(pn,yv)
            rows.append(dict(model=nm,clean_macro=r.mean(),clean_min=r.min(),**{f'clean_c{c}':r[c] for c in range(4)},noise_macro=rn.mean(),noise_min=rn.min()))
            print(rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(out,index=False)
