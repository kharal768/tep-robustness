"""Additional bound checks (integer optimum, clique centres)."""
import os
import sys
from bounds_lib import *
(Xtr, ytr, _), (Xv, yv, _), (Xh, yh, _) = load_sets(); rows=[]
for stride in (8, 4):
    for setn, Z, yy in (('validation', Xv, yv), ('heldout', Xh, yh)):
        t0 = time.time(); mac, wor, nr = solve_c(Z, yy, 0.15, stride)
        rows.append(dict(set=setn, eps=0.15, stride=stride, macro=mac, worst=wor, rows=nr, sec=time.time()-t0)); print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv('results_r6_bounds015_stride.csv', index=False)
