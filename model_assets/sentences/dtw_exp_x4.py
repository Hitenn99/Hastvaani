import sys, itertools
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import dtw_exp_x as X  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402
import pose_features as PF  # noqa: E402
X.D0 = load_pose()
res = []
for ws, wa, wl in itertools.product((1.5, 2.0, 3.0), (6, 8, 10, 14), (0, 0.5, 1)):
    a = X.run(f'desc ws{ws} wa{wa} wl{wl}', shape='desc', ws=ws, wa=wa, wl=wl, trim=0.4)
    res.append((a, ws, wa, wl))
top = sorted(res, key=lambda r: -(r[0][1] + r[0][0]))[:5]
print('BEST cross (with LOO check):')
for a, ws, wa, wl in top:
    kw = dict(shape='desc', ws=ws, wa=wa, wl=wl, trim=0.4)
    loo = score(distance_matrix(X.feats(X.D0, **kw), X.feats(X.D0, True, **kw), PF.BAND), X.D0['y'], verbose=False)
    print(a.round(3), (ws, wa, wl), 'LOO', loo.round(3))
