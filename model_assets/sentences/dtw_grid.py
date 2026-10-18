"""Weight grid for the pose-aware DTW features (dtw_exp_pose.py). Run from the repo root."""
import itertools
import sys
sys.path.insert(0, 'model_assets/sentences')
import dtw_exp_pose as X  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402

X.DATA = X.load_pose(); X.MIRR = X.mirror_all(X.DATA)
res = []
for wh, wl, wa, fr in itertools.product((0.25, 0.5, 1.0), (1, 2, 3), (0, 1, 2), ('hands', 'any')):
    kw = dict(w_hand=wh, w_loc=wl, w_arm=wa, frames=fr)
    s, _ = X.feats(X.DATA, **kw); sm, _ = X.feats(X.MIRR, **kw)
    r = score(distance_matrix(s, sm, 0.2), X.DATA['y'], verbose=False)
    res.append((r, kw)); print(kw, r.round(4), flush=True)
print('BEST by top-1:'); [print(r.round(4), kw) for r, kw in sorted(res, key=lambda t: -t[0][0])[:5]]
print('BEST by top-5:'); [print(r.round(4), kw) for r, kw in sorted(res, key=lambda t: -t[0][2])[:5]]
