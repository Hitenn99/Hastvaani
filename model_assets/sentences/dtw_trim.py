"""Trim idle (hands-down) lead-in/lead-out before DTW. Run from the repo root."""
import sys
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import dtw_exp_pose as X  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402

D0 = X.load_pose()
# how high are the wrists (pose), in shoulder widths below the shoulder line?
ys = []
for i in range(len(D0['y'])):
    n = int(D0['length'][i]); px = np.array(D0['wh'][i])
    pose = D0['pose'][i, :n, :, :2] * px
    o, s = X.body_frame(np.concatenate([pose, D0['pose'][i, :n, :, 2:]], -1), D0['pm'][i, :n])
    wy = (pose[:, [15, 16], 1] - o[1]) / s
    ys.append(wy.min(1))
allv = np.concatenate(ys)
print('min-wrist-y percentiles 5/25/50/75/95:', np.percentile(allv, [5, 25, 50, 75, 95]).round(2))
print('first-frame median', np.median([y[0] for y in ys]).round(2), 'last-frame median', np.median([y[-1] for y in ys]).round(2))

orig_feats = X.feats
def trimmed(d, thr, margin=1, **kw):
    # compute per-clip active window, then cut raw arrays to it
    d2 = {k: v.copy() if hasattr(v, 'copy') else v for k, v in d.items()}
    for i in range(len(d['y'])):
        n = int(d['length'][i]); px = np.array(d['wh'][i])
        pose = d['pose'][i, :n, :, :2] * px
        o, s = X.body_frame(np.concatenate([pose, d['pose'][i, :n, :, 2:]], -1), d['pm'][i, :n])
        act = np.where(((pose[:, [15, 16], 1] - o[1]) / s).min(1) < thr)[0]
        if len(act) < 4:
            continue
        a, b = max(0, act[0] - margin), min(n, act[-1] + 1 + margin)
        for k in ('P', 'M', 'pose', 'pm'):
            arr = d[k][i].copy(); seg = arr[a:b].copy(); arr[:] = 0; arr[:b - a] = seg
            d2[k][i] = arr
        d2['length'][i] = b - a
    return orig_feats(d2, **kw)

MIRR = X.mirror_all(D0)
best = dict(w_hand=1.0, w_loc=1, w_arm=2, frames='hands')
for thr in (99, 2.5, 2.0, 1.5, 1.0):
    for fr in ('hands', 'any'):
        kw = dict(best, frames=fr)
        s, _ = trimmed(D0, thr, **kw); sm, _ = trimmed(MIRR, thr, **kw)
        r = score(distance_matrix(s, sm, 0.2), D0['y'], verbose=False)
        print(f'thr {thr} frames={fr}:', r.round(4), flush=True)
