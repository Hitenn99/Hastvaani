"""Baseline-feature variants for sentence DTW (see dtw_lab.py). Run from the repo root."""
import sys
import time

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import distance_matrix, load, mirror, score  # noqa: E402
from train_sentences import normalize, sort_hands  # noqa: E402


def feats(P, M, length, w_z=1.0, w_pos=0.0, w_pres=1.0, w_scale=0.0):
    P, M = sort_hands(P, M)
    F = normalize(P, M).astype(np.float64)  # (N,T,128)
    zmask = np.ones(128)
    zmask[2:126:3] = w_z
    zmask[126:] = w_pres
    F = F * zmask
    out = []
    for i in range(len(P)):
        n = length[i]
        m = M[i, :n]
        keep = m.sum(-1) > 0
        f = F[i, :n][keep]
        p, mm = P[i, :n][keep], m[keep]
        extra = []
        if w_pos or w_scale:
            # joint hand centroid in image coords, relative to the clip's mean, / clip hand size
            cnt = mm.sum(-1) * 21
            c = (p[..., :2] * mm[:, :, None, None]).sum((1, 2)) / cnt[:, None]
            size = np.linalg.norm(p[:, :, 9, :2] - p[:, :, 0, :2], axis=-1)
            S = max(np.median(size[mm > 0]), 1e-4)
            if w_pos:
                extra.append(w_pos * (c - c.mean(0)) / S)
            if w_scale:
                # the per-frame spread of the hands (what normalize() divides out), relative
                d = np.linalg.norm((p[..., :2] - c[:, None, None, :]), axis=-1) * mm[:, :, None]
                spread = d.max((1, 2)) / S
                extra.append(w_scale * np.log(np.maximum(spread, 1e-3))[:, None])
        out.append(np.concatenate([f] + extra, -1) if extra else f)
    return out


def run(name, band=0.0, aggs=(('min', 1),), **kw):
    P, M, length, y, _ = DATA
    t = time.time()
    s = feats(P, M, length, **kw)
    sm = feats(mirror(P), M.copy(), length, **kw)
    D = distance_matrix(s, sm, band)
    print(f'{name} ({time.time()-t:.0f}s)', flush=True)
    for agg, k in aggs:
        if len(aggs) > 1:
            print(f'  agg={agg} k={k}', end='')
        score(D, y, agg, k)
    return D


if __name__ == '__main__':
    DATA = load()
    run('pos 0.25', w_pos=0.25)
    run('pos 0.5', w_pos=0.5)
    run('pos 1', w_pos=1)
    run('scale 0.5', w_scale=0.5)
    run('band 0.2', band=0.2)
    run('band 0.35', band=0.35)

