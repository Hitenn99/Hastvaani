"""
eval_knn2.py -- sweeps over eval_knn.py's design choices to find what actually helps,
using the same full leave-one-out (687-video) protocol for a reliable comparison.
Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/eval_knn2.py
"""
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from train_sentences import normalize, sort_hands  # noqa: E402

rng = np.random.default_rng(7)


def collapse_and_resample(P, M, length, fixed_len, drop_z=False):
    P, M = sort_hands(P, M)
    feats_full = normalize(P, M)
    if drop_z:
        # zero every 3rd coord (x,y,z per point) -- keeps vector length/layout, just mutes z
        feats_full = feats_full.copy()
        feats_full[..., 2:126:3] = 0.0
    out = np.zeros((len(P), fixed_len, feats_full.shape[-1]), np.float32)
    for i in range(len(P)):
        n = length[i]
        present = M[i, :n].sum(-1) > 0
        real = feats_full[i, :n][present]
        if len(real) == 0:
            continue
        idx = np.linspace(0, len(real) - 1, fixed_len)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, len(real) - 1)
        frac = (idx - lo)[:, None]
        out[i] = real[lo] * (1 - frac) + real[hi] * frac
    return out.reshape(len(P), -1)


def mirror(P, M):
    Pm = P.copy()
    Pm[..., 0] = 1.0 - Pm[..., 0]
    return sort_hands(Pm, M.copy())


def small_rotate(P, M, sigma=0.1):
    P2, M2 = sort_hands(P.copy(), M.copy())
    present = M2.sum(-1) > 0
    cnt = np.maximum(M2.sum(-1), 1e-6)[..., None] * 21.0
    c = (P2[..., :2] * M2[..., None, None]).sum((-3, -2)) / cnt
    th = rng.normal(0, sigma, len(P2))
    R = np.stack([np.cos(th), -np.sin(th), np.sin(th), np.cos(th)], 1).reshape(-1, 2, 2)
    xy = P2[..., :2] - c[:, :, None, None, :]
    xy = np.einsum('nij,nthpj->nthpi', R, xy)
    P2[..., :2] = xy + c[:, :, None, None, :]
    P2 *= M2[..., None, None]
    return P2, M2


def pairwise_sqdist(A, B):
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    return aa + bb - 2 * (A @ B.T)


def loo_topk(query_X, ref_X, ref_y, ref_src, y, n, k_list=(1, 3, 5)):
    d2 = pairwise_sqdist(query_X, ref_X)
    results = {k: 0 for k in k_list}
    for i in range(n):
        dist = d2[i].copy()
        dist[ref_src == i] = np.inf
        order = np.argsort(dist)
        top_classes, seen = [], set()
        for j in order:
            c = ref_y[j]
            if c not in seen:
                seen.add(c)
                top_classes.append(c)
            if len(top_classes) >= max(k_list):
                break
        for k in k_list:
            if y[i] in top_classes[:k]:
                results[k] += 1
    return {k: v / n for k, v in results.items()}


def run(name, fixed_len, drop_z, extra_rotations):
    d = np.load('model_assets/sentences/sequences.npz', allow_pickle=True)
    P, M, length, y = d['P'], d['M'], d['length'], d['y']
    n = len(P)

    X = collapse_and_resample(P, M, length, fixed_len, drop_z)
    Pm, Mm = mirror(P, M)
    Xm = collapse_and_resample(Pm, Mm, length, fixed_len, drop_z)

    ref_Xs = [X, Xm]
    ref_ys = [y, y]
    ref_srcs = [np.arange(n), np.arange(n)]
    for _ in range(extra_rotations):
        Pr, Mr = small_rotate(P, M)
        Xr = collapse_and_resample(Pr, Mr, length, fixed_len, drop_z)
        ref_Xs.append(Xr)
        ref_ys.append(y)
        ref_srcs.append(np.arange(n))

    ref_X = np.concatenate(ref_Xs)
    ref_y = np.concatenate(ref_ys)
    ref_src = np.concatenate(ref_srcs)

    res = loo_topk(X, ref_X, ref_y, ref_src, y, n)
    print(f'{name:45s} top-1 {res[1]:.4f}  top-3 {res[3]:.4f}  top-5 {res[5]:.4f}')


run('baseline (len24, xyz, mirror only)', 24, False, 0)
run('len16', 16, False, 0)
run('len32', 32, False, 0)
run('len40', 40, False, 0)
run('len24, drop z', 24, True, 0)
run('len24, +4 rotation copies', 24, False, 4)
run('len32, +4 rotation copies', 32, False, 4)
run('len32, drop z, +4 rotation copies', 32, True, 4)
