"""Quick sample check: does true DTW alignment beat fixed-length resampling for kNN?
Tests on a random subsample of queries (full 687x1374 pairwise DTW would be too slow to
iterate on quickly). Run from the repo root."""
import sys
import time

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from train_sentences import normalize, sort_hands  # noqa: E402

rng = np.random.default_rng(0)


def real_seqs(P, M, length):
    P, M = sort_hands(P, M)
    feats = normalize(P, M)
    out = []
    for i in range(len(P)):
        n = length[i]
        present = M[i, :n].sum(-1) > 0
        out.append(feats[i, :n][present].astype(np.float32))
    return out


def dtw_dist(a, b):
    T1, T2 = len(a), len(b)
    if T1 == 0 or T2 == 0:
        return 1e9
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)  # (T1,T2)
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, T1 + 1):
        row = cost[i - 1]
        prev = D[i - 1, :T2]
        cur = D[i, 1:]
        left = D[i, :T2]
        for j in range(1, T2 + 1):
            D[i, j] = row[j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[T1, T2] / (T1 + T2)


d = np.load('model_assets/sentences/sequences.npz', allow_pickle=True)
P, M, length, y = d['P'], d['M'], d['length'], d['y']
n = len(P)
seqs = real_seqs(P, M, length)

Pm = P.copy(); Pm[..., 0] = 1.0 - Pm[..., 0]
seqs_m = real_seqs(*sort_hands(Pm, M.copy()), length)

ref_seqs = seqs + seqs_m
ref_y = np.concatenate([y, y])
ref_src = np.concatenate([np.arange(n), np.arange(n)])

sample = rng.choice(n, 150, replace=False)
t0 = time.time()
correct1 = correct3 = correct5 = 0
for k, i in enumerate(sample):
    q = seqs[i]
    dists = []
    for j in range(len(ref_seqs)):
        if ref_src[j] == i:
            continue
        dists.append((dtw_dist(q, ref_seqs[j]), ref_y[j]))
    dists.sort(key=lambda t: t[0])
    top_classes, seen = [], set()
    for dist, c in dists:
        if c not in seen:
            seen.add(c); top_classes.append(c)
        if len(top_classes) >= 5:
            break
    if y[i] == top_classes[0]:
        correct1 += 1
    if y[i] in top_classes[:3]:
        correct3 += 1
    if y[i] in top_classes[:5]:
        correct5 += 1
    if (k + 1) % 25 == 0:
        print(f'{k+1}/150 done, {time.time()-t0:.0f}s elapsed', flush=True)

n_s = len(sample)
print(f'\nDTW sample LOO (n={n_s}): top-1 {correct1/n_s:.4f}  top-3 {correct3/n_s:.4f}  top-5 {correct5/n_s:.4f}')
print(f'total time {time.time()-t0:.0f}s -> est. full-687 time {(time.time()-t0)*687/150/60:.1f} min')
