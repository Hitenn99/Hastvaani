"""
dtw_lab.py
------------
Fast (numba, parallel) leave-one-out harness for trying sentence-feature / DTW variants.
Same protocol as eval_dtw_full.py: each of the 687 videos is queried against every OTHER
video (and every other video's mirrored copy); its own original + mirror are excluded.

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/dtw_lab.py
"""
import sys
import time

import numba as nb
import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from train_sentences import normalize, sort_hands  # noqa: E402


@nb.njit(cache=True, fastmath=True)
def dtw_nb(a, b, band):
    T1, T2 = a.shape[0], b.shape[0]
    F = a.shape[1]
    prev = np.full(T2 + 1, np.inf)
    cur = np.full(T2 + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, T1 + 1):
        cur[:] = np.inf
        # Sakoe-Chiba band on the *relative* diagonal (band<=0 disables it)
        if band > 0:
            centre = i * T2 / T1
            lo = max(1, int(centre - band * T2))
            hi = min(T2, int(centre + band * T2) + 1)
        else:
            lo, hi = 1, T2
        for j in range(lo, hi + 1):
            c = 0.0
            for k in range(F):
                d = a[i - 1, k] - b[j - 1, k]
                c += d * d
            m = prev[j - 1]
            if prev[j] < m:
                m = prev[j]
            if cur[j - 1] < m:
                m = cur[j - 1]
            cur[j] = c + m
        prev, cur = cur, prev
    return prev[T2] / (T1 + T2)


@nb.njit(parallel=True, cache=True)
def all_dists(flat_q, off_q, len_q, flat_r, off_r, len_r, band):
    nq, nr = len(off_q), len(off_r)
    out = np.empty((nq, nr))
    for i in nb.prange(nq):
        a = flat_q[off_q[i]:off_q[i] + len_q[i]]
        for j in range(nr):
            b = flat_r[off_r[j]:off_r[j] + len_r[j]]
            out[i, j] = dtw_nb(a, b, band)
    return out


def pack(seqs):
    lens = np.array([len(s) for s in seqs], np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    return np.concatenate(seqs).astype(np.float64), offs, lens


def load():
    d = np.load('model_assets/sentences/sequences.npz', allow_pickle=True)
    return d['P'], d['M'], d['length'], d['y'], [str(s) for s in d['sentences']]


def distance_matrix(seqs, seqs_m, band=0.0):
    fq, oq, lq = pack(seqs)
    fr, orr, lr = pack(seqs + seqs_m)
    return all_dists(fq, oq, lq, fr, orr, lr, band)


def score(D, y, agg='min', k=2, verbose=True):
    """D: (n, 2n) query x [refs, mirrored refs]. Returns top1/3/5 (LOO, own clip excluded)."""
    n = len(y)
    ref_y = np.concatenate([y, y])
    C = y.max() + 1
    hits = np.zeros(3)
    for i in range(n):
        d = D[i].copy()
        d[i] = np.inf
        d[n + i] = np.inf
        cls_score = np.full(C, np.inf)
        for c in range(C):
            dc = np.sort(d[ref_y == c])
            dc = dc[np.isfinite(dc)]
            if len(dc) == 0:
                continue
            if agg == 'min':
                cls_score[c] = dc[0]
            else:  # mean of k best
                cls_score[c] = dc[:k].mean()
        order = np.argsort(cls_score)
        for h, t in enumerate((1, 3, 5)):
            hits[h] += y[i] in order[:t]
    r = hits / n
    if verbose:
        print(f'   top-1 {r[0]:.4f}  top-3 {r[1]:.4f}  top-5 {r[2]:.4f}', flush=True)
    return r


def mirror(P):
    Pm = P.copy()
    Pm[..., 0] = 1.0 - Pm[..., 0]
    return Pm


def real_frames(feats, M, length):
    out = []
    for i in range(len(feats)):
        n = length[i]
        present = M[i, :n].sum(-1) > 0
        out.append(feats[i, :n][present].astype(np.float64))
    return out


def baseline_feats(P, M, length):
    P, M = sort_hands(P, M)
    return real_frames(normalize(P, M), M, length)


if __name__ == '__main__':
    P, M, length, y, sentences = load()
    t = time.time()
    seqs = baseline_feats(P, M, length)
    seqs_m = baseline_feats(mirror(P), M.copy(), length)
    D = distance_matrix(seqs, seqs_m)
    print(f'baseline ({time.time()-t:.0f}s)')
    score(D, y)
