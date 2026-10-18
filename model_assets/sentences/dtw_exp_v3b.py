"""Metric + late-fusion experiments (v3). Run from the repo root."""
import sys, time, itertools
import numba as nb
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_lab import pack, score  # noqa: E402


@nb.njit(cache=True, fastmath=True)
def dtw_m(a, b, band, metric):
    T1, T2 = a.shape[0], b.shape[0]; F = a.shape[1]
    prev = np.full(T2 + 1, np.inf); cur = np.full(T2 + 1, np.inf); prev[0] = 0.0
    for i in range(1, T1 + 1):
        cur[:] = np.inf
        centre = i * T2 / T1
        lo = max(1, int(centre - band * T2)); hi = min(T2, int(centre + band * T2) + 1)
        for j in range(lo, hi + 1):
            c = 0.0
            for k in range(F):
                d = a[i - 1, k] - b[j - 1, k]
                c += d * d if metric != 2 else abs(d)
            if metric == 1: c = np.sqrt(c)
            m = min(prev[j - 1], prev[j], cur[j - 1])
            cur[j] = c + m
        prev, cur = cur, prev
    return prev[T2] / (T1 + T2)


@nb.njit(parallel=True, cache=True)
def all_d(fq, oq, lq, fr, orr, lr, band, metric):
    out = np.empty((len(oq), len(orr)))
    for i in nb.prange(len(oq)):
        a = fq[oq[i]:oq[i] + lq[i]]
        for j in range(len(orr)):
            out[i, j] = dtw_m(a, fr[orr[j]:orr[j] + lr[j]], band, metric)
    return out


def dm(s, sm, metric, cols=None):
    if cols is not None:
        s = [x[:, cols] for x in s]; sm = [x[:, cols] for x in sm]
    return all_d(*pack(s), *pack(s + sm), PF.BAND, metric)


if __name__ == '__main__':
    d = load_pose(); y = d['y']
    s = PF.dataset_features(d); sm = PF.dataset_features(d, mirror=True)
    for metric, name in ((0, 'sq'), (1, 'euclid'), (2, 'L1')):
        t = time.time(); D = dm(s, sm, metric); print(f'{name} ({time.time()-t:.0f}s)', end=' '); score(D, y)
    shape = list(range(0, 128)); loc = list(range(128, 132)); arm = list(range(132, 140))
    for metric in (0, 1):
        Ds = {g: dm(s, sm, metric, c) for g, c in (('shape', shape), ('loc', loc + arm), ('arm', arm))}
        for g, D in Ds.items():
            print(f'metric {metric} only {g}', end=' '); score(D, y)
        # normalise each channel by its median distance, then weighted sum
        Z = {g: D / np.median(D) for g, D in Ds.items()}
        best = None
        for ws, wl in itertools.product((0.5, 1, 2), (0.5, 1, 2)):
            r = score(ws * Z['shape'] + wl * Z['loc'], y, verbose=False)
            print(f'metric {metric} fuse shape {ws} loc+arm {wl}: {r.round(4)}')
