"""
eval_knn.py
-------------
A fast, honest leave-one-out (LOO) accuracy check across ALL 687 recorded videos (not
just the 99-video single-split used to score the GRU) -- for every video, compare it
against every OTHER video (excluding itself) and predict via nearest-neighbour distance.
This is both:
  (a) a much more statistically reliable accuracy estimate (687 test folds vs 99, and no
      single arbitrary "last video" split), and
  (b) a genuinely different, training-free classifier worth comparing against the GRU --
      classical k-NN/DTW-style methods are a strong, low-overfitting baseline for exactly
      this kind of tiny-sample-per-class gesture problem.

Each sequence's REAL (hand-detected) frames are extracted (masked-out frames dropped, not
just zero-padded) and resampled to a fixed number of frames, then flattened into one
vector; L2 distance between these vectors is the metric. This also tests whether adding
augmented copies of the training data as extra "reference points" (mirror flip mainly)
helps -- a natural way to give k-NN some of the same benefit the GRU's augmentation gave it.

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/eval_knn.py
"""
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from train_sentences import normalize, sort_hands  # noqa: E402

FIXED_LEN = 24


def collapse_and_resample(P, M, length, fixed_len=FIXED_LEN):
    """(N,T,2,21,3),(N,T,2) real per-video sequences -> (N, fixed_len*128) fixed-size vectors.
    Frames with no hand detected are dropped first (not resampled through), so the vector
    represents only the actual hand-shape motion, at a common 'speed'."""
    P, M = sort_hands(P, M)
    feats_full = normalize(P, M)  # (N,T,128)
    out = np.zeros((len(P), fixed_len, 128), np.float32)
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


def main():
    d = np.load('model_assets/sentences/sequences.npz', allow_pickle=True)
    P, M, length, y = d['P'], d['M'], d['length'], d['y']
    sentences = [str(s) for s in d['sentences']]
    n = len(P)
    print('videos', n, 'classes', len(sentences))

    X = collapse_and_resample(P, M, length)

    Pm, Mm = mirror(P, M)
    Xm = collapse_and_resample(Pm, Mm, length)

    # Reference bank = every real video (as itself + its mirror), each tagged with its class
    # and its ORIGINAL video index (so a query never matches against its own reference, but
    # CAN match its own mirrored copy's twin -- that's fine, still a different vector).
    ref_X = np.concatenate([X, Xm])
    ref_y = np.concatenate([y, y])
    ref_src = np.concatenate([np.arange(n), np.arange(n)])

    def pairwise_sqdist(A, B):
        """||a-b||^2 via the matmul trick -- avoids materialising an (N,M,D) tensor."""
        aa = (A * A).sum(1)[:, None]
        bb = (B * B).sum(1)[None, :]
        return aa + bb - 2 * (A @ B.T)

    def loo_eval(query_X, k_list=(1, 3, 5)):
        d2 = pairwise_sqdist(query_X, ref_X)
        results = {k: 0 for k in k_list}
        for i in range(n):
            dist = d2[i].copy()
            dist[ref_src == i] = np.inf  # exclude self (and its mirror-of-self)
            order = np.argsort(dist)
            for k in k_list:
                top_classes = []
                seen = set()
                for j in order:
                    c = ref_y[j]
                    if c not in seen:
                        seen.add(c)
                        top_classes.append(c)
                    if len(top_classes) >= k:
                        break
                if y[i] in top_classes:
                    results[k] += 1
        return {k: v / n for k, v in results.items()}

    print('plain 1-NN classification (distance to nearest single neighbour, no voting):')
    res = loo_eval(X)
    for k, acc in res.items():
        print(f'  top-{k}: {acc:.4f}')

    print('\nwith k-NN MAJORITY vote (k=5 neighbours, class = most common among them):')
    d2 = pairwise_sqdist(X, ref_X)
    top1_vote = 0
    for i in range(n):
        dist = d2[i].copy()
        dist[ref_src == i] = np.inf
        order = np.argsort(dist)[:7]
        votes = {}
        for rank, j in enumerate(order):
            w = 1.0 / (1 + rank)
            votes[ref_y[j]] = votes.get(ref_y[j], 0) + w
        pred = max(votes, key=votes.get)
        if pred == y[i]:
            top1_vote += 1
    print(f'  top-1 (weighted vote): {top1_vote/n:.4f}')


if __name__ == '__main__':
    main()
