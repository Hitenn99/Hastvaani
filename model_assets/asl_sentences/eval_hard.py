"""Harder (still synthetic) ASL check: queries from held-out signers with each word's first/last
~20% cut (co-articulation eats word edges), random per-word speed 0.7-1.4x and landmark jitter.
Run from the repo root after extract_words.py."""
import sys
import numpy as np
sys.path.insert(0, 'model_assets/asl_sentences'); sys.path.insert(0, 'model_assets/sentences')
import build_bank as B  # noqa: E402
from dtw_lab import all_dists, pack  # noqa: E402
from pose_features import BAND  # noqa: E402


def distort(seq, rng):
    n = len(seq)
    cut = int(round(n * rng.uniform(0.1, 0.25)))
    seq = seq[cut // 2: n - (cut - cut // 2)] if n - cut >= 3 else seq
    m = max(3, int(round(len(seq) * rng.uniform(0.7, 1.4))))
    idx = np.clip(np.round(np.linspace(0, len(seq) - 1, m)).astype(int), 0, len(seq) - 1)
    return seq[idx] + rng.normal(0, 0.03, (m, seq.shape[1])) * (seq[idx] != 0)


words = B.load_words()
signers = sorted({c[0] for cl in words.values() for c in cl})
res = []
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    test = set(rng.choice(signers, int(len(signers) * 0.3), replace=False).tolist())
    refs, ry, qs, qy = [], [], [], []
    for ci, e in enumerate(B.CATALOG):
        for k in range(10):
            for mir in (False, True):
                s = B.compose(words, e['gloss'], rng, lambda c: c[0] not in test, mir)
                if s is not None: refs.append(s); ry.append(ci)
        for k in range(5):
            parts = []
            for w in e['gloss']:
                pool = [c for c in words[w] if c[0] in test]
                if not pool: parts = None; break
                parts.append(distort(pool[rng.integers(len(pool))][1], rng))
            if parts: qs.append(np.concatenate(parts)); qy.append(ci)
    D = all_dists(*pack(qs), *pack(refs), BAND)
    order = B.rank_classes(D, np.array(ry), len(B.CATALOG))
    r = [np.mean([y in o[:k] for y, o in zip(qy, order)]) for k in (1, 3, 5)]
    print(f'seed {seed}: {len(qs)} queries -> top-1 {r[0]:.3f} top-3 {r[1]:.3f} top-5 {r[2]:.3f}', flush=True)
    res.append(r)
m = np.mean(res, 0); print(f'HARD MEAN: top-1 {m[0]:.3f} top-3 {m[1]:.3f} top-5 {m[2]:.3f}')
