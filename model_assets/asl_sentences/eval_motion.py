"""Does temporal smoothing + velocity (helped ISL cross-setup) help the ASL bank too?
Same signer-held-out protocol as build_bank.py evaluate(). Run from the repo root."""
import sys
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
sys.path.insert(0, 'model_assets/asl_sentences')
import build_bank as B  # noqa: E402
from dtw_lab import all_dists, pack  # noqa: E402
from dtw_exp_y import smooth, with_vel  # noqa: E402

words = B.load_words(); C = len(B.CATALOG)
signers = sorted({c[0] for clips in words.values() for c in clips})
sets = []
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    test = set(rng.choice(signers, int(len(signers) * 0.3), replace=False).tolist())
    refs, ref_y, qs, q_y = [], [], [], []
    for ci, e in enumerate(B.CATALOG):
        for _ in range(7):
            for mirror in (False, True):
                s = B.compose(words, e['gloss'], rng, lambda c: c[0] not in test, mirror)
                if s is not None: refs.append(s); ref_y.append(ci)
        for _ in range(5):
            s = B.compose(words, e['gloss'], rng, lambda c: c[0] in test)
            if s is not None: qs.append(s); q_y.append(ci)
    sets.append((refs, np.array(ref_y), qs, np.array(q_y)))

for w, wv in ((1, 0), (3, 0), (3, 1), (3, 3), (5, 1), (5, 3), (5, 5)):
    rs = []
    for refs, ref_y, qs, q_y in sets:
        f = lambda S: with_vel(smooth(S, w), wv) if wv else smooth(S, w)  # noqa: E731
        D = all_dists(*pack(f(qs)), *pack(f(refs)), B.BAND)
        o = B.rank_classes(D, ref_y, C)
        rs.append([np.mean([y in oo[:k] for y, oo in zip(q_y, o)]) for k in (1, 3, 5)])
    print(f'smooth {w} vel {wv}: {np.mean(rs, 0).round(3)}', flush=True)
