"""Per-sentence accuracy for both sentence banks, written to tmp/per_sentence_accuracy.json.
ISL: cross-setup test (eval_cross.py protocol) from the cached v3 distance matrix tmp/D_v3.npy
(run dtw_shortlist.py first). ASL: build_bank.py's signer-held-out test, per sentence.
Run from the repo root."""
import json
import sys
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
sys.path.insert(0, 'model_assets/asl_sentences')
from dtw_lab import all_dists, pack  # noqa: E402


def isl():
    d = np.load('model_assets/sentences/sequences_pose.npz', allow_pickle=True)
    y = d['y']; n = len(y); C = y.max() + 1; ph = d['wh'][:, 0] < 1000
    D = np.load('tmp/D_v3.npy'); ry = np.r_[y, y]; rp = np.r_[ph, ph]
    hits = np.zeros((C, 2)); cnt = np.zeros(C)
    for i in range(n):
        ok = rp != ph[i]
        b = np.array([D[i, ok & (ry == k)].min() if (ok & (ry == k)).any() else np.inf for k in range(C)])
        o = np.argsort(b)
        hits[y[i]] += [y[i] == o[0], y[i] in o[:3]]; cnt[y[i]] += 1
    return [str(s) for s in d['sentences']], hits, cnt


def asl(seeds=range(5), n_queries=10):
    import build_bank as B
    words = B.load_words(); C = len(B.CATALOG)
    signers = sorted({c[0] for clips in words.values() for c in clips})
    hits = np.zeros((C, 2)); cnt = np.zeros(C)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test = set(rng.choice(signers, int(len(signers) * 0.3), replace=False).tolist())
        refs, ref_y, qs, q_y = [], [], [], []
        for ci, e in enumerate(B.CATALOG):
            for _ in range(7):
                for mirror in (False, True):
                    s = B.compose(words, e['gloss'], rng, lambda c: c[0] not in test, mirror)
                    if s is not None: refs.append(s); ref_y.append(ci)
            for _ in range(n_queries):
                s = B.compose(words, e['gloss'], rng, lambda c: c[0] in test)
                if s is not None: qs.append(s); q_y.append(ci)
        D = all_dists(*pack(qs), *pack(refs), B.BAND)
        order = B.rank_classes(D, np.array(ref_y), C)
        for yy, o in zip(q_y, order):
            hits[yy] += [yy == o[0], yy in o[:3]]; cnt[yy] += 1
        print(f'ASL seed {seed} done', flush=True)
    return [e['sentence'] for e in B.CATALOG], hits, cnt


if __name__ == '__main__':
    out = {}
    for lang, fn in (('isl', isl), ('asl', asl)):
        names, hits, cnt = fn()
        out[lang] = [{'sentence': s, 'n': int(c), 'top1': float(h[0] / c) if c else None,
                      'top3': float(h[1] / c) if c else None} for s, h, c in zip(names, hits, cnt)]
    json.dump(out, open('tmp/per_sentence_accuracy.json', 'w'), indent=1)
    print('wrote tmp/per_sentence_accuracy.json')
