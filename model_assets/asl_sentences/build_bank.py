"""
build_bank.py
---------------
ASL sentence builder, step 2: builds DTW reference sequences for each catalog sentence by
CONCATENATING the v2 (hands + body pose) features of real WLASL word clips, in the sentence's
ASL gloss order -- then evaluates and exports them in the same format as the ISL bank, so the
same browser matcher (webapp/src/sentences/dtw.ts) works unchanged.

Why concatenation: there is no public ASL dataset of fixed sentences each recorded several
times (the ISL-CSLTR format), but WLASL has ~6 signers per word. Each word clip is featurised
on its own (body reference + idle-trim per clip, exactly pose_features.clip_features), so
the rest pose before/after every word is cut and only the signing is chained.

  python model_assets/asl_sentences/build_bank.py eval    # signer-held-out evaluation
  python model_assets/asl_sentences/build_bank.py export  # write webapp/public/sentence_model_asl/

Honest-evaluation caveat: queries are ALSO concatenations (of word clips by signers never used
in the references), not real continuously-signed sentences -- real signing blends words
(co-articulation), so live accuracy will be lower than this number.
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import all_dists, pack  # noqa: E402
# v2 features (not ISL's v3): ASL signs differ mostly by handshape + the two hands' relative
# position, which v2 keeps -- v2 77.9% vs v3 at best 59% top-1 on the signer-held-out test.
from pose_features_v2 import BAND, FEATURE_LEN, clip_features  # noqa: E402

CACHE = pathlib.Path('tmp/asl_word_lm')
CATALOG = json.load(open('model_assets/asl_sentences/catalog.json'))
OUT = pathlib.Path('webapp/public/sentence_model_asl')
MIN_FRAMES = 3


def load_words():
    """gloss -> list of (signer, feats, mirrored feats) for every usable WLASL clip."""
    words = {}
    for e in CATALOG:
        for w in e['gloss']:
            if w in words:
                continue
            clips = []
            for f in sorted((CACHE / w).glob('*.npz')):
                d = np.load(f)
                if len(d['P']) < MIN_FRAMES:
                    continue
                W, H = d['wh']
                P, pose = d['P'], d['pose']
                a = clip_features(P, d['M'], pose, d['pm'], W, H)
                Pm, posem = P.copy(), pose.copy()
                Pm[..., 0] = 1.0 - Pm[..., 0]
                posem[..., 0] = 1.0 - posem[..., 0]
                b = clip_features(Pm, d['M'], posem, d['pm'], W, H)
                if len(a) >= MIN_FRAMES:
                    clips.append((int(d['signer']), a, b))
            words[w] = clips
    return words


def compose(words, gloss, rng, pool_filter=None, mirror=False):
    """One sentence sequence: one random clip per word (from clips passing pool_filter)."""
    parts = []
    for w in gloss:
        pool = [c for c in words[w] if pool_filter is None or pool_filter(c)]
        if not pool:
            return None
        c = pool[rng.integers(len(pool))]
        parts.append(c[2] if mirror else c[1])
    return np.concatenate(parts)


def rank_classes(D, ref_y, n_classes):
    best = np.full((D.shape[0], n_classes), np.inf)
    for c in range(n_classes):
        best[:, c] = D[:, ref_y == c].min(1)
    return np.argsort(best, 1)


def evaluate(words, n_refs=10, n_queries=5, test_frac=0.3, seeds=(0, 1, 2)):
    signers = sorted({c[0] for clips in words.values() for c in clips})
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test = set(rng.choice(signers, int(len(signers) * test_frac), replace=False).tolist())
        is_train = lambda c: c[0] not in test  # noqa: E731
        is_test = lambda c: c[0] in test  # noqa: E731
        refs, ref_y, qs, q_y = [], [], [], []
        for ci, e in enumerate(CATALOG):
            for k in range(n_refs):
                for mirror in (False, True):
                    s = compose(words, e['gloss'], rng, is_train, mirror)
                    if s is not None:
                        refs.append(s); ref_y.append(ci)
            for k in range(n_queries):
                s = compose(words, e['gloss'], rng, is_test)
                if s is not None:
                    qs.append(s); q_y.append(ci)
        fq, oq, lq = pack(qs)
        fr, orr, lr = pack(refs)
        D = all_dists(fq, oq, lq, fr, orr, lr, BAND)
        order = rank_classes(D, np.array(ref_y), len(CATALOG))
        q_y = np.array(q_y)
        r = [np.mean([y in o[:k] for y, o in zip(q_y, order)]) for k in (1, 3, 5)]
        covered = len(set(q_y.tolist()))
        print(f'seed {seed}: {len(test)} held-out signers, {len(qs)} queries over {covered} sentences, '
              f'{len(refs)} refs -> top-1 {r[0]:.3f} top-3 {r[1]:.3f} top-5 {r[2]:.3f}', flush=True)
        results.append(r)
    m = np.mean(results, 0)
    print(f'MEAN over seeds: top-1 {m[0]:.3f}  top-3 {m[1]:.3f}  top-5 {m[2]:.3f}')
    return m


def export(words, n_refs=7, seed=0):
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    meta, offset = [], 0
    with open(OUT / 'dtw_bank.bin', 'wb') as f:
        for ci, e in enumerate(CATALOG):
            for k in range(n_refs):
                for mirror in (False, True):
                    s = compose(words, e['gloss'], rng, None, mirror)
                    f.write(s.astype(np.float32).tobytes())
                    meta.append({'offset': offset, 'length': len(s), 'classIndex': ci})
                    offset += len(s)
    json.dump({'featureLen': FEATURE_LEN, 'band': BAND, 'features': {'version': 2}, 'entries': meta},
              open(OUT / 'dtw_meta.json', 'w'))
    json.dump([e['sentence'] for e in CATALOG], open(OUT / 'labels.json', 'w'), ensure_ascii=False)
    json.dump({e['sentence']: ' '.join(w.upper() for w in e['gloss']) for e in CATALOG},
              open(OUT / 'glosses.json', 'w'), ensure_ascii=False, indent=1)
    mb = os.path.getsize(OUT / 'dtw_bank.bin') / 1024 / 1024
    print(f'wrote {len(meta)} reference sequences ({mb:.1f} MB) to {OUT}')


if __name__ == '__main__':
    words = load_words()
    thin = {w: len(c) for w, c in words.items() if len(c) < 3}
    print(f'{len(words)} words, {sum(len(c) for c in words.values())} usable clips; words with <3 clips: {thin}')
    if 'export' in sys.argv:
        export(words)
    else:
        evaluate(words, n_refs=7)
