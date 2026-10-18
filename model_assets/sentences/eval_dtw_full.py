"""Full leave-one-out DTW kNN evaluation across all 687 videos. Run from the repo root."""
import sys
import time

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from train_sentences import normalize, sort_hands  # noqa: E402


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
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, T1 + 1):
        row = cost[i - 1]
        for j in range(1, T2 + 1):
            D[i, j] = row[j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[T1, T2] / (T1 + T2)


def main():
    d = np.load('model_assets/sentences/sequences.npz', allow_pickle=True)
    P, M, length, y = d['P'], d['M'], d['length'], d['y']
    sentences = [str(s) for s in d['sentences']]
    n = len(P)
    seqs = real_seqs(P, M, length)

    Pm = P.copy(); Pm[..., 0] = 1.0 - Pm[..., 0]
    seqs_m = real_seqs(*sort_hands(Pm, M.copy()), length)

    ref_seqs = seqs + seqs_m
    ref_y = np.concatenate([y, y])
    ref_src = np.concatenate([np.arange(n), np.arange(n)])

    t0 = time.time()
    correct1 = correct3 = correct5 = 0
    per_class_total = np.zeros(len(sentences), int)
    per_class_correct = np.zeros(len(sentences), int)
    for i in range(n):
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
        per_class_total[y[i]] += 1
        if y[i] == top_classes[0]:
            correct1 += 1
            per_class_correct[y[i]] += 1
        if y[i] in top_classes[:3]:
            correct3 += 1
        if y[i] in top_classes[:5]:
            correct5 += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f'{i+1}/{n} done, {elapsed:.0f}s elapsed, eta {eta:.0f}s, running top-1 {correct1/(i+1):.4f}', flush=True)

    print(f'\nFULL DTW LOO (n={n}): top-1 {correct1/n:.4f}  top-3 {correct3/n:.4f}  top-5 {correct5/n:.4f}')
    print(f'total time {(time.time()-t0)/60:.1f} min')

    import json
    worst = sorted(range(len(sentences)), key=lambda c: per_class_correct[c] / max(per_class_total[c], 1))[:10]
    print('worst classes (correct/total):')
    for c in worst:
        print(f'  {sentences[c]}: {per_class_correct[c]}/{per_class_total[c]}')

    json.dump({
        'top1': correct1 / n, 'top3': correct3 / n, 'top5': correct5 / n, 'n': n,
    }, open('model_assets/sentences/dtw_result.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
