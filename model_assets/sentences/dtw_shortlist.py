"""How accuracy depends on (a) how many candidate sentences are in play and (b) whether the
signer's own takes are in the bank. Run from the repo root."""
import sys
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_exp_v3b import all_d  # noqa: E402
from dtw_lab import pack  # noqa: E402


def class_scores(Drow, ref_y, ok, C):
    best = np.full(C, np.inf)
    for c in range(C):
        m = ok & (ref_y == c)
        if m.any(): best[c] = Drow[m].min()
    return best


if __name__ == '__main__':
    d = load_pose(); y = d['y']; n = len(y); C = y.max() + 1
    phone = d['wh'][:, 0] < 1000
    s = PF.dataset_features(d); sm = PF.dataset_features(d, mirror=True)
    D = all_d(*pack(s), *pack(s + sm), PF.BAND, 0)
    np.save('tmp/D_v3.npy', D)
    ref_y = np.concatenate([y, y]); ref_phone = np.concatenate([phone, phone])
    rng = np.random.default_rng(0)
    for setting in ('cross', 'same-setup LOO'):
        S = []
        for i in range(n):
            ok = (ref_phone != phone[i]) if setting == 'cross' else (ref_phone == phone[i])
            ok = ok.copy(); ok[i] = ok[n + i] = False
            S.append(class_scores(D[i], ref_y, ok, C))
        S = np.array(S)
        print(setting)
        for k in (101, 50, 20, 10, 5):
            hits = np.zeros(3); cnt = 0
            for i in range(n):
                if not np.isfinite(S[i, y[i]]): continue
                for _ in range(5):
                    others = rng.choice(np.delete(np.arange(C), y[i]), k - 1, replace=False)
                    cand = np.concatenate([[y[i]], others])
                    order = cand[np.argsort(S[i, cand])]
                    hits += [y[i] in order[:t] for t in (1, 3, 5)]; cnt += 1
            print(f'  {k:3d} candidates: top-1 {hits[0]/cnt:.3f} top-3 {hits[1]/cnt:.3f} top-5 {hits[2]/cnt:.3f}')
