"""Cross-setup test: queries from one recording setup (phone 240x320 or camera 1920x1080), references
ONLY from the other -- the closest proxy for 'a new person on a new camera' this dataset allows.
Run from the repo root."""
import sys
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_exp_v3b import all_d  # noqa: E402
from dtw_lab import pack  # noqa: E402


def cross_eval(s, sm, d, metric=0, verbose=True):
    y = d['y']; n = len(y)
    phone = d['wh'][:, 0] < 1000
    refs = s + sm; ref_y = np.concatenate([y, y]); ref_phone = np.concatenate([phone, phone])
    D = all_d(*pack(s), *pack(refs), PF.BAND, metric)
    out = {}
    for name, qmask in (('phone->camera', phone), ('camera->phone', ~phone)):
        hits = np.zeros(3); cnt = 0
        for i in np.where(qmask)[0]:
            ok = ref_phone != phone[i]
            best = np.full(y.max() + 1, np.inf)
            for c in range(len(best)):
                m = ok & (ref_y == c)
                if m.any(): best[c] = D[i, m].min()
            order = np.argsort(best)
            hits += [y[i] in order[:k] for k in (1, 3, 5)]; cnt += 1
        out[name] = hits / cnt
        if verbose: print(f'  {name} (n={cnt}): top-1 {hits[0]/cnt:.3f} top-3 {hits[1]/cnt:.3f} top-5 {hits[2]/cnt:.3f}')
    return out


if __name__ == '__main__':
    d = load_pose()
    s = PF.dataset_features(d); sm = PF.dataset_features(d, mirror=True)
    print('v2 shipped (sq):'); cross_eval(s, sm, d, 0)
    print('v2 euclid:'); cross_eval(s, sm, d, 1)
