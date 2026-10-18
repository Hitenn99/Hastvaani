"""Round-Y experiments: temporal smoothing, velocity blend, v2+v3 fusion. Scored on the cross-setup
test and camera->camera. Run from the repo root."""
import sys, time
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
import pose_features_v2 as PF2  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_exp_v3b import all_d  # noqa: E402
from dtw_lab import pack  # noqa: E402

d = load_pose(); y = d['y']; n = len(y); C = y.max() + 1
ph = d['wh'][:, 0] < 1000
ry = np.r_[y, y]; rp = np.r_[ph, ph]


def evaluate(D, tag):
    res = []
    for name, q, same in (('cross', np.ones(n, bool), False), ('cam', ~ph, True)):
        h = np.zeros(3); c = 0
        for i in np.where(q)[0]:
            ok = (rp == ph[i]) if same else (rp != ph[i]); ok = ok.copy(); ok[i] = ok[n + i] = False
            b = np.full(C, np.inf)
            for k in range(C):
                m = ok & (ry == k)
                if m.any(): b[k] = D[i, m].min()
            if not np.isfinite(b[y[i]]): continue
            o = np.argsort(b); h += [y[i] in o[:t] for t in (1, 3, 5)]; c += 1
        res.append(h / c)
    print(f'{tag:40s} cross {res[0].round(3)}  cam {res[1].round(3)}', flush=True)
    return res


def dist(s, sm):
    return all_d(*pack(s), *pack(s + sm), PF.BAND, 0)


def smooth(seqs, w):
    if w <= 1: return seqs
    k = np.ones(w) / w
    out = []
    for x in seqs:
        if len(x) < w: out.append(x); continue
        pad = np.pad(x, ((w // 2, w - 1 - w // 2), (0, 0)), mode='edge')
        out.append(np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(x.shape[1])], 1))
    return out


def with_vel(seqs, wv):
    out = []
    for x in seqs:
        v = np.gradient(x, axis=0) if len(x) > 1 else np.zeros_like(x)
        out.append(np.concatenate([x, wv * v], 1))
    return out


if __name__ == '__main__':
    s = PF.dataset_features(d); sm = PF.dataset_features(d, mirror=True)
    D3 = np.load('tmp/D_v3.npy')
    evaluate(D3, 'v3 shipped')
    for w in (3, 5, 7):
        t = time.time(); evaluate(dist(smooth(s, w), smooth(sm, w)), f'v3 smooth {w}')
    s5, sm5 = smooth(s, 5), smooth(sm, 5)
    for wv in (1.0, 3.0):
        evaluate(dist(with_vel(s5, wv), with_vel(sm5, wv)), f'v3 smooth5 + vel {wv}')
    s2 = PF2.dataset_features(d) if hasattr(PF2, 'dataset_features') else None
    if s2 is not None:
        D2 = dist(s2, PF2.dataset_features(d, mirror=True))
        np.save('tmp/D_v2.npy', D2)
        evaluate(D2, 'v2')
        for a in (0.25, 0.5):
            evaluate(D3 / np.median(D3) + a * D2 / np.median(D2), f'v3 + {a} v2 fused')
