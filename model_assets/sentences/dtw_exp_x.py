"""Feature search against the CROSS-SETUP test (eval_cross.py). Run from the repo root."""
import sys, itertools
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from eval_cross import cross_eval  # noqa: E402
from train_sentences import normalize, sort_hands  # noqa: E402

TIPS = [4, 8, 12, 16, 20]


def hand_desc(h):
    """Compact per-hand shape: 5 fingertips + 5 PIP-ish joints relative to wrist, scaled by palm size (xy only)."""
    size = np.linalg.norm(h[..., 9, :2] - h[..., 0, :2], axis=-1)[..., None, None] + 1e-6
    pts = h[..., [4, 8, 12, 16, 20, 3, 6, 10, 14, 18], :2] - h[..., 0:1, :2]
    return (pts / size).reshape(*h.shape[:-2], -1)


def feats(d, mirror=False, shape='v1', ws=1.0, wl=1.0, wa=2.0, trim=0.6, centre=False, smooth=0):
    out = []
    for i in range(len(d['y'])):
        n = int(d['length'][i])
        P, M, pose, pm = d['P'][i, :n].copy(), d['M'][i, :n], d['pose'][i, :n].copy(), d['pm'][i, :n]
        if mirror:
            P[..., 0] = 1 - P[..., 0]; pose[..., 0] = 1 - pose[..., 0]
        W, H = d['wh'][i]
        base = PF.clip_features(P, M, pose, pm, W, H) if trim == 0.6 else None
        # rebuild with options (same body frame / trim logic as PF, parametrised)
        Ps, Ms = sort_hands(P[None], M[None]); Ps, Ms = Ps[0], Ms[0]
        px = np.array([W, H], float)
        pose_px = pose[..., :2] * px; hands_px = Ps[..., :2] * px
        body = pm > 0
        sh = pose_px[body][:, [11, 12]]
        origin = np.median(sh.mean(1), 0); scale = max(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)), 1e-3)
        parts = []
        if shape == 'v1':
            parts.append(ws * normalize(Ps[None], Ms[None])[0][:, :126])
        elif shape == 'desc':
            hp = Ps.copy(); hp[..., :2] = hands_px
            parts.append(ws * (hand_desc(hp) * Ms[..., None]).reshape(n, -1) / 3)
        parts.append(Ms)
        palm = hands_px[:, :, PF.PALM].mean(2)
        loc = (palm - origin) / scale * Ms[..., None]
        arms = np.stack([pose_px[:, [13, 15]], pose_px[:, [14, 16]]], 1)
        order = np.argsort(arms[:, :, 1, 0], axis=1, kind='stable')
        arms = np.take_along_axis(arms, order[:, :, None, None], 1)
        arm = (arms - origin) / scale * pm[:, None, None, None]
        loc = loc.reshape(n, -1); arm = arm.reshape(n, -1)
        f = np.concatenate(parts + [wl * loc, wa * arm], -1)
        start, end = 0, n
        wy = ((pose_px[:, [15, 16], 1] - origin[1]) / scale).min(1)
        act = np.where(body & (wy < trim))[0]
        if len(act) >= 4: start, end = act[0], act[-1] + 1
        keep = (Ms.sum(-1) > 0) | body; keep[:start] = False; keep[end:] = False
        f = f[keep]
        if centre and len(f):
            k0 = f.shape[1] - 12  # loc(4)+arm(8) are the last 12
            f[:, k0:] -= f[:, k0:].mean(0)
        if smooth and len(f) > 2:
            k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            f = np.stack([np.convolve(np.pad(f[:, j], smooth, mode='edge'), k, 'valid') for j in range(f.shape[1])], 1)
        out.append(f)
    return out


def run(name, metric=0, **kw):
    s, sm = feats(D0, **kw), feats(D0, True, **kw)
    r = cross_eval(s, sm, D0, metric, verbose=False)
    a = (r['phone->camera'] + r['camera->phone']) / 2
    print(f'{name:45s} mean top1 {a[0]:.3f} top3 {a[1]:.3f} top5 {a[2]:.3f}', flush=True)
    return a


if __name__ == '__main__':
    D0 = load_pose()
    run('v1 shape, default')
    run('no shape', shape='none')
    run('desc shape', shape='desc')
    run('v1 shape 0.5', ws=0.5)
    run('v1 shape 0.5, euclid', ws=0.5, metric=1)
    run('no trim', trim=99)
    run('trim 1.0', trim=1.0)
    run('centre loc/arm', centre=True)
    run('smooth 1', smooth=1)
    run('smooth 2', smooth=2)
    run('arm 1', wa=1)
    run('arm 3', wa=3)
    run('loc 2', wl=2)
