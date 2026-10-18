"""Feature experiments for sentence DTW (see dtw_lab.py). Run from the repo root."""
import sys
import time

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import distance_matrix, load, mirror, score  # noqa: E402
from train_sentences import sort_hands  # noqa: E402

TIPS = [4, 8, 12, 16, 20]


def rich_feats(P, M, length, w_shape=1.0, w_pos=1.0, w_vel=0.0, w_pres=1.0, smooth=0):
    """Per real frame:
      shape: each hand centred on its own wrist, scaled by its own size (63 each)
      pos:   each hand's wrist + palm-centre xy in image coords, centred on the clip's
             mean hand position and scaled by the clip's median hand size (so camera
             distance / framing drop out, but where the hands move over time stays)
      vel:   frame-to-frame delta of pos
    """
    P, M = sort_hands(P, M)
    out = []
    for i in range(len(P)):
        n = length[i]
        p, m = P[i, :n].astype(np.float64), M[i, :n]
        keep = m.sum(-1) > 0
        p, m = p[keep], m[keep]
        T = len(p)
        if smooth and T > 2:
            # light temporal smoothing only across frames where the same slot is present
            ps = p.copy()
            for t in range(T):
                lo, hi = max(0, t - smooth), min(T, t + smooth + 1)
                for h in range(2):
                    if m[t, h]:
                        w = m[lo:hi, h]
                        ps[t, h] = (p[lo:hi, h] * w[:, None, None]).sum(0) / w.sum()
            p = ps
        wrist = p[:, :, 0, :]  # (T,2,3)
        size = np.linalg.norm(p[:, :, 9, :2] - p[:, :, 0, :2], axis=-1)  # wrist->middle MCP
        valid_sizes = size[m > 0]
        S = np.median(valid_sizes) if len(valid_sizes) else 1.0
        S = max(S, 1e-4)
        shape = (p - wrist[:, :, None, :]) / np.maximum(size, 1e-4)[:, :, None, None]
        shape = shape * m[:, :, None, None]
        palm = p[:, :, [0, 5, 9, 13, 17], :2].mean(2)  # (T,2,2)
        pos_pts = np.concatenate([wrist[:, :, :2], palm], -1)  # (T,2,4)
        present_pts = pos_pts[m > 0]
        centre = present_pts.reshape(-1, 2, 2).mean((0, 1)) if len(present_pts) else np.zeros(2)
        pos = (pos_pts - np.tile(centre, 2)) / S
        pos = pos * m[:, :, None]
        feats = [w_shape * shape.reshape(T, -1) / np.sqrt(21), w_pos * pos.reshape(T, -1),
                 w_pres * m]
        if w_vel:
            vel = np.zeros_like(pos)
            vel[1:] = (pos[1:] - pos[:-1]) * (m[1:, :, None] * m[:-1, :, None])
            feats.append(w_vel * vel.reshape(T, -1))
        out.append(np.concatenate(feats, -1))
    return out


def run(name, **kw):
    P, M, length, y, _ = DATA
    t = time.time()
    s = rich_feats(P, M, length, **kw)
    sm = rich_feats(mirror(P), M.copy(), length, **kw)
    D = distance_matrix(s, sm, kw.pop('band', 0.0) if 'band' in kw else 0.0)
    print(f'{name} ({time.time()-t:.0f}s)', flush=True)
    score(D, y)
    return D


if __name__ == '__main__':
    DATA = load()
    y = DATA[3]
    configs = {
        'shape only': dict(w_shape=1, w_pos=0, w_pres=1),
        'shape+pos 0.5': dict(w_shape=1, w_pos=0.5),
        'shape+pos 1': dict(w_shape=1, w_pos=1),
        'shape+pos 2': dict(w_shape=1, w_pos=2),
        'shape+pos 1+vel 2': dict(w_shape=1, w_pos=1, w_vel=2),
        'shape+pos 1 smooth1': dict(w_shape=1, w_pos=1, smooth=1),
    }
    for name, kw in configs.items():
        run(name, **kw)
