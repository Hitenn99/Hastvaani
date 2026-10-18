"""Body-relative (pose) feature experiments for sentence DTW. Run from the repo root
after extract_pose.py has written model_assets/sentences/sequences_pose.npz."""
import sys
import time

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import distance_matrix, score  # noqa: E402
from train_sentences import normalize, sort_hands  # noqa: E402

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, NOSE = 11, 12, 13, 14, 15, 16, 0
PALM = [0, 5, 9, 13, 17]


def load_pose():
    d = np.load('model_assets/sentences/sequences_pose.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def mirror_all(d):
    d = dict(d)
    P = d['P'].copy(); P[..., 0] = 1.0 - P[..., 0]
    pose = d['pose'].copy(); pose[..., 0] = 1.0 - pose[..., 0]
    d['P'], d['pose'] = P, pose
    return d


def body_frame(pose_px, pm):
    """Clip-level body reference from the frames where a body was found: shoulder-midpoint
    origin and shoulder-width scale (medians, so one bad frame can't move it)."""
    ok = pm > 0
    if ok.sum() == 0:
        return None
    sh = pose_px[ok][:, [L_SH, R_SH], :2]
    origin = np.median(sh.mean(1), 0)
    width = np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    return origin, max(width, 1e-3)


def feats(d, w_hand=1.0, w_loc=1.0, w_arm=0.0, w_pres=1.0, frames='hands', fallback='centroid'):
    P, M = sort_hands(d['P'], d['M'])
    F = normalize(P, M).astype(np.float64)
    F[..., 126:] *= w_pres
    F[..., :126] *= w_hand
    out, n_nobody = [], 0
    for i in range(len(P)):
        n = int(d['length'][i])
        W, H = d['wh'][i]
        px = np.array([W, H])
        pose = d['pose'][i, :n].astype(np.float64)
        pm = d['pm'][i, :n]
        pose_px = pose[..., :2] * px
        hands_px = P[i, :n, :, :, :2].astype(np.float64) * px
        m = M[i, :n]
        bf = body_frame(np.concatenate([pose_px, pose[..., 2:]], -1), pm)
        if bf is None:
            n_nobody += 1
            # no body anywhere: fall back to the hands' own clip centroid / size
            hp = hands_px[m > 0]
            origin = hp.reshape(-1, 2).mean(0) if len(hp) else np.zeros(2)
            size = np.linalg.norm(hands_px[:, :, 9] - hands_px[:, :, 0], axis=-1)[m > 0]
            scale = 4.0 * (np.median(size) if len(size) else 1.0)
        else:
            origin, scale = bf
        # hand palm centres relative to the body, per sorted slot
        palm = hands_px[:, :, PALM].mean(2)  # (n,2,2)
        loc = (palm - origin) / scale * m[..., None]
        parts = [F[i, :n], w_loc * loc.reshape(n, -1)]
        if w_arm:
            # elbows + wrists from the POSE model (present even when the hand model misses),
            # arms ordered by wrist x like the hands are
            arms = np.stack([pose_px[:, [L_EL, L_WR]], pose_px[:, [R_EL, R_WR]]], 1)  # (n,2,2,2)
            order = np.argsort(arms[:, :, 1, 0], axis=1)
            arms = np.take_along_axis(arms, order[:, :, None, None], 1)
            arm = (arms - origin) / scale * pm[:, None, None, None]
            parts.append(w_arm * arm.reshape(n, -1))
        f = np.concatenate(parts, -1)
        if frames == 'hands':
            keep = m.sum(-1) > 0
        else:  # hands or body
            keep = (m.sum(-1) > 0) | (pm > 0)
        out.append(f[keep])
    return out, n_nobody


def run(name, band=0.2, **kw):
    t = time.time()
    s, nb = feats(DATA, **kw)
    sm, _ = feats(MIRR, **kw)
    D = distance_matrix(s, sm, band)
    print(f'{name} ({time.time()-t:.0f}s, {nb} clips w/o body)', flush=True)
    return score(D, DATA['y'])


if __name__ == '__main__':
    DATA = load_pose()
    MIRR = mirror_all(DATA)
    print('body found in', f"{(DATA['pm'].sum() / DATA['length'].sum()):.3f}", 'of frames')
    run('baseline (w_loc=0)', w_loc=0)
    for wl in (0.5, 1, 2):
        run(f'loc {wl}', w_loc=wl)
    for wa in (0.5, 1):
        run(f'loc 1 + arm {wa}', w_loc=1, w_arm=wa)
    run('loc 1 + arm 1, frames=any', w_loc=1, w_arm=1, frames='any')
