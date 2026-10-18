"""v3 experiments on top of pose_features (v2). Run from the repo root."""
import sys, time
import numpy as np
sys.path.insert(0, 'model_assets/sentences')
import pose_features as PF  # noqa: E402
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402
from train_sentences import normalize, sort_hands  # noqa: E402


def fill_gaps(P, M, max_gap):
    """Hold each hand slot's last/next seen landmarks across short detection dropouts
    (linear interpolation between the two neighbours)."""
    P, M = P.copy(), M.copy()
    T = len(P)
    for h in range(2):
        seen = np.where(M[:, h] > 0)[0]
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 < b - a <= max_gap + 1:
                for t in range(a + 1, b):
                    w = (t - a) / (b - a)
                    P[t, h] = (1 - w) * P[a, h] + w * P[b, h]
                    M[t, h] = 1
    return P, M


def assign_by_pose(P, M, pose, pm):
    """Put each detected hand in the slot of the pose arm (left/right) whose wrist is nearest,
    so a hand keeps its identity when the hands cross (instead of sorting by x)."""
    P2, M2 = np.zeros_like(P), np.zeros_like(M)
    for t in range(len(P)):
        hs = [P[t, h] for h in range(2) if M[t, h]]
        if not pm[t] or not hs:
            P2[t], M2[t] = P[t], M[t]
            continue
        wr = pose[t, [15, 16], :2]
        if len(hs) == 1:
            d = np.linalg.norm(wr - hs[0][0, :2], axis=-1)
            k = int(np.argmin(d)); P2[t, k] = hs[0]; M2[t, k] = 1
        else:
            d00 = np.linalg.norm(wr[0] - hs[0][0, :2]) + np.linalg.norm(wr[1] - hs[1][0, :2])
            d01 = np.linalg.norm(wr[0] - hs[1][0, :2]) + np.linalg.norm(wr[1] - hs[0][0, :2])
            if d00 <= d01: P2[t, 0], P2[t, 1] = hs[0], hs[1]
            else: P2[t, 0], P2[t, 1] = hs[1], hs[0]
            M2[t] = 1
    return P2, M2


def feats_v3(d, mirror=False, gap=0, by_pose=False):
    out = []
    for i in range(len(d['y'])):
        n = int(d['length'][i])
        P, M, pose, pm = d['P'][i, :n].copy(), d['M'][i, :n].copy(), d['pose'][i, :n].copy(), d['pm'][i, :n]
        if mirror:
            P[..., 0] = 1.0 - P[..., 0]; pose[..., 0] = 1.0 - pose[..., 0]
        W, H = d['wh'][i]
        if by_pose:
            # slots = pose arms; for mirrored clips swap arm identity so "right arm" stays the dominant slot
            px = np.array([W, H])
            Ppx = P.copy(); Ppx[..., :2] *= px
            posepx = pose.copy(); posepx[..., :2] *= px
            _, M2 = assign_by_pose(Ppx, M, posepx, pm)
            P, _ = assign_by_pose(P, M, pose * np.r_[px, 1, 1] / np.r_[px, 1, 1], pm)
            M = M2
            if mirror:
                P = P[:, ::-1].copy(); M = M[:, ::-1].copy()
        if gap:
            P, M = fill_gaps(P, M, gap)
        out.append(PF.clip_features(P, M, pose, pm, W, H, sort=not by_pose) if by_pose else PF.clip_features(P, M, pose, pm, W, H))
    return out


def run(name, D, y, agg='min', k=1):
    print(name, end=' '); return score(D, y, agg, k)


if __name__ == '__main__':
    d = load_pose(); y = d['y']
    for gap in (0, 2, 4, 8):
        t = time.time()
        D = distance_matrix(feats_v3(d, gap=gap), feats_v3(d, True, gap=gap), PF.BAND)
        run(f'gap {gap} ({time.time()-t:.0f}s)', D, y)
