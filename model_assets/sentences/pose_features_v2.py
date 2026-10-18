"""
pose_features_v2.py
---------------------
Per-clip feature sequence v2 (hands + body pose) -- now used by the ASL sentence bank only
(asl_sentences/build_bank.py): ASL word signs are told apart mostly by handshape and the two
hands' relative position, which v2's full 126-coord hand block keeps (signer-held-out ASL
test: v2 77.9% top-1 vs 59% with v3's compact shape). ISL uses pose_features.py (v3).
MUST stay numerically identical to webapp/src/sentences/features.ts (clipFeaturesV2).

Why each piece is here (all measured with the full 687-video leave-one-out test in
dtw_lab.py / dtw_trim*.py; hands-only v1 was top-1 24.5%, this is 32.2%):
  * hand shape: the v1 128-float two-hand feature (train_sentences.normalize), unchanged
  * hand location: each hand's palm centre relative to the signer's BODY (shoulder-midpoint
    origin, shoulder-width scale) -- v1 threw away where the hands were (head vs. chest)
  * arms: pose-model elbows + wrists, same body frame (weight 2) -- tracked even in frames
    where the hand model loses the hand
  * trimming: frames before the first / after the last frame where a wrist is raised above
    0.6 shoulder-widths below the shoulder line are dropped (hands resting in the lap), so
    DTW doesn't have to align idle lead-in/lead-out of different lengths
  * a frame is kept if a hand OR a body was detected in it
"""
import numpy as np

from train_sentences import normalize, sort_hands

FEATURE_LEN = 128 + 4 + 8
W_LOC, W_ARM = 1.0, 2.0
TRIM_THR = 0.6
BAND = 0.2  # Sakoe-Chiba band for DTW, fraction of the reference length
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
PALM = [0, 5, 9, 13, 17]


def clip_features(P, M, pose, pm, W, H):
    """One clip's raw per-frame detections -> (n, FEATURE_LEN) float64.
    P (T,2,21,3) hand landmarks, M (T,2) hand presence, pose (T,33,>=2) body landmarks,
    pm (T,) body presence; all x/y normalised to the W x H frame (MediaPipe convention).
    Returns an empty (0, FEATURE_LEN) array if no frame had a hand or a body."""
    P, M = sort_hands(P[None], M[None])
    F = normalize(P, M)[0].astype(np.float64)
    P, M = P[0], M[0]
    T = len(P)
    px = np.array([W, H], np.float64)
    pose_px = pose[..., :2].astype(np.float64) * px
    hands_px = P[..., :2].astype(np.float64) * px

    body = pm > 0
    if body.any():
        sh = pose_px[body][:, [L_SH, R_SH]]
        origin = np.median(sh.mean(1), 0)
        scale = max(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)), 1e-3)
    else:  # never happened in the training data; keeps the live page working regardless
        hp = hands_px[M > 0].reshape(-1, 2)
        origin = hp.mean(0) if len(hp) else np.zeros(2)
        size = np.linalg.norm(hands_px[:, :, 9] - hands_px[:, :, 0], axis=-1)[M > 0]
        scale = max(4.0 * (np.median(size) if len(size) else 1.0), 1e-3)

    palm = hands_px[:, :, PALM].mean(2)  # (T,2,2)
    loc = (palm - origin) / scale * M[..., None]
    arms = np.stack([pose_px[:, [L_EL, L_WR]], pose_px[:, [R_EL, R_WR]]], 1)  # (T,2,2,2)
    order = np.argsort(arms[:, :, 1, 0], axis=1, kind='stable')
    arms = np.take_along_axis(arms, order[:, :, None, None], 1)
    arm = (arms - origin) / scale * pm[:, None, None, None]
    feats = np.concatenate([F, W_LOC * loc.reshape(T, -1), W_ARM * arm.reshape(T, -1)], -1)

    start, end = 0, T
    if body.any():
        wrist_y = ((pose_px[:, [L_WR, R_WR], 1] - origin[1]) / scale).min(1)
        active = np.where(body & (wrist_y < TRIM_THR))[0]
        if len(active) >= 4:
            start, end = active[0], active[-1] + 1
    keep = (M.sum(-1) > 0) | body
    keep[:start] = False
    keep[end:] = False
    return feats[keep]


def dataset_features(d, mirror=False):
    """sequences_pose.npz dict -> list of per-clip feature arrays (optionally x-mirrored)."""
    out = []
    for i in range(len(d['y'])):
        n = int(d['length'][i])
        P, pose = d['P'][i, :n].copy(), d['pose'][i, :n].copy()
        if mirror:
            P[..., 0] = 1.0 - P[..., 0]
            pose[..., 0] = 1.0 - pose[..., 0]
        W, H = d['wh'][i]
        out.append(clip_features(P, d['M'][i, :n], pose, d['pm'][i, :n], W, H))
    return out
