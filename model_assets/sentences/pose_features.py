"""
pose_features.py
------------------
Per-clip feature sequence for the sentence DTW matcher (v3: compact hand shape + body pose).
Used by BOTH the ISL bank (export_dtw_bank.py) and the ASL bank (asl_sentences/build_bank.py).
MUST stay numerically identical to webapp/src/sentences/features.ts (clipFeatures) --
see check_pose_parity.py.

v3 was tuned on the CROSS-SETUP test (eval_cross.py: queries from the phone recordings,
references only from the camera recordings and vice versa -- the closest this dataset gets to
"a new person on a new camera", i.e. a real user). v2 had been tuned on leave-one-out, where
the same signer's other takes of a sentence are in the bank, and scored only ~11% top-1 cross-
setup. v3: top-1 17.2% / top-3 31.3% / top-5 36.9% cross-setup (LOO 27.7 / 44.3 / 52.3).
v3.1 (shipped 2026-10-09, add_motion): 5-frame smoothing + arm velocity -> cross-setup
20.0 / 32.3 / 41.3% (experiments: dtw_exp_y.py).

Per kept frame (54 floats, + 8 arm-velocity floats in v3.1 = 62):
  * hand shape (2 x 20): per hand, 5 fingertips + 5 middle joints relative to the wrist, in
    pixels, divided by that hand's palm length (wrist -> middle-finger knuckle); x W_SHAPE/3.
    Coarser than v1's 63 raw coords per hand, which over-fitted each signer's exact handshape.
  * presence (2)
  * hand location (2 x 2): palm centre relative to the body (shoulder-midpoint origin,
    shoulder-width scale), x W_LOC
  * arms (2 x 2 x 2): pose elbows + wrists in the same body frame, x W_ARM -- the most
    signer-independent cue, hence the largest weight
  * trimming: only frames between the first and last frame with a wrist raised above
    TRIM_THR shoulder-widths below the shoulder line (drops the hands-in-lap lead-in/out)
  * a frame is kept if a hand OR a body was detected in it
Hands are ordered by wrist x (same as v1/v2).
"""
import numpy as np

from train_sentences import sort_hands

ARM_COLS = list(range(46, 54))
FEATURE_LEN = 40 + 2 + 4 + 8 + len(ARM_COLS)  # + arm velocity (add_motion)
# ISL defaults; each bank stores the params it was built with in dtw_meta.json ("features"),
# and the browser reads them from there (ASL uses its own, see asl_sentences/build_bank.py).
W_SHAPE, W_LOC, W_ARM = 2.0, 0.5, 8.0
TRIM_THR = 0.4
SMOOTH, W_VEL = 5, 5.0
ISL_PARAMS = {'wShape': W_SHAPE, 'wLoc': W_LOC, 'wArm': W_ARM, 'trimThr': TRIM_THR,
              'smooth': SMOOTH, 'wVel': W_VEL}
BAND = 0.2  # Sakoe-Chiba band for DTW, fraction of the reference length
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
PALM = [0, 5, 9, 13, 17]
SHAPE_PTS = [4, 8, 12, 16, 20, 3, 6, 10, 14, 18]


def clip_features(P, M, pose, pm, W, H, params=None):
    """One clip's raw per-frame detections -> (n, FEATURE_LEN) float64.
    P (T,2,21,3) hand landmarks, M (T,2) hand presence, pose (T,33,>=2) body landmarks,
    pm (T,) body presence; all x/y normalised to the W x H frame (MediaPipe convention).
    Returns an empty (0, FEATURE_LEN) array if no frame had a hand or a body.
    `params`: {wShape, wLoc, wArm, trimThr} (default: ISL_PARAMS)."""
    p = ISL_PARAMS if params is None else params
    P, M = sort_hands(P[None], M[None])
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

    palm_len = np.linalg.norm(hands_px[:, :, 9] - hands_px[:, :, 0], axis=-1)[..., None, None] + 1e-6
    shape = (hands_px[:, :, SHAPE_PTS] - hands_px[:, :, :1]) / palm_len * M[..., None, None]
    palm = hands_px[:, :, PALM].mean(2)  # (T,2,2)
    loc = (palm - origin) / scale * M[..., None]
    arms = np.stack([pose_px[:, [L_EL, L_WR]], pose_px[:, [R_EL, R_WR]]], 1)  # (T,2,2,2)
    order = np.argsort(arms[:, :, 1, 0], axis=1, kind='stable')
    arms = np.take_along_axis(arms, order[:, :, None, None], 1)
    arm = (arms - origin) / scale * pm[:, None, None, None]
    feats = np.concatenate([p['wShape'] / 3 * shape.reshape(T, -1), M.astype(np.float64),
                            p['wLoc'] * loc.reshape(T, -1), p['wArm'] * arm.reshape(T, -1)], -1)

    start, end = 0, T
    if body.any():
        wrist_y = ((pose_px[:, [L_WR, R_WR], 1] - origin[1]) / scale).min(1)
        active = np.where(body & (wrist_y < p['trimThr']))[0]
        if len(active) >= 4:
            start, end = active[0], active[-1] + 1
    keep = (M.sum(-1) > 0) | body
    keep[:start] = False
    keep[end:] = False
    return add_motion(feats[keep], p)


def add_motion(f, p):
    """v3.1 (2026-10-09): moving-average smoothing over `smooth` kept frames (edge-padded; clips
    shorter than the window are left as is), then the arm features' velocity (central
    differences, one-sided at the ends) x wVel appended. Cross-setup 16.9/30.9/36.8 ->
    20.0/32.3/41.3% (top-1/3/5); no gain on ASL, so the ASL bank does not use it."""
    w, wv = int(p.get('smooth', 1)), float(p.get('wVel', 0.0))
    if w > 1 and len(f) >= w:
        pad = np.pad(f, ((w // 2, w - 1 - w // 2), (0, 0)), mode='edge')
        c = np.cumsum(np.concatenate([np.zeros((1, f.shape[1])), pad]), 0)
        f = (c[w:] - c[:-w]) / w
    if wv:
        arm = f[:, ARM_COLS]
        v = np.zeros_like(arm)
        if len(f) >= 2:
            v[1:-1] = (arm[2:] - arm[:-2]) / 2
            v[0] = arm[1] - arm[0]
            v[-1] = arm[-1] - arm[-2]
        f = np.concatenate([f, wv * v], 1)
    return f


def dataset_features(d, mirror=False, params=None):
    """sequences_pose.npz dict -> list of per-clip feature arrays (optionally x-mirrored)."""
    out = []
    for i in range(len(d['y'])):
        n = int(d['length'][i])
        P, pose = d['P'][i, :n].copy(), d['pose'][i, :n].copy()
        if mirror:
            P[..., 0] = 1.0 - P[..., 0]
            pose[..., 0] = 1.0 - pose[..., 0]
        W, H = d['wh'][i]
        out.append(clip_features(P, d['M'][i, :n], pose, d['pm'][i, :n], W, H, params))
    return out
