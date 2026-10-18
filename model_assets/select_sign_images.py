"""
select_sign_images.py
------------------------
Picks ONE representative photo per letter for the website's Learn page, for
both ASL and ISL, and writes them to webapp/public/signs/{asl,isl}/<LETTER>.jpg.

"Representative" = the training photo the landmark classifier gets right with the
highest confidence (a clear, canonical example of the sign). ASL uses hand
crops from asl_data/by_class_cropped; ISL uses isl_data/by_class.
Run from the repo root.
"""
import pathlib

import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

LABELS = [chr(65 + i) for i in range(26)]
OUT = pathlib.Path('webapp/public/signs')
(OUT / 'asl').mkdir(parents=True, exist_ok=True)
(OUT / 'isl').mkdir(parents=True, exist_ok=True)


def make_landmarker(n):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE, num_hands=n, min_hand_detection_confidence=0.5))


def detect(lmk, img):
    r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img)))
    return [np.array([[l.x, l.y, l.z] for l in h], np.float32) for h in r.hand_landmarks]


def save(img, path, size):
    w, h = img.size
    s = size / max(w, h)
    img.resize((round(w * s), round(h * s)), Image.LANCZOS).save(path, quality=90)


# ---------- ASL ----------
def norm_asl(P):
    P = P - P[:, :1, :]
    s = np.linalg.norm(P[:, :, :2], axis=2).max(1)[:, None, None] + 1e-6
    return (P / s).reshape(len(P), -1).astype(np.float32)


asl_model = tf.keras.models.load_model('model_assets/asl_landmarks_model.keras')
lmk1 = make_landmarker(1)
for li, L in enumerate(LABELS):
    files = sorted(pathlib.Path(f'asl_data/by_class_cropped/{L}').glob('*.jpg'))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(120, len(files))).astype(int)]
    best, best_conf = None, -1
    for f in files:
        img = Image.open(f).convert('RGB')
        if min(img.size) < 96:
            continue
        hands = detect(lmk1, img)
        if len(hands) != 1:
            continue
        p = asl_model.predict(norm_asl(hands[0][None]), verbose=0)[0]
        if p.argmax() == li and p[li] > best_conf:
            best, best_conf = f, float(p[li])
    print('ASL', L, best.name if best else None, round(best_conf, 3), flush=True)
    if best:
        save(Image.open(best).convert('RGB'), OUT / 'asl' / f'{L}.jpg', 320)


# ---------- ISL ----------
def norm_isl(P, M):
    key = np.where(M > 0, P[:, :, 0, 0], 9.0)
    o = np.argsort(key, 1)
    P = np.take_along_axis(P, o[:, :, None, None], 1)
    M = np.take_along_axis(M, o, 1)
    c = (P * M[:, :, None, None]).sum((1, 2)) / (M.sum(1)[:, None] * 21.0)
    Q = (P - c[:, None, None, :]) * M[:, :, None, None]
    s = np.linalg.norm(Q[..., :2], axis=3).max((1, 2))[:, None, None, None] + 1e-6
    return np.concatenate([(Q / s).reshape(len(P), -1), M], 1).astype(np.float32)


isl_model = tf.keras.models.load_model('model_assets/isl_landmarks_model.keras')
for li, L in enumerate(LABELS):
    d = np.load(f'tmp/isl_lm/{L}.npz')
    P, M, seq = d['P'], d['M'], d['seq']
    probs = isl_model.predict(norm_isl(P, M), verbose=0)
    conf = probs[:, li].copy()
    conf[probs.argmax(1) != li] = -1
    # Only consider frames using the letter's usual number of hands.
    usual = np.round(np.median(M.sum(1)))
    conf[M.sum(1) != usual] = -1
    # Spread the choice around the middle of the recording, not the very first frames.
    order = np.argsort(-conf)
    idx = order[0]
    files = sorted(pathlib.Path(f'isl_data/by_class/{L}').glob('*.jpg'))[::2]
    f = files[int(seq[idx])]
    print('ISL', L, f.name, round(float(conf[idx]), 3), flush=True)
    save(Image.open(f).convert('RGB'), OUT / 'isl' / f'{L}.jpg', 320)
print('done')
