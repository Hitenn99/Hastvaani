"""
landmarks_isl.py
------------------
Extracts MediaPipe hand landmarks for up to TWO hands per image from the ISL
photos (isl_data/by_class), using every 2nd frame (the frames are near-
duplicate video frames, so this loses almost nothing) and 4 worker processes.
Saves model_assets/landmarks_isl.npz with:
  P    (N, 2, 21, 3)  landmarks in NORMALIZED image coords, hands sorted by wrist x
  M    (N, 2)         hand-present mask
  y    (N,)           class index
  seq  (N,)           frame index within its class (for a temporal split)
Run from the repo root.
"""
import multiprocessing as mp_proc
import pathlib

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

LABELS = [chr(ord('A') + i) for i in range(26)]
DATA_DIR = pathlib.Path('isl_data/by_class')
STEP = 2


CACHE = pathlib.Path('tmp/isl_lm')
CACHE.mkdir(parents=True, exist_ok=True)


def work(label_idx):
    cf = CACHE / f'{LABELS[label_idx]}.npz'
    if cf.exists():
        d = np.load(cf)
        return label_idx, d['P'], d['M'], d['seq']
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )
    lmk = vision.HandLandmarker.create_from_options(options)
    files = sorted((DATA_DIR / LABELS[label_idx]).glob('*.jpg'))[::STEP]
    P, M, seq = [], [], []
    for k, f in enumerate(files):
        try:
            arr = np.array(Image.open(f).convert('RGB'))
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception as e:  # MemoryError etc: skip the frame, keep going
            print('skip', f, repr(e), flush=True)
            continue
        hands = [np.array([[l.x, l.y, l.z] for l in h], dtype=np.float32) for h in res.hand_landmarks]
        if not hands:
            continue
        hands.sort(key=lambda h: h[0, 0])
        p = np.zeros((2, 21, 3), np.float32)
        m = np.zeros(2, np.float32)
        for i, h in enumerate(hands[:2]):
            p[i] = h
            m[i] = 1
        P.append(p); M.append(m); seq.append(k)
    np.savez(cf, P=np.array(P), M=np.array(M), seq=np.array(seq))
    print(LABELS[label_idx], len(P), '/', len(files), flush=True)
    return label_idx, np.array(P), np.array(M), np.array(seq)


if __name__ == '__main__':
    with mp_proc.Pool(2, maxtasksperchild=1) as pool:
        out = pool.map(work, range(26))
    P = np.concatenate([o[1] for o in out if len(o[1])])
    M = np.concatenate([o[2] for o in out if len(o[1])])
    seq = np.concatenate([o[3] for o in out if len(o[1])])
    y = np.concatenate([[o[0]] * len(o[1]) for o in out if len(o[1])])
    np.savez_compressed('model_assets/landmarks_isl.npz', P=P, M=M, y=y, seq=seq)
    print('saved', P.shape, 'two-hand fraction', float((M.sum(1) == 2).mean()))
