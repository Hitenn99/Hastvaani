"""
landmarks_asl.py
------------------
Extracts MediaPipe hand landmarks (21 x 3) from the ASL training crops
(asl_data/by_class_cropped) and from the held-out EitanG98 validation split
(crossval_data/val.parquet), and saves them to landmarks_asl.npz.

Landmarks are invariant to background / lighting / skin tone / camera, which
is exactly what the image CNN failed to generalize across (see README.md).
Run from the repo root.
"""
import io
import pathlib

import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

LABELS = [chr(ord('A') + i) for i in range(26)]
DATA_DIR = pathlib.Path('asl_data/by_class_cropped')

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
landmarker = vision.HandLandmarker.create_from_options(options)


def extract(img_rgb):
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
    if not res.hand_landmarks:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]], dtype=np.float32)


X, y, src = [], [], []
for li, label in enumerate(LABELS):
    for f in sorted((DATA_DIR / label).glob('*.jpg')):
        # Keep the file name so we can tell the two source datasets apart.
        arr = np.array(Image.open(f).convert('RGB'))
        lm = extract(arr)
        if lm is not None:
            X.append(lm); y.append(li); src.append(f.name)
    print(label, len(X), flush=True)

Xv, yv = [], []
df = pq.read_table('crossval_data/val.parquet', columns=['image', 'label']).to_pandas()
missed = 0
for _, row in df.iterrows():
    arr = np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB'))
    lm = extract(arr)
    if lm is None:
        missed += 1
        continue
    Xv.append(lm); yv.append(int(row['label']))
print('val kept', len(Xv), 'missed', missed)

np.savez_compressed('model_assets/landmarks_asl.npz', X=np.array(X), y=np.array(y),
                    Xv=np.array(Xv), yv=np.array(yv))
print('saved')
