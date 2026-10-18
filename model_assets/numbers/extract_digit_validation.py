"""
extract_digit_validation.py
------------------------------
Builds a small REAL-PHOTO validation set for the rule-based digit classifier
(webapp/src/numbers/digits.ts). Nothing here trains a model and nothing touches
the alphabet pipeline.

Sources where the sign coincides with the digit charts the user supplied:
  * Hemg/Indian_sign_language_dataset (numbers_data/isl_all.parquet), labels are
    class ids 0..8 = '1'..'9' of THAT dataset. Its conventions differ from the
    user's chart for several digits, so only verified matches are used:
        dataset '1'->1, '2'->2, '4'->4, '5'->5,
        dataset '3' (three fingers up, thumb holding little finger) -> chart 6,
        dataset '8' (thumb out + index + middle)                     -> chart 3
  * ASL alphabet landmarks (model_assets/landmarks_asl.npz, cross-source Eitan
    split): ASL letter O == digit 0, V/U == 2, W == 6, F == 9, B == 4.
Run from the repo root.
"""
import io

import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

PER_CLASS = 120
HEMG_MAP = {0: 1, 1: 2, 3: 4, 4: 5, 2: 6, 7: 3}  # hemg label id -> chart digit

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.5))

df = pq.read_table('numbers_data/isl_all.parquet').to_pandas()
P, y, src = [], [], []
for lab, digit in HEMG_MAP.items():
    idx = np.where(df['label'].values == lab)[0]
    idx = idx[np.linspace(0, len(idx) - 1, PER_CLASS).astype(int)]
    kept = 0
    for i in idx:
        im = np.array(Image.open(io.BytesIO(df.iloc[i]['image']['bytes'])).convert('RGB'))
        r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
        if len(r.hand_landmarks) == 1:
            P.append([[l.x, l.y, l.z] for l in r.hand_landmarks[0]]); y.append(digit); src.append('isl'); kept += 1
    print('ISL dataset label', lab, '-> digit', digit, kept, '/', PER_CLASS, flush=True)

d = np.load('model_assets/landmarks_asl.npz')
letter_map = {'O': 0, 'V': 2, 'U': 2, 'W': 6, 'F': 9, 'B': 4}
for L, digit in letter_map.items():
    ix = np.where(d['yv'] == ord(L) - 65)[0]
    for i in ix:
        P.append(d['Xv'][i].tolist()); y.append(digit); src.append('asl_letter_' + L)
    print('ASL letter', L, '-> digit', digit, len(ix), flush=True)

np.savez('numbers_data/digit_validation.npz', P=np.array(P, np.float32), y=np.array(y), src=np.array(src))
print('saved', len(P))
