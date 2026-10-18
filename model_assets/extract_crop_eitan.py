"""
extract_crop_eitan.py
------------------------
Extracts + hand-crops EitanG98/asl_letters' TRAIN split (a second,
independent ASL photo source -- different photographer/camera/background
than Marxulia) so it can be combined with the Marxulia training set. Using
two independent sources forces the model to learn features that hold across
cameras instead of one source's specific lighting/color signature -- see the
cross-dataset validation results in README.md (27.2% accuracy on
EitanG98's *validation* split despite 88.9% on Marxulia's own held-out split)
that showed the single-source model wasn't actually learning hand shape.

The EitanG98 *validation* split is deliberately NOT touched here -- it stays
completely held out, never trained on, so it remains a genuine, honest
final check of real-world generalization after retraining.
"""
import io
import os

import cv2
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
BBOX_PADDING = 0.12
OUT_DIR = 'asl_data/by_class_cropped'  # merge straight into the combined training pool

base_options = mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.4,
)
landmarker = vision.HandLandmarker.create_from_options(options)


def compute_square_bbox(landmarks, width, height, padding_fraction):
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * padding_fraction
    pad_y = (max_y - min_y) * padding_fraction
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    side = max(max_x - min_x, max_y - min_y)
    x = max(0, min(cx - side / 2, width - 1))
    y = max(0, min(cy - side / 2, height - 1))
    w = min(side, width - x)
    h = min(side, height - y)
    return int(x), int(y), int(w), int(h)


def main():
    tbl = pq.read_table('crossval_data/train.parquet', columns=['image', 'label'])
    df = tbl.to_pandas()
    print('EitanG98 train rows:', len(df))

    for cls in LABEL_NAMES:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

    counts = {}
    detected = 0
    for idx, row in df.iterrows():
        true_letter = LABEL_NAMES[row['label']]
        img_bytes = row['image']['bytes']
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_rgb = np.array(pil_img)
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            continue
        x, y, bw, bh = compute_square_bbox(result.hand_landmarks[0], w, h, BBOX_PADDING)
        if bw <= 4 or bh <= 4:
            continue

        crop_rgb = img_rgb[y:y + bh, x:x + bw]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        n = counts.get(true_letter, 0)
        # Prefix "eitan_" so these don't collide with Marxulia's 00000.jpg-style names.
        out_path = os.path.join(OUT_DIR, true_letter, f'eitan_{n:05d}.jpg')
        cv2.imwrite(out_path, crop_bgr)
        counts[true_letter] = n + 1
        detected += 1

        if idx % 100 == 0:
            print('progress', idx, '/', len(df))

    print(f'TOTAL: {detected}/{len(df)} EitanG98 train images cropped and merged into {OUT_DIR}')
    print('per-class counts:', counts)


if __name__ == '__main__':
    main()
