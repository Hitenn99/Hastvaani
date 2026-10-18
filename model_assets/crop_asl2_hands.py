"""
crop_asl2_hands.py
--------------------
Root-cause fix for the ASL2 training collapse (loss stuck exactly at
ln(26)=3.258 regardless of learning rate, architecture depth, normalization,
dropout, or optimizer -- verified via 6+ isolated ablation experiments,
including a plain from-scratch numpy/Keras run with no custom layers at all).

The raw asl_sign_languages_alphabets_v03 photos have WILDLY inconsistent hand
scale and position -- some fill the whole frame, some show a hand small in
the corner of a much larger scene, at resolutions from 215x214 to 447x407.
Naively resizing each whole photo to 64x64 forces a from-scratch CNN to learn
scale/position invariance across huge variation from only ~400 examples per
class, which turned out to be too hard: even a plain CNN with no
regularization or custom layers stayed at chance accuracy for 25+ epochs on
this raw data. It never overfit, meaning the raw images have no learnable
structure without extra cropping.

Fix: run the same MediaPipe HandLandmarker + bbox logic used at inference
time (see src/handTracking.ts / src/config.ts in the webapp) on each RAW
training photo, crop to the detected hand with the same 12% padding, and
train on THOSE crops instead. This both removes the scale/position variance
problem and makes the training distribution match the deployment distribution
exactly (the whole point of retraining on a new dataset in the first place).
Images where no hand is detected are dropped and reported.
"""
import os
import glob

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BBOX_PADDING = 0.12  # matches webapp/src/config.ts MODE_CONFIGS.asl.bboxPadding
SRC_DIR = 'asl_data/by_class'
OUT_DIR = 'asl_data/by_class_cropped'

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
    classes = sorted(os.listdir(SRC_DIR))
    total = 0
    detected = 0
    for cls in classes:
        os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)
        files = sorted(glob.glob(os.path.join(SRC_DIR, cls, '*.jpg')))
        for f in files:
            total += 1
            img_bgr = cv2.imread(f)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)
            if not result.hand_landmarks:
                continue
            h, w = img_rgb.shape[:2]
            x, y, bw, bh = compute_square_bbox(result.hand_landmarks[0], w, h, BBOX_PADDING)
            if bw <= 4 or bh <= 4:
                continue
            crop = img_bgr[y:y + bh, x:x + bw]
            out_path = os.path.join(OUT_DIR, cls, os.path.basename(f))
            cv2.imwrite(out_path, crop)
            detected += 1
        print(f'{cls}: {sum(1 for _ in glob.glob(os.path.join(OUT_DIR, cls, "*.jpg")))} / {len(files)} cropped')

    print(f'TOTAL: {detected} / {total} images had a detectable hand and were cropped')


if __name__ == '__main__':
    main()
