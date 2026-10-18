"""
crossval_asl2.py
------------------
Independent generalization check: evaluates the trained ASL2 model
(model_assets/kaggle_asl2_model.keras) against EitanG98/asl_letters, a
completely different Hugging Face ASL dataset it has never seen (different
photographer(s), different camera, different background/lighting than the
Marxulia dataset it was trained on). Runs the exact same MediaPipe
hand-detection + crop pipeline the live app uses, so this measures the whole
pipeline's real generalization, not just the classifier in isolation.
"""
import io
import json

import cv2
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

IMG_SIZE = 96
BBOX_PADDING = 0.12
LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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


import sys
sys.path.insert(0, 'model_assets')
from train_asl2_model import SpatialAttention

model = tf.keras.models.load_model(
    'model_assets/kaggle_asl2_model.keras', custom_objects={'SpatialAttention': SpatialAttention}
)

print('Loading cross-validation dataset...')
tbl = pq.read_table('crossval_data/val.parquet', columns=['image', 'label'])
df = tbl.to_pandas()
print('total rows:', len(df))

per_class_total = {l: 0 for l in LABEL_NAMES}
per_class_correct = {l: 0 for l in LABEL_NAMES}
per_class_nohand = {l: 0 for l in LABEL_NAMES}
confusion = {}

rng = np.random.default_rng(42)
# Evaluate up to 10 random images per class for speed.
by_label = {}
for idx, row in df.iterrows():
    by_label.setdefault(row['label'], []).append(idx)

for label_idx, indices in by_label.items():
    true_letter = LABEL_NAMES[label_idx]
    sample_indices = rng.choice(indices, size=min(10, len(indices)), replace=False)
    for idx in sample_indices:
        row = df.iloc[idx]
        img_bytes = row['image']['bytes']
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_rgb = np.array(pil_img)
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        per_class_total[true_letter] += 1
        if not result.hand_landmarks:
            per_class_nohand[true_letter] += 1
            continue

        x, y, bw, bh = compute_square_bbox(result.hand_landmarks[0], w, h, BBOX_PADDING)
        if bw <= 4 or bh <= 4:
            per_class_nohand[true_letter] += 1
            continue

        crop = img_rgb[y:y + bh, x:x + bw]
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        arr = crop_resized.astype(np.float32) / 255.0
        arr = arr[None, ...]
        pred = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(pred))
        pred_letter = LABEL_NAMES[pred_idx]
        confidence = float(pred[pred_idx])

        if pred_letter == true_letter:
            per_class_correct[true_letter] += 1
        confusion.setdefault(true_letter, {}).setdefault(pred_letter, 0)
        confusion[true_letter][pred_letter] += 1

print()
print(f'{"Letter":<8}{"Correct":<10}{"NoHand":<10}{"Total":<8}{"Accuracy"}')
total_correct, total_evaluated, total_nohand = 0, 0, 0
for l in LABEL_NAMES:
    t = per_class_total[l]
    c = per_class_correct[l]
    nh = per_class_nohand[l]
    evaluated = t - nh
    acc = c / evaluated if evaluated > 0 else float('nan')
    print(f'{l:<8}{c:<10}{nh:<10}{t:<8}{acc:.2f}' if evaluated > 0 else f'{l:<8}{c:<10}{nh:<10}{t:<8}N/A')
    total_correct += c
    total_evaluated += evaluated
    total_nohand += nh

print()
print(f'OVERALL: {total_correct}/{total_evaluated} correct where a hand was detected ({total_correct/total_evaluated*100:.1f}%)')
print(f'Hand-detection failures: {total_nohand}/{sum(per_class_total.values())}')
print()
print('Confusion (true -> predicted counts, only showing mistakes):')
for true_letter, preds in confusion.items():
    mistakes = {k: v for k, v in preds.items() if k != true_letter}
    if mistakes:
        print(f'  {true_letter} -> {mistakes}')

with open('model_assets/crossval_results.json', 'w') as f:
    json.dump({
        'overall_accuracy': total_correct / total_evaluated if total_evaluated else None,
        'total_correct': total_correct,
        'total_evaluated': total_evaluated,
        'total_nohand': total_nohand,
        'confusion': confusion,
    }, f, indent=2)
