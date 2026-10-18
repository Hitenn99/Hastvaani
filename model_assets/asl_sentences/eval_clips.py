"""Whole-clip (browser-style) check: runs hands+pose detection over each concatenated reference
clip webapp/public/sentence_clips_asl/<i>.mp4 as ONE query (body frame + trim over the whole clip,
exactly like the page), then classifies it against the exported ASL bank. Pessimistic in one
way (the clip returns to rest between words, which the bank never contains) and optimistic in
another (its word clips are also in the bank). Run from the repo root."""
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import dtw_nb  # noqa: E402
from pose_features import BAND, clip_features  # noqa: E402

hd = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3))
pd = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='model_assets/pose_landmarker_lite.task'),
    running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.3))
meta = json.load(open('webapp/public/sentence_model_asl/dtw_meta.json'))
bank = np.fromfile('webapp/public/sentence_model_asl/dtw_bank.bin', np.float32).reshape(-1, meta['featureLen']).astype(np.float64)
labels = json.load(open('webapp/public/sentence_model_asl/labels.json'))
ref_y = np.array([e['classIndex'] for e in meta['entries']])

hits = np.zeros(3)
idxs = range(len(labels)) if len(sys.argv) < 2 else range(0, len(labels), int(sys.argv[1]))
n = 0
for i in idxs:
    cap = cv2.VideoCapture(f'webapp/public/sentence_clips_asl/{i}.mp4')
    P, M, POSE, PM, k = [], [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        k += 1
        if k % 2: continue
        H, W = fr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = hd.detect(img); q = pd.detect(img)
        hands = sorted((np.array([[l.x, l.y, l.z] for l in h], np.float32) for h in r.hand_landmarks), key=lambda h: h[0, 0])
        p = np.zeros((2, 21, 3), np.float32); m = np.zeros(2, np.float32)
        for j, h in enumerate(hands[:2]): p[j], m[j] = h, 1
        po = np.zeros((33, 4), np.float32)
        if q.pose_landmarks: po[:] = [[l.x, l.y, l.z, l.visibility] for l in q.pose_landmarks[0]]
        P.append(p); M.append(m); POSE.append(po); PM.append(1.0 if q.pose_landmarks else 0.0)
    f = clip_features(np.array(P), np.array(M), np.array(POSE), np.array(PM), W, H)
    d = np.array([dtw_nb(f, bank[e['offset']:e['offset'] + e['length']], BAND) for e in meta['entries']])
    best = np.array([d[ref_y == c].min() for c in range(len(labels))])
    order = np.argsort(best)
    rank = int(np.where(order == i)[0][0])
    hits += [rank < 1, rank < 3, rank < 5]; n += 1
    print(f'[{i}] {labels[i]!r}: rank {rank}, top = {labels[order[0]]!r}', flush=True)
print(f'WHOLE-CLIP: n={n} top-1 {hits[0]/n:.3f} top-3 {hits[1]/n:.3f} top-5 {hits[2]/n:.3f}')
