"""
check_pose_parity.py
----------------------
Dumps a few real clips' RAW detections (as the browser would see them), plus the Python
reference features (pose_features.clip_features) and DTW distances to some shipped bank
entries, to tmp/pose_parity.json -- then webapp/scripts/check-pose-parity.mts runs the TS port
(features.ts clipFeatures + dtw.ts) on the same input and compares.

Run from the repo root:
  .venv_tfjs/Scripts/python.exe model_assets/sentences/check_pose_parity.py
  node --experimental-strip-types webapp/scripts/check-pose-parity.mts
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_lab import dtw_nb  # noqa: E402
import pose_features  # noqa: E402  (v3, ISL bank)
import pose_features_v2  # noqa: E402  (v2, ASL bank)

d = np.load('model_assets/sentences/sequences_pose.npz', allow_pickle=True)
rng = np.random.default_rng(0)
clips = []
for base, mod in (('/sentence_model', pose_features), ('/sentence_model_asl', pose_features_v2)):
  meta = json.load(open(f'webapp/public{base}/dtw_meta.json'))
  bank = np.fromfile(f'webapp/public{base}/dtw_bank.bin', np.float32).reshape(-1, meta['featureLen'])
  clip_features, BAND = mod.clip_features, mod.BAND
  n_before = len(clips)
  for i in rng.choice(len(d['y']), 8, replace=False):
      n = int(d['length'][i])
      P, M, pose, pm = d['P'][i, :n], d['M'][i, :n], d['pose'][i, :n], d['pm'][i, :n]
      W, H = (float(v) for v in d['wh'][i])
      if len(clips) == n_before + 7:  # also exercise the no-body fallback path
          pm = np.zeros_like(pm)
      frames = []
      for t in range(n):
          hands = [[{'x': float(a), 'y': float(b), 'z': float(c)} for a, b, c in P[t, h]] for h in range(2) if M[t, h]]
          body = [{'x': float(a), 'y': float(b), 'z': float(c)} for a, b, c, _ in pose[t]] if pm[t] else None
          if hands or body:
              frames.append({'hands': hands, 'pose': body})
      # the browser never sees frames with nothing detected, so the reference drops them too
      keep = (M.sum(-1) > 0) | (pm > 0)
      f = clip_features(P[keep], M[keep], pose[keep], pm[keep], W, H)
      entries = rng.choice(len(meta['entries']), 6, replace=False).tolist()
      dists = []
      for e in entries:
          m = meta['entries'][e]
          ref = bank[m['offset']:m['offset'] + m['length']].astype(np.float64)
          dists.append(float(dtw_nb(f.astype(np.float32).astype(np.float64), ref, BAND)))
      clips.append({'base': base, 'spec': meta['features'], 'W': W, 'H': H, 'frames': frames, 'features': f.tolist(), 'entries': entries, 'dists': dists})
json.dump(clips, open('tmp/pose_parity.json', 'w'))
print('wrote tmp/pose_parity.json:', [len(c['features']) for c in clips], 'feature frames')
