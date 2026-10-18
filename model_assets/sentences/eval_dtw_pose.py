"""Leave-one-out evaluation of the shipped v2 (hands + pose) DTW features. Run from the repo root.
Same protocol as eval_dtw_full.py (every video vs. all OTHER videos + their mirrors)."""
import json
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from dtw_exp_pose import load_pose  # noqa: E402
from dtw_lab import distance_matrix, score  # noqa: E402
from pose_features import BAND, dataset_features  # noqa: E402

if __name__ == '__main__':
    d = load_pose()
    D = distance_matrix(dataset_features(d), dataset_features(d, mirror=True), BAND)
    r = score(D, d['y'])
    json.dump({'top1': r[0], 'top3': r[1], 'top5': r[2], 'n': int(len(d['y'])), 'features': 'v2 hands+pose'},
              open('model_assets/sentences/dtw_result.json', 'w'), indent=2)
