"""
export_dtw_bank.py
---------------------
Exports the DTW k-NN reference bank for the browser: every recorded video's per-frame v3
feature sequence (hand shape + body pose, see pose_features.py), plus a mirrored copy of each,
concatenated into one binary blob with an index. No model weights -- classification at
query time is "which reference sequence is closest by DTW" (webapp/src/sentences/dtw.ts).

Leave-one-out (eval_dtw_pose.py): v1 hands only 24.5% top-1, v2 31.7%, v3 27.7%. But on the
cross-setup test (eval_cross.py, the realistic "new signer" case) v2 was ~11% and v3 (this) is
17.2% top-1 / 31.3% top-3 / 36.9% top-5, which is why v3 ships; v3.1 (smoothing + arm velocity)
20.0 / 32.3 / 41.3%.

Output:
  webapp/public/sentence_model/dtw_bank.bin   -- all reference sequences' (T,FEATURE_LEN)
                                                   float32 frames, back-to-back
  webapp/public/sentence_model/dtw_meta.json  -- {featureLen, band, features (version + params the
                                                   browser must use), entries: [{offset, length,
                                                   classIndex}, ...]} (offset in frames)
  webapp/public/sentence_model/labels.json    -- class index -> sentence text

Needs model_assets/sentences/sequences_pose.npz (extract_landmarks.py, then extract_pose.py).
Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/export_dtw_bank.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, 'model_assets/sentences')
from pose_features import BAND, FEATURE_LEN, ISL_PARAMS, dataset_features  # noqa: E402


def main():
    d = np.load('model_assets/sentences/sequences_pose.npz', allow_pickle=True)
    y = d['y']
    sentences = [str(s) for s in d['sentences']]

    all_seqs = dataset_features(d) + dataset_features(d, mirror=True)
    all_y = np.concatenate([y, y])

    out_dir = 'webapp/public/sentence_model'
    os.makedirs(out_dir, exist_ok=True)

    meta = []
    offset = 0
    with open(f'{out_dir}/dtw_bank.bin', 'wb') as f:
        for seq, cls in zip(all_seqs, all_y):
            if len(seq) == 0:
                continue
            f.write(seq.astype(np.float32).tobytes())
            meta.append({'offset': offset, 'length': len(seq), 'classIndex': int(cls)})
            offset += len(seq)

    with open(f'{out_dir}/dtw_meta.json', 'w') as f:
        json.dump({'featureLen': FEATURE_LEN, 'band': BAND, 'features': {'version': 3, **ISL_PARAMS}, 'entries': meta}, f)
    with open(f'{out_dir}/labels.json', 'w') as f:
        json.dump(sentences, f, ensure_ascii=False)

    bin_mb = os.path.getsize(f'{out_dir}/dtw_bank.bin') / 1024 / 1024
    print(f'wrote {len(meta)} reference sequences ({bin_mb:.1f} MB) + dtw_meta.json + labels.json to {out_dir}')


if __name__ == '__main__':
    main()
