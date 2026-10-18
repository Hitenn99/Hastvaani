"""
extract_pose.py
-----------------
Adds MediaPipe POSE landmarks (body: face, shoulders, elbows, wrists...) to the per-frame
hand landmarks already extracted by extract_landmarks.py, so the sentence matcher can tell
WHERE the hands are relative to the signer's body (on the head vs. in front of the chest),
independent of how the camera framed them.

Decodes exactly the same frames as extract_landmarks.py (every 2nd frame, capped at
MAX_LEN) so index t here lines up with index t in the hand cache.

Per video -> tmp/isl_sentence_pose/<sentence>/<video_stem>.npz with
  pose (T,33,4) = x, y, z, visibility (all zero where no body was found)
  pm   (T,)     = 1 if a body was found in that frame
  wh   (2,)     = frame width, height in pixels (x/y are normalised to these)
Resumable. A final pass aggregates hands + pose into model_assets/sentences/sequences_pose.npz.

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/extract_pose.py
"""
import multiprocessing as mp_proc
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from extract_landmarks import CACHE as HAND_CACHE, MAX_LEN, STEP, list_jobs, safe_name

CACHE = pathlib.Path('tmp/isl_sentence_pose')
OUT = pathlib.Path('model_assets/sentences/sequences_pose.npz')


def process_one(args):
    sentence, path = args
    out_dir = CACHE / sentence
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f'{safe_name(path)}.npz'
    if cache_file.exists():
        return sentence, path.name, 'cached'

    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.3,
    ))
    cap = cv2.VideoCapture(str(path))
    pose, pm = [], []
    wh = np.zeros(2, np.float32)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % STEP:
                continue
            if len(pose) >= MAX_LEN:
                break
            wh[:] = (frame.shape[1], frame.shape[0])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            p = np.zeros((33, 4), np.float32)
            if res.pose_landmarks:
                p[:] = [[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks[0]]
            pose.append(p)
            pm.append(1.0 if res.pose_landmarks else 0.0)
    except Exception as e:
        print('  warning', path, repr(e), flush=True)
    finally:
        cap.release()
        det.close()

    np.savez(cache_file, pose=np.array(pose, np.float32).reshape(-1, 33, 4),
             pm=np.array(pm, np.float32), wh=wh)
    return sentence, path.name, f'{len(pose)} frames, body in {int(sum(pm))}'


if __name__ == '__main__':
    sentences, jobs = list_jobs()
    print(f'{len(sentences)} sentences, {len(jobs)} videos', flush=True)
    with mp_proc.Pool(3, maxtasksperchild=8) as pool:
        for i, (s, name, status) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            print(f'[{i}/{len(jobs)}] {s} / {name}: {status}', flush=True)

    # ---- aggregate (same order/filtering as extract_landmarks.py's aggregate) ----
    P_all, M_all, POSE_all, PM_all, WH_all, lens, labels, sources = [], [], [], [], [], [], [], []
    for si, s in enumerate(sentences):
        for f in sorted((HAND_CACHE / s).glob('*.npz')):
            d = np.load(f)
            if len(d['P']) < 4:
                continue
            q = np.load(CACHE / s / f.name)
            n = min(len(d['P']), MAX_LEN)
            assert len(q['pose']) >= n, (f, len(q['pose']), n)
            p_pad = np.zeros((MAX_LEN, 2, 21, 3), np.float32)
            m_pad = np.zeros((MAX_LEN, 2), np.float32)
            po_pad = np.zeros((MAX_LEN, 33, 4), np.float32)
            pm_pad = np.zeros(MAX_LEN, np.float32)
            p_pad[:n], m_pad[:n] = d['P'][:n], d['M'][:n]
            po_pad[:n], pm_pad[:n] = q['pose'][:n], q['pm'][:n]
            P_all.append(p_pad); M_all.append(m_pad); POSE_all.append(po_pad); PM_all.append(pm_pad)
            WH_all.append(q['wh']); lens.append(n); labels.append(si); sources.append(f.stem)
    np.savez_compressed(
        OUT, P=np.array(P_all), M=np.array(M_all), pose=np.array(POSE_all), pm=np.array(PM_all),
        wh=np.array(WH_all), length=np.array(lens), y=np.array(labels),
        sentences=np.array(sentences), source=np.array(sources),
    )
    print('saved', OUT, 'sequences:', len(lens))
