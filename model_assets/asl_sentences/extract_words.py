"""
extract_words.py
------------------
ASL sentence builder, step 1: per-frame two-hand + body-pose landmarks for every WLASL video
of every word used by catalog.json (ASL sentences are built from these word clips -- see
build_bank.py; there is no public ASL dataset of fixed sentences recorded several times, the
format the ISL sentence builder uses).

Same detection settings and frame sampling (~12.5 fps) as the ISL pipeline
(model_assets/sentences/extract_landmarks.py + extract_pose.py), so the resulting features
are directly comparable. WLASL videos have varying fps, so frames are sampled by TIME
(one every 80 ms) rather than "every 2nd frame".

Per video -> tmp/asl_word_lm/<gloss>/<video_id>.npz with P (T,2,21,3), M (T,2),
pose (T,33,4), pm (T,), wh (2,), signer. Resumable. Run from the repo root:
  .venv_tfjs/Scripts/python.exe model_assets/asl_sentences/extract_words.py
"""
import json
import multiprocessing as mp_proc
import pathlib

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WLASL = pathlib.Path('asl_sentences_data/wlasl')
CACHE = pathlib.Path('tmp/asl_word_lm')
CATALOG = pathlib.Path('model_assets/asl_sentences/catalog.json')
SAMPLE_MS = 80.0
MAX_LEN = 90


def list_jobs():
    words = sorted({w for e in json.load(open(CATALOG)) for w in e['gloss']})
    meta = {g['gloss']: g['instances'] for g in json.load(open(WLASL / 'WLASL_v0.3.json'))}
    jobs = []
    for w in words:
        for ins in meta[w]:
            path = WLASL / 'videos' / f"{ins['video_id']}.mp4"
            if path.exists():
                jobs.append((w, path, ins['signer_id'], ins['frame_start'], ins['frame_end']))
    return words, jobs


def process_one(args):
    word, path, signer, f_start, f_end = args
    out_dir = CACHE / word
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f'{path.stem}.npz'
    if cache_file.exists():
        return word, path.name, 'cached'

    hands_det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3,
    ))
    pose_det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.3,
    ))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    P, M, POSE, PM = [], [], [], []
    wh = np.zeros(2, np.float32)
    idx, next_t = 0, 0.0
    try:
        while len(P) < MAX_LEN:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1  # WLASL frame numbers are 1-based
            if idx < f_start or (f_end != -1 and idx > f_end):
                continue
            t = (idx - max(f_start, 1)) * 1000.0 / fps
            if t + 1e-6 < next_t:
                continue
            next_t += SAMPLE_MS
            wh[:] = (frame.shape[1], frame.shape[0])
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = hands_det.detect(img)
            hands = sorted((np.array([[l.x, l.y, l.z] for l in h], np.float32) for h in res.hand_landmarks),
                           key=lambda h: h[0, 0])
            p = np.zeros((2, 21, 3), np.float32)
            m = np.zeros(2, np.float32)
            for k, h in enumerate(hands[:2]):
                p[k], m[k] = h, 1
            pres = pose_det.detect(img)
            q = np.zeros((33, 4), np.float32)
            if pres.pose_landmarks:
                q[:] = [[l.x, l.y, l.z, l.visibility] for l in pres.pose_landmarks[0]]
            P.append(p); M.append(m); POSE.append(q); PM.append(1.0 if pres.pose_landmarks else 0.0)
    except Exception as e:
        print('  warning', path, repr(e), flush=True)
    finally:
        cap.release()
        hands_det.close()
        pose_det.close()

    np.savez(cache_file, P=np.array(P, np.float32).reshape(-1, 2, 21, 3), M=np.array(M, np.float32).reshape(-1, 2),
             pose=np.array(POSE, np.float32).reshape(-1, 33, 4), pm=np.array(PM, np.float32), wh=wh,
             signer=np.int32(signer))
    return word, path.name, f'{len(P)} frames, hands in {int((np.array(M).sum(-1) > 0).sum()) if M else 0}, body in {int(sum(PM))}'


if __name__ == '__main__':
    words, jobs = list_jobs()
    print(f'{len(words)} words, {len(jobs)} videos', flush=True)
    with mp_proc.Pool(3, maxtasksperchild=16) as pool:
        for i, (w, name, status) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            print(f'[{i}/{len(jobs)}] {w} / {name}: {status}', flush=True)
    print('done', flush=True)
