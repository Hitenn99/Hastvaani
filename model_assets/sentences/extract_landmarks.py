"""
extract_landmarks.py
-----------------------
Extracts a per-frame, two-hand landmark SEQUENCE from every video in the ISL-CSLTR
sentence-level corpus (isl_sentences_data/), for training a sign-language *sentence*
classifier (a fixed catalog of ~101 known sentences, not open-vocabulary translation).

Completely separate from the alphabet/digit pipelines: nothing here touches
model_assets/train_asl2_model.py, train_isl_landmarks.py, or the digit rules.

For each video:
  - every 2nd decoded frame (source is 25fps; ~12.5fps is plenty for these ~3-5s clips)
  - MediaPipe HandLandmarker (up to 2 hands) -> (2, 21, 3) landmarks + presence mask,
    hands sorted by wrist x (same convention as landmarks_isl.py)
Saved per video to tmp/isl_sentence_lm/<sentence>/<video_stem>.npz (resumable: a run
that crashes partway just skips files that already have a cache entry). A final pass
aggregates everything into model_assets/sentences/sequences.npz.

Run from the repo root:  .venv_tfjs/Scripts/python.exe model_assets/sentences/extract_landmarks.py
"""
import multiprocessing as mp_proc
import pathlib
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = pathlib.Path('isl_sentences_data/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level')
CACHE = pathlib.Path('tmp/isl_sentence_lm')
OUT = pathlib.Path('model_assets/sentences/sequences.npz')
STEP = 2  # decode every 2nd frame
MAX_LEN = 90  # ~7s at 12.5fps; these clips are ~3-5s so this is a generous cap


def safe_name(p: pathlib.Path) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', p.stem)


def list_jobs():
    sentences = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    jobs = []
    for s in sentences:
        for f in sorted((ROOT / s).glob('*')):
            if f.suffix.lower() in ('.mp4', '.mov', '.avi'):
                jobs.append((s, f))
    return sentences, jobs


def process_one(args):
    sentence, path = args
    out_dir = CACHE / sentence
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f'{safe_name(path)}.npz'
    if cache_file.exists():
        return sentence, path.name, 'cached'

    lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='model_assets/hand_landmarker.task'),
        running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3,
    ))
    cap = cv2.VideoCapture(str(path))
    P, M = [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % STEP:
                continue
            if len(P) >= MAX_LEN:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            hands = [np.array([[l.x, l.y, l.z] for l in h], dtype=np.float32) for h in res.hand_landmarks]
            hands.sort(key=lambda h: h[0, 0])
            p = np.zeros((2, 21, 3), np.float32)
            m = np.zeros(2, np.float32)
            for k, h in enumerate(hands[:2]):
                p[k] = h
                m[k] = 1
            P.append(p)
            M.append(m)
    except Exception as e:  # corrupt frame, codec hiccup, etc: keep whatever was decoded so far
        print('  warning', path, repr(e), flush=True)
    finally:
        cap.release()

    if len(P) < 4:  # too short / unreadable to be useful
        np.savez(cache_file, P=np.zeros((0, 2, 21, 3), np.float32), M=np.zeros((0, 2), np.float32))
        return sentence, path.name, f'skip ({len(P)} frames)'

    np.savez(cache_file, P=np.array(P, np.float32), M=np.array(M, np.float32))
    return sentence, path.name, f'{len(P)} frames'


if __name__ == '__main__':
    sentences, jobs = list_jobs()
    print(f'{len(sentences)} sentences, {len(jobs)} videos')
    with mp_proc.Pool(2, maxtasksperchild=8) as pool:
        for i, (s, name, status) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            print(f'[{i}/{len(jobs)}] {s} / {name}: {status}', flush=True)

    # ---- aggregate ----
    seqs, masks, lens, labels, sources = [], [], [], [], []
    for si, s in enumerate(sentences):
        for f in sorted((CACHE / s).glob('*.npz')):
            d = np.load(f)
            if len(d['P']) < 4:
                continue
            p_pad = np.zeros((MAX_LEN, 2, 21, 3), np.float32)
            m_pad = np.zeros((MAX_LEN, 2), np.float32)
            n = min(len(d['P']), MAX_LEN)
            p_pad[:n] = d['P'][:n]
            m_pad[:n] = d['M'][:n]
            seqs.append(p_pad)
            masks.append(m_pad)
            lens.append(n)
            labels.append(si)
            sources.append(f.stem)
    np.savez_compressed(
        OUT,
        P=np.array(seqs, np.float32),
        M=np.array(masks, np.float32),
        length=np.array(lens),
        y=np.array(labels),
        sentences=np.array(sentences),
        source=np.array(sources),
    )
    print('saved', OUT, 'sequences:', len(seqs))
