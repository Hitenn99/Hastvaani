"""
make_sample_clips.py
-----------------------
Picks one demo video per sentence (the largest file in that sentence's folder, which in
this dataset correlates with higher resolution / clearer footage) and re-encodes it small
for the web: /sentence_model labels.json order -> webapp/public/sentence_clips/<index>.mp4
(H.264, scaled down, no audio). These are reference clips for the Learn-style "how do I
sign this" browser on the Sentence Builder page -- nothing here touches the trained model.

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/sentences/make_sample_clips.py
"""
import json
import pathlib
import subprocess
import sys

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = pathlib.Path('isl_sentences_data/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level')
OUT = pathlib.Path('webapp/public/sentence_clips')
OUT.mkdir(parents=True, exist_ok=True)

with open('model_assets/sentences/labels.json') as f:
    sentences = json.load(f)

total_bytes = 0
for i, sentence in enumerate(sentences):
    files = sorted((ROOT / sentence).glob('*'))
    files = [f for f in files if f.suffix.lower() in ('.mp4', '.mov', '.avi')]
    if not files:
        print(f'[{i}] {sentence}: NO VIDEO FOUND', flush=True)
        continue
    src = max(files, key=lambda f: f.stat().st_size)
    dst = OUT / f'{i}.mp4'
    if dst.exists() and '--force' not in sys.argv:
        total_bytes += dst.stat().st_size
        continue
    cmd = [
        FFMPEG, '-y', '-i', str(src),
        '-vf', 'scale=-2:320', '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30',
        '-movflags', '+faststart',
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        print(f'[{i}] {sentence}: FFMPEG FAILED ({src.name})\n{r.stderr[-500:]}', flush=True)
        continue
    size = dst.stat().st_size
    total_bytes += size
    print(f'[{i}] {sentence}: {src.name} -> {size/1024:.0f} KB', flush=True)

print(f'\ntotal: {total_bytes/1024/1024:.1f} MB across {len(sentences)} sentences')
