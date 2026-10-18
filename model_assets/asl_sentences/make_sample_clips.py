"""
make_sample_clips.py
-----------------------
ASL reference clips for the Sentence Builder's "how do I sign this" browser: for every
catalog sentence, one real WLASL clip per word (the largest file for that word, which tends
to be the clearest), joined in ASL gloss order and re-encoded small ->
webapp/public/sentence_clips_asl/<index>.mp4 (H.264, 320px tall, silent). The ISL equivalent
is model_assets/sentences/make_sample_clips.py.

Run from the repo root: .venv_tfjs/Scripts/python.exe model_assets/asl_sentences/make_sample_clips.py
"""
import json
import pathlib
import subprocess
import sys

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
WLASL = pathlib.Path('asl_sentences_data/wlasl')
OUT = pathlib.Path('webapp/public/sentence_clips_asl')
OUT.mkdir(parents=True, exist_ok=True)

catalog = json.load(open('model_assets/asl_sentences/catalog.json'))
meta = {g['gloss']: g['instances'] for g in json.load(open(WLASL / 'WLASL_v0.3.json'))}


def best_clip(word):
    files = [WLASL / 'videos' / f"{i['video_id']}.mp4" for i in meta[word]]
    files = [f for f in files if f.exists()]
    return max(files, key=lambda f: f.stat().st_size)


total = 0
for i, e in enumerate(catalog):
    dst = OUT / f'{i}.mp4'
    if dst.exists() and '--force' not in sys.argv:
        total += dst.stat().st_size
        continue
    srcs = [best_clip(w) for w in e['gloss']]
    cmd = [FFMPEG, '-y']
    for s in srcs:
        cmd += ['-i', str(s)]
    # every clip -> same 426x320 canvas / 25 fps / pixel format, then concatenate
    chains = ''.join(
        f'[{k}:v]scale=426:320:force_original_aspect_ratio=decrease,pad=426:320:(ow-iw)/2:(oh-ih)/2,'
        f'fps=25,setsar=1,format=yuv420p[v{k}];' for k in range(len(srcs)))
    chains += ''.join(f'[v{k}]' for k in range(len(srcs))) + f'concat=n={len(srcs)}:v=1:a=0[out]'
    cmd += ['-filter_complex', chains, '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '30', '-movflags', '+faststart', str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        print(f'[{i}] {e["sentence"]}: FFMPEG FAILED\n{r.stderr[-600:]}', flush=True)
        continue
    total += dst.stat().st_size
    print(f'[{i}] {e["sentence"]}: {" + ".join(e["gloss"])} -> {dst.stat().st_size/1024:.0f} KB', flush=True)

print(f'\ntotal: {total/1024/1024:.1f} MB across {len(catalog)} sentences')
