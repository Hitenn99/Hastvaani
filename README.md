# Sign Language Recognizer

A browser-only web app that watches your webcam and recognizes **ASL (American)**
and **ISL (Indian)** alphabet signs in real time, showing the recognized letters
on screen and speaking them aloud. Everything — hand detection, the CNN, and
text-to-speech — runs client-side; there is no backend server.

Pipeline (per the *"Gesture Recognition in Sign Language Translation"* paper this
reproduces): **webcam → MediaPipe hand-landmark detection → crop to hand(s) →
CNN with spatial attention → smoothed classification → on-screen text + speech.**

## Project layout

```
webapp/                  the actual deployable app (Vite + TypeScript)
  src/                    app source
  public/tfjs_model_asl/  converted ASL model (TF.js graph model) + labels.json
  public/tfjs_model_isl/  converted ISL model (TF.js graph model) + labels.json
model_assets/             Python: Keras models + training/conversion scripts
asl_data/                 ASL training images (not committed — see "Rebuilding the models" below)
isl_data/                 ISL training images (not committed — see "Rebuilding the models" below)
.venv_tfjs/               Python 3.11 venv used for model conversion/training (not committed)
```

## Running locally

```bash
cd webapp
npm install
npm run dev
```

Open the printed `http://localhost:5173` URL, allow camera access, and pick
ASL or ISL with the toggle at the top.

## Building / deploying

```bash
cd webapp
npm run build     # outputs webapp/dist — a static site, nothing else needed
npm run preview   # sanity-check the production build locally
```

- **Vercel:** `vercel deploy` from `webapp/` (a `vercel.json` is already set up).
- **Netlify:** `netlify deploy` from `webapp/` (a `netlify.toml` is already set up).
- **GitHub Pages:** run `npm run build`, publish `webapp/dist`. If your Pages
  site lives under a subpath (`username.github.io/repo-name`), add a
  `vite.config.ts` with `export default { base: '/repo-name/' }` before
  building, so the model/asset URLs resolve correctly.

No environment variables or API keys are needed. MediaPipe's hand-tracking
model and WASM runtime are fetched from Google's/jsDelivr's public CDN on
first load (then cached by the browser) — this is the one piece that isn't
self-hosted, since bundling it would add tens of MB to the repo.

## The two models

### ASL (American Sign Language)

**This model was replaced after real-world testing showed it performing
badly.** The full story, in order:

1. **Originally provided pre-trained model** (`model_assets/kaggle_asl_model.keras`),
   trained on the real Kaggle ["Sign Language MNIST"](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
   dataset, reported 99.80% accuracy on Kaggle's own held-out test set. 24
   classes (`A–Y`, skipping `J`/`Z`). Converted to TF.js and verified
   numerically identical to the original (`max abs diff: 1.7e-5`) — the
   conversion itself was never the problem.
2. **A real user tested it on an actual webcam and it was unusable** — signs
   were recognized as the wrong letter almost every time, and the same wrong
   letter kept repeating. Two real bugs were found and fixed first: the live
   hand crop used 35% padding (way looser than Sign Language MNIST's
   edge-to-edge training crops, confirmed by pulling and inspecting real
   samples from the dataset), and the letter-locking logic dropped its lock
   on a single noisy frame, causing the same letter to be re-appended
   repeatedly. Both fixes helped, but accuracy was still bad — Sign Language
   MNIST's photos are grayscale, extremely tightly cropped, and have almost
   no lighting/background/skin-tone diversity, essentially a controlled-lab
   distribution a live webcam never resembles closely enough.
3. **The model was retrained from scratch** on a more realistic dataset,
   [`Marxulia/asl_sign_languages_alphabets_v03`](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03)
   (Hugging Face mirror of a Kaggle ASL-alphabet-style dataset, no login
   needed): 10,873 real photos, all 26 letters `A–Z` (including `J`/`Z` as
   static single-frame approximations — real ASL J/Z involve motion a photo
   can't capture, so treat those two letters as lower-confidence). Using the
   project's established CNN+GroupNorm+SpatialAttention architecture
   from-scratch on this data **completely failed** — loss stuck at
   `ln(26)=3.258` (uniform/chance prediction) regardless of learning rate,
   with or without the attention block, with or without dropout, even
   reproduced in a raw NumPy/Keras loop with zero custom layers. Root cause:
   the raw photos have wildly inconsistent hand scale and position (some
   fill the frame, some are a hand in the corner of a much larger scene, at
   resolutions from 215×214 to 447×407) — too much variation for a
   from-scratch CNN to learn from ~330 images/class. Fixing that (below) still
   collapsed at from-scratch training, so the from-scratch approach was
   abandoned entirely in favor of **transfer learning**.
4. **Fix, part 1 — consistent framing:** `model_assets/crop_asl2_hands.py`
   runs MediaPipe hand detection (the same detector + bbox logic the deployed
   app uses) on every raw training photo and crops to the hand region before
   training, so the model learns from the same framing it'll see live.
   8,514 of 10,873 photos had a detectable hand and were kept.
5. **Fix, part 2 — transfer learning:** even on the now-consistently-cropped
   images, a from-scratch CNN still collapsed. `model_assets/train_asl2_model.py`
   uses an **ImageNet-pretrained MobileNetV2** backbone (frozen, then its top
   ~30 layers fine-tuned at a low learning rate) with the same
   SpatialAttention block + classifier head as the rest of this project on
   top, 96×96 RGB input. This is what actually worked: **93.5% validation
   accuracy after the frozen-backbone phase alone**, finishing at **88.9%
   genuine held-out test accuracy** after fine-tuning (`model_assets/asl2_test_accuracy.txt`).
   Unlike the ISL number below, this dataset has real cross-photo diversity
   (not one recording session), so this figure is a meaningful, trustworthy
   estimate of generalization — not an artifact of data leakage.
- **Verified end-to-end**: after conversion, the TF.js model's output was
  compared against the original Python model's output on a real photo in
  Node — both predicted the same letter (`B`) with matching confidence
  (`0.9761028` vs `0.9761032`, diff `4.2e-7`) — and separately confirmed
  working through the *entire* live pipeline (webcam frame → MediaPipe hand
  detection → crop → model → predicted letter) in a headless-browser test
  with a real hand photo standing in for the camera.
- The originally-provided model files (`kaggle_asl_model.keras`,
  `kaggle_asl_labels.json`, `kaggle_model.py`) are kept in `model_assets/`
  for reference/provenance but are no longer used by the app.

### ISL (Indian Sign Language)

- **Trained from scratch** for this project (no pre-trained model was
  provided). The task's suggested Kaggle datasets required a Kaggle account
  and API token, which wasn't available in this environment; instead, the
  same underlying dataset (character-level ISL photos, matching
  `vaishnaviasonawane/indian-sign-language-dataset` on Kaggle) was pulled
  from its public, no-login Hugging Face mirror,
  [`Hemg/Indian_sign_language_dataset`](https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset)
  (42,745 real photos, 35 classes). Digits (1–9) were dropped as out of scope,
  leaving **31,945 images across the 26 letters A–Z**, extracted to
  `isl_data/by_class/<LETTER>/*.jpg` by `model_assets/extract.py`.
- These are genuinely **two-handed** signs (see `isl_data/sample_*.png`),
  photographed against a dark background — unlike ASL fingerspelling.
- Architecture: same proven pattern as ASL — Conv2D blocks + GroupNormalization
  + spatial attention + softmax (`model_assets/train_isl_model.py`) — sized up
  for 64×64 **RGB** input (downsized from the source 128×128 for practical
  CPU training time; there's no GPU in this environment).
- Trained with an 80/10/10 stratified train/val/test split (fixed seed),
  Adam + early stopping on validation accuracy (stopped at epoch 14; best
  validation accuracy 99.72% at epoch 10).
- **Real, held-out test accuracy: 100.0%** (`model_assets/isl_test_accuracy.txt`
  has the raw number). **This number is real but misleading, and you should
  not trust it as a measure of real-world accuracy** — see the next paragraph.
- Converted to TF.js the same verified way as ASL, via `model_assets/convert_isl_for_tfjs.py`.

**Why the ISL number is essentially meaningless despite being "real":** a
100% held-out test accuracy on a 26-class problem is a red flag, not a
success, so this was investigated rather than reported at face value. Checking
pixel differences between images *within* a class (`isl_data/by_class/A/`)
shows a mean absolute pixel difference of only ~2–15 (out of 255) between
frames — including frames 400+ apart in the sequence. In other words, each
letter's ~1,200 photos are not 1,200 independent examples; they're
consecutive frames from what looks like **one continuous recording of one
person, in one location, under one lighting setup**, per letter. A random
80/10/10 split therefore puts near-duplicate frames from the same recording
into train, validation, *and* test, so the model doesn't need to generalize
at all to score well on the held-out set — it only needs to recognize that
one session. This is a genuine, honestly-computed test-set number; it is just
not evidence the model will work on a different person's hand, background, or
lighting, which is exactly the situation a live webcam presents. Treat the
ASL model's studio-vs-webcam gap (next section) as the *optimistic* case, and
expect the ISL model's real-world accuracy gap to be considerably larger.

## Live-webcam accuracy vs. reported test accuracy

The ASL model's 88.9% test number comes from real, diverse photos (different
sessions, backgrounds, lighting) and is a genuinely meaningful estimate — but
it's still photos, not a live camera, so expect webcam accuracy to be
somewhat lower. The ISL model's 100% number is **not** trustworthy as a
real-world estimate at all — see the explanation above (single recording
session, near-zero cross-photo diversity). Both models will benefit from a
plain background and good, even lighting, which the app's guidance text
tells the user directly rather than hiding the gap.

## Per-frame recognition pipeline

1. Grab a video frame from `getUserMedia`.
2. Run MediaPipe `HandLandmarker` (`runningMode: "VIDEO"`) — 1 hand for ASL,
   up to 2 for ISL.
3. Compute the landmarks' bounding box (union of both hands for ISL), pad by
   12% (ASL) or 20% (ISL) on each side — matching each model's training-crop
   framing — then square it around its center.
4. Crop that region to an offscreen canvas, resize to the model's input size
   (96×96 for ASL, 64×64 for ISL).
5. Keep RGB, normalize to `[0, 1]` (the ASL model rescales internally to the
   `[-1, 1]` range its MobileNetV2 backbone expects — baked into the graph,
   so the JS side doesn't need model-specific preprocessing logic).
6. Run the TF.js graph model; take the softmax argmax.
7. Only accept predictions with confidence ≥ 60%.
8. Majority-vote over the last 8 accepted frames before "locking in" a
   letter, so single noisy frames don't flicker the display. The lock
   survives brief confidence dips (up to 10 consecutive missed frames)
   rather than resetting on the first one — an earlier version reset
   immediately, which caused the same letter to be re-appended over and over
   while a sign was held steady (e.g. `HHHHHH`). The same locked-in letter is
   only spoken once per ~2 seconds regardless.
9. Append newly locked-in letters to the on-screen word builder (with
   space/backspace/clear/speak-word controls).

Detection and inference run inside a `requestAnimationFrame` loop, not a
blocking synchronous loop, so the UI never freezes.

## Known limitations / out of scope (by design, per the original spec)

- ASL's `J` and `Z` are real ASL letters that require motion (a trace/hook
  shape drawn in the air); the model only sees a static photo, so treat
  predictions for those two letters as a best-effort approximation, less
  reliable than the other 24 letters.
- ISL scope is the static alphabet only — no dynamic words/phrases.
- No backend, no accounts/auth.
- MediaPipe's model/WASM files load from a CDN on first use (see above).

## Rebuilding the models yourself

Both conversion/training scripts live in `model_assets/` and were run against
a Python 3.11 venv (`.venv_tfjs/`) — **Python 3.11, not later**, since
`tensorflow` doesn't yet support newer versions on Windows at time of writing.

```bash
py -3.11 -m venv .venv_tfjs

# Install tensorflowjs without its declared deps, then add back only the ones
# actually needed for plain-Keras conversion:
./.venv_tfjs/Scripts/pip install --no-deps tensorflowjs==4.22.0
./.venv_tfjs/Scripts/pip install "tensorflow>=2.13.0,<3" tf-keras packaging~=23.1 six importlib_resources tensorflow-hub
./.venv_tfjs/Scripts/pip install jax jaxlib   # do NOT add flax -- see note below

# tensorflowjs unconditionally does `import tensorflow_decision_forests` at
# import time, purely to register some ops -- but the real package pulls in
# `ydf`, which pulls in `uvloop`, which flatly refuses to install on Windows
# ("uvloop does not support Windows at the moment"). A tiny empty stub module
# satisfies the import without needing any of that:
mkdir -p .venv_tfjs/Lib/site-packages/tensorflow_decision_forests
echo "# stub -- see README.md" > .venv_tfjs/Lib/site-packages/tensorflow_decision_forests/__init__.py
# (flax is also a declared dep, but it's only imported lazily inside a function
# body for a JAX-specific conversion path this project never uses, so it's
# skipped entirely -- installing it pulls in orbax-checkpoint, which ships
# test-fixture paths deep enough to exceed Windows' 260-char path limit.)

# ASL v1 (the ORIGINALLY PROVIDED, now-unused model -> TF.js; kept for reference):
./.venv_tfjs/Scripts/python model_assets/rebuild_for_tfjs.py

# ASL v2 (the model the app actually uses -- download, hand-crop, train, convert):
./.venv_tfjs/Scripts/pip install mediapipe opencv-python Pillow
curl -L "https://huggingface.co/api/datasets/Marxulia/asl_sign_languages_alphabets_v03/parquet/default/train/0.parquet" -o asl_data/train.parquet
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o model_assets/hand_landmarker.task
./.venv_tfjs/Scripts/python model_assets/extract_asl2.py         # unpacks to asl_data/by_class/
./.venv_tfjs/Scripts/python model_assets/crop_asl2_hands.py      # hand-crops to asl_data/by_class_cropped/ (8,514/10,873 kept)
./.venv_tfjs/Scripts/python model_assets/train_asl2_model.py     # MobileNetV2 transfer learning; reports real test accuracy
./.venv_tfjs/Scripts/python model_assets/convert_asl2_for_tfjs.py

# ISL (download the dataset, extract, train from scratch, then convert):
curl -L "https://huggingface.co/api/datasets/Hemg/Indian_sign_language_dataset/parquet/default/train/0.parquet" -o isl_data/train.parquet
./.venv_tfjs/Scripts/python model_assets/extract.py             # unpacks the alphabet classes to isl_data/by_class/
./.venv_tfjs/Scripts/python model_assets/train_isl_model.py     # trains + reports real (see caveat above) test accuracy
./.venv_tfjs/Scripts/python model_assets/convert_isl_for_tfjs.py
```

All of these scripts must be run **from the repo root** (not from inside
`model_assets/`), since they use paths relative to the trained model files
and datasets saved there. `convert_asl2_for_tfjs.py` writes to
`tfjs_model_asl_new/` at the repo root (to avoid clobbering the live model
before you've checked it) — move it to `webapp/public/tfjs_model_asl/`
yourself once you're happy with it.

## Update (2026-09-21): ASL model v3 (combined datasets)

The live ASL model is now trained on Marxulia + EitanG98 (train split) combined,
with heavier color/geometric augmentation (geometric aug applied per batch, not
per image) and a shorter, gentler fine-tune (last 10 backbone layers).

| | Same-source test | Cross-source (EitanG98 val, never trained on) |
|---|---|---|
| v1 (Marxulia only) | 88.9% | 27.2% |
| v3 (current, combined) | 73.0% | **53.2%** (125/235, 25 no-hand) |

The cross-source number is the honest generalization estimate. Expect live
webcam accuracy to be in that ballpark or lower; plain background and even
lighting help. Training now writes `asl2_checkpoint.keras` (best val acc) so a
crash doesn't lose progress, and redirects TEMP to `tmp/` (C: was full).
The previous live model is backed up at `tmp/tfjs_model_asl_backup/`.
Note: `convert_asl2_for_tfjs.py` expects `kaggle_asl2_model.keras` and
`kaggle_asl2_labels.json` in the repo root (training writes them there; they
were moved into `model_assets/` afterwards).

## Update 2 (2026-09-21): ASL now uses a hand-landmark classifier (live model)

The image CNN (v3) was slow on the CPU backend and only 53% cross-source. ASL
now classifies the 21 MediaPipe hand landmarks (already computed in the app)
with a tiny MLP -- invariant to background/lighting/skin tone, and runs as
plain JS matmuls (no TF.js, instant).

- Cross-source accuracy (EitanG98 val, never trained on, hand detected): **82.3%** (672 images); same-source holdout 91.8%.
- Pipeline: `landmarks_asl.py` -> `train_asl_landmarks.py` -> `export_asl_landmarks.py` (writes `webapp/public/asl_landmarks/model.json`). Feature normalization is duplicated in `train_asl_landmarks.py` and `webapp/src/model.ts` -- keep them in sync.
- The CNN files (`tfjs_model_asl/`) are kept but no longer used by the app. ISL is unchanged (still image CNN on CPU).

## Update 3 (2026-09-21): ISL now uses a two-hand landmark classifier too

Both modes now run a small MLP on MediaPipe hand landmarks in plain JS; TF.js
was removed from the app (bundle ~1 MB -> 164 kB). The old image CNNs
(`tfjs_model_asl/`, `tfjs_model_isl/`) remain only for reference.

- ISL features: up to 2 hands, sorted by wrist x, centered on the joint centroid,
  scaled by hand-pair size, 126 coords + 2 presence flags = 128 floats.
  Pipeline: `landmarks_isl.py` -> `train_isl_landmarks.py` (exports
  `webapp/public/isl_landmarks/model.json`). Feature code is duplicated in
  `train_isl_landmarks.py` and `webapp/src/model.ts`; parity was verified
  numerically (max prob diff 6e-8).
- **Accuracy caveat:** temporal-split test accuracy is 100% (last 18% of each
  letter's frames), but the dataset is one signer / one session, so this is NOT
  evidence of real-world accuracy. Landmarks remove background/lighting/skin-tone
  dependence, which should generalize better than the image CNN did, but it has
  not been measured on another signer. Signs follow the dataset's chart, not the
  official ISLRTC chart (they differ for several letters).

## Update 4 (2026-09-21): multi-page website UI (Tailwind CSS v4)

The app is now a 5-page site (Vite multi-page): `index.html` (home), `translate.html`
(the recognizer), `learn.html` (alphabet guide, data in `src/alphabet.ts`),
`how-it-works.html`, `about.html`. Shared header/footer live in `webapp/partials/`
and are injected by a small plugin in `webapp/vite.config.ts`. Styling is in
`webapp/src/style.css` (Tailwind theme tokens + a few components); page behavior
(mobile menu, scroll reveal, Learn page, copy button) is in `webapp/src/site.ts`.
The recognizer (`main.ts`, `model.ts`, `handTracking.ts`, `smoothing.ts`) was NOT
changed -- `translate.html` keeps every element id that `main.ts` reads.
If `npm run build` fails with "Cannot find native binding", run
`npm i -D @tailwindcss/oxide-win32-x64-msvc lightningcss-win32-x64-msvc` (npm optional-dependency bug).

Update 5: dark theme. Colors are semantic tokens in `src/style.css` (`paper`, `ink`, `surface`, `night`, ...);
`:root[data-theme="dark"]` overrides them. The theme is chosen by an inline script (injected in `<head>` by
`vite.config.ts`, so there is no flash), defaults to the OS preference, and the header button toggles + saves it in
localStorage. The logo sits on a white tile in dark mode because the emblem's navy would disappear on a dark page.

## Update 6 (2026-09-21): Math with signs (digits 0-9)

New page `webapp/math.html` (+ `src/math.ts`, `src/numbers/`): a calculator and a 10-question practice game
driven by signed digits, with on-screen operator buttons (+ - x / and % as remainder), a digit guide, and optional
per-user calibration. **Completely separate from the alphabet recogniser** (nothing in model.ts / main.ts / the A-Z models
was changed); it only shares the MediaPipe hand tracker.

- **No ML training for digits.** The user's charts show one drawing per digit, and the public digit datasets found on
  Hugging Face do not match those charts (Vedant3907 "Numbers-Sign-Language" is not real ASL numerals; Hemg's ISL digits 6-9
  differ from the chart). So digits are recognised by a rule-based classifier on landmark geometry
  (`src/numbers/digits.ts`): which fingers are extended, thumb out/tucked, and which fingertip the thumb touches.
  Per the charts the ASL and ISL handshapes for 0-9 coincide, so one classifier serves both.
- **Validation** (`model_assets/numbers/extract_digit_validation.py`, real photos where the sign matches the charts,
  872 samples): overall 97%. Per digit: 0 83% (n=23), 1 100%, 2 95%, 3 100%, 4 97%, 5 100%, 6 97%, 9 90% (n=29).
  **7 and 8 could not be validated** (no matching photos) -- they use the same thumb-touch mechanism as 6 and 9.
  Thresholds were tuned on this same set, so real-world numbers may be a bit lower.
- Optional calibration (`src/numbers/calibration.ts`): the user records their own samples in the browser (landmarks only,
  localStorage); a kNN vote is blended with the rules.
- Big data files used only for validation live in `numbers_data/` (parquets ~450 MB; safe to delete after extraction).

Update 7: separate ASL / ISL digit guides. The Math page's Digit guide now shows each language's own chart
(`webapp/public/signs/num_isl/0-9.png`, `num_asl/1-10.png`, cropped from the two charts supplied by the site owner with
`model_assets/numbers/crop_digit_charts.py`). ASL has no 0 in its chart (shown as the standard O handshape, marked as such) and
its 10 is a shaken thumbs-up (reference only, not recognised). The ISL chart is Fig. 1 of Ekbote & Joshi, "Indian sign language
recognition using ANN and SVM classifiers" (2017) -- confirm you may republish it. Recognition is still shared because the two
charts' handshapes for 1-5 and 7-9 coincide (ISL's own 0 and its 6 with the little finger folded under the thumb are the differences).

Update 8: How-it-works stats now show top-1 (image model 53%, landmark model 82%) and top-3 (landmark model 93.0%) on the
672 cross-source ASL photos (Eitan val, hand detected). Top-3 = correct letter among the model's 3 best guesses; 48 further
photos with no detected hand are excluded (counting them as wrong: 76.8% top-1, 86.8% top-3).

## Update 9 (2026-09-23): Sentence builder (ISL, GRU sequence model)

New page `webapp/sentence.html` (+ `src/sentence.ts`, `src/sentences/`): record a short clip of yourself
signing an ISL sentence, and it's matched against a **fixed catalog of 101 everyday ISL sentences** (isolated-
sentence recognition, not open-vocabulary translation -- there is no dataset or realistic budget for that).
Completely separate from the alphabet/digit code; only the shared MediaPipe hand tracker is reused.

- **Dataset:** [ISL-CSLTR](https://data.mendeley.com/datasets/kcmpdxky7p/1) (Ekbote-style corpus), 687 videos
  across 101 sentences, two native signers + 5 volunteers, ~5 recordings/sentence.
- **Pipeline:** `model_assets/sentences/extract_landmarks.py` (two-hand MediaPipe landmarks per frame, every
  video, resumable per-video cache in `tmp/isl_sentence_lm/`) -> `train_sentences.py` (2 stacked
  Bidirectional-GRU layers + Dense head, Keras `Masking`, heavy augmentation -- ~5 videos/class is very little
  data) -> `export_sentence_model.py` (dumps weights to `webapp/public/sentence_model/model.json`, transposed
  for the hand-written TS forward pass in `webapp/src/sentences/model.ts` -- no TF.js).
- **The GRU forward pass in TS was numerically verified against Keras before shipping** (see `model.ts`'s
  header comment): gate order/reset-after formula, Bidirectional weight order/merge, and -- the part that
  actually mattered -- **masking must be checked per-timestep from the data, not from a single "first N
  frames are real" cutoff**, because ~85% of the real recorded clips have at least one frame *in the middle*
  where MediaPipe briefly lost the hand(s) (fast motion, hand leaving frame). Verified end-to-end against the
  real trained model on real held-out clips: TS and Keras agree exactly (max prob diff ~4e-6, 8/8 top-1 match).
- **Honest accuracy (held-out video per sentence, never trained on): top-1 15.2%, top-3 23.2%, top-5 29.3%**
  (chance is ~1% over 101 classes). Train accuracy reaches 97% -- the gap is expected and is why the result UI
  shows several ranked guesses, not just one. This is a genuinely harder problem than the letters/digits (one
  static hand shape vs. a multi-second motion sequence) with far less data (~5 videos/class vs. thousands of
  photos), and is presented on the site as an early prototype, not a finished translator.
- Two real bugs were caught and fixed by parity-testing *before* the multi-hour training run: a broadcasting-axis
  bug in `sort_hands()` (fixed hand-mask reordering too), and a broadcasting-shape bug in `augment()`'s rotation
  step (wrong number of inserted axes for the sequence case).

## Update 10 (2026-09-23): reference clips on the Sentence Builder

Each of the 101 sentences now has a short reference video (from the ISL-CSLTR training set itself) you can
watch before recording -- click a sentence in "Browse the known sentences" on `sentence.html`. Generated by
`model_assets/sentences/make_sample_clips.py` (picks the largest/clearest source file per sentence, re-encodes
with ffmpeg -- via the `imageio-ffmpeg` package, no system ffmpeg install needed -- to small silent 320px-tall
H.264 clips) into `webapp/public/sentence_clips/<label-index>.mp4`. All 101 together are ~5.5 MB.

**Gotcha found while building this:** `.venv_tfjs/Scripts/pip` (the pip.exe binary) still has the OLD drive
path baked into it from before the project folder was moved, so running it directly silently installs into the
*old* C: venv (which may not even exist if that folder gets deleted) while `python.exe` itself correctly uses
the new location. **Always install with `.venv_tfjs/Scripts/python.exe -m pip install ...`, never `.venv_tfjs/Scripts/pip
install ...`.**

## Update 11 (2026-09-24): sentence builder now uses DTW, not the GRU -- real accuracy roughly doubled

The GRU sequence model (Update 9, top-1 15.2%/top-3 23.2%/top-5 29.3%) has been replaced by **DTW k-NN**
(dynamic time warping against a bank of real training clips) after both were measured on the SAME rigorous
leave-one-out test across all 687 training videos (not the old single 99-video split, which was noisy):

| | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| GRU (old, single 99-video split) | 15.2% | 23.2% | 29.3% |
| **DTW k-NN (full 687-video leave-one-out)** | **24.5%** | **35.4%** | **41.3%** |

Chance is ~1% over 101 classes. DTW wins because with only ~5 recordings per sentence, a 250K-parameter neural
net doesn't have enough data to learn a general rule (it hit 97% *training* accuracy -- pure memorisation) --
comparing directly against real examples via a good distance metric doesn't need to generalise past that.
A simpler fixed-length-resample k-NN was tried first and also beat the GRU (~18%/28%/33%), confirming the GRU
itself was the weak link, not the general "compare against real examples" idea; true DTW (frame-by-frame
alignment that tolerates signing at a different speed) beat that too.

- `model_assets/sentences/eval_dtw_full.py` -- the leave-one-out evaluator (also has a mirror-augmented
  reference set; a fixed-length-resample ablation is in `eval_knn.py`/`eval_knn2.py`).
- `model_assets/sentences/export_dtw_bank.py` -- exports every video's real per-frame landmarks (+ a mirrored
  copy of each, 1374 sequences total) to `webapp/public/sentence_model/dtw_bank.bin` (23.3 MB binary) +
  `dtw_meta.json` (offset/length/class index per sequence).
- `webapp/src/sentences/dtw.ts` -- the DTW recursion and k-NN classification, run as plain TS (no ML runtime
  needed since there's no model to run -- just a distance calculation). **Verified against the exact shipped
  binary data**: 24 query/reference distance pairs matched Python's `dtw_dist()` to ~1e-7 (float32 precision),
  and a clean (non-webcam-recompressed) query correctly recovered its true sentence at rank 0 with 100%
  confidence end-to-end through the real bank file.
- Recording semantics changed: only frames where a hand was actually detected are pushed into the query buffer
  (previously the GRU's masking needed all-zero placeholder frames for "no hand" gaps; DTW's reference bank
  never stored those either, so the query must match).
- The GRU code/model files are left in `model_assets/sentences/` and `webapp/src/sentences/model.ts` for
  reference (documents an approach that was tried and measurably lost) but nothing in the app imports them
  anymore.
- **Caveat on the live smoke test**: an automated headless-browser test using a *webcam feed built by
  re-encoding a training clip three times* (mp4 -> ffmpeg frame extraction -> fake camera -> re-detected by
  MediaPipe in-browser) scored badly on that one clip -- traced to the repeated re-encoding degrading landmark
  quality, not a bug (confirmed by feeding the SAME algorithm clean, once-extracted landmark data end-to-end
  through the real shipped bank, which correctly recovered the right sentence at 100% confidence). A real
  webcam recording only goes through one capture step, not three, so this shouldn't recur in practice, but it
  means the 24.5% top-1 number -- measured on the original, non-recompressed training videos -- is the number
  to trust, not a guarantee that every possible video pipeline behaves identically.

## Update 12 (2026-10-01): sentence builder v2 -- body pose + idle trimming (top-1 24.5% -> 31.7%)

The v1 DTW features were hands only, each frame centred on the hands' own centroid, so they threw
away *where* the hands were (on the head for "comb your hair" vs. in front of the chest), and every
clip's idle hands-in-lap start/end had to be aligned too. Measured on the same full 687-video
leave-one-out test (now 13 s instead of 17 min, via the numba harness `model_assets/sentences/dtw_lab.py`):

| | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| v1 DTW, hands only | 24.5% | 35.4% | 41.3% |
| + Sakoe-Chiba band 0.2 | 25.0% | 36.7% | 42.5% |
| + body-relative hand position + pose elbows/wrists | 28.0% | 39.7% | 45.6% |
| **+ trim idle lead-in/lead-out (v2, shipped)** | **31.7%** | **47.9%** | **53.7%** |

- **Pose:** `model_assets/sentences/extract_pose.py` runs MediaPipe `pose_landmarker_lite` on exactly the
  same frames as the hand extraction (~1.5 h, resumable cache in `tmp/isl_sentence_pose/`) and writes
  `sequences_pose.npz` (hands + pose + frame size). The browser runs the same lite pose model
  (`webapp/src/sentences/pose.ts`), only on the 80 ms sample ticks while recording.
- **Features (140/frame):** v1's 128 hand-shape floats + each hand's palm centre relative to the body
  (shoulder-midpoint origin, shoulder-width scale, pixel coords so portrait/landscape framing drops out)
  + pose elbows & wrists (weight 2). A frame is kept if a hand OR a body was seen. Frames before the
  first / after the last "wrist raised above 0.6 shoulder-widths below the shoulders" frame are trimmed.
  Single source of truth: `model_assets/sentences/pose_features.py` <-> `webapp/src/sentences/features.ts`
  (`clipFeatures`); DTW band lives in `dtw_meta.json`.
- **Parity verified:** `check_pose_parity.py` + `node --experimental-strip-types webapp/scripts/check-pose-parity.mts`
  -- features max abs diff 2.4e-6, DTW distances rel. diff 2e-7 (incl. the no-body fallback path). A
  headless-Chrome smoke test with a fake camera (a training clip of "comb your hair", a class v1 got 0/7
  on) returned "comb your hair" as the top match through the real page.
- Tried and rejected (lower on the same test): per-hand shape normalisation, velocity features, z-dropping,
  mean-of-k class aggregation, lower trim thresholds. Weights were picked on this LOO test, so treat the
  numbers as slightly optimistic; the landscape around the chosen point was smooth (31-32% top-1).
- v1 export script + bank are backed up in `tmp/export_dtw_bank_v1_backup.py` and `tmp/sentence_model_v1_backup/`.
- Users must now keep head + shoulders in frame; the page says so and warns after a recording where the
  body wasn't visible most of the time.

## Update 13 (2026-10-01): ASL mode on the Sentence Builder (WLASL word-chaining)

`sentence.html` now has an **ISL · Indian / ASL · American** toggle (remembered in localStorage). ISL is unchanged.
Each language has its own folder: `webapp/public/sentence_model[_asl]/` (bank, meta, labels, glosses) and
`webapp/public/sentence_clips[_asl]/`; `dtw.ts` / `catalog.ts` take the folder as `base`.

- **Why word-chaining:** no public ASL dataset has fixed sentences each recorded several times (How2Sign has
  each sentence ~once; ASL-Homework-RGBD is gated; RWTH-BOSTON is old B/W). So `model_assets/asl_sentences/catalog.json`
  defines 101 everyday sentences (the ISL list's sentences where ASL has the words, plus common ASL phrases) in
  **ASL gloss order**, using only WLASL glosses (140 words). *The glosses were written by us, not by a Deaf
  ASL signer -- have them reviewed.*
- **Data:** WLASL (Kaggle "wlasl-processed", 11,980 videos) extracted to `asl_sentences_data/wlasl/`.
  License **C-UDA: academic/computational use only, no commercial use**; the clips come from third-party sites
  (watermarks visible), so check before publicly hosting `sentence_clips_asl/`.
- **Pipeline:** `extract_words.py` (hands + pose for the 1,050 clips of the 140 words, ~2 h, cache
  `tmp/asl_word_lm/`) -> `build_bank.py eval|export` (per-word v2 features incl. idle trim, concatenated in gloss
  order; 7 random chains per sentence + mirrors = 1,414 refs, 31 MB) -> `make_sample_clips.py` (reference clips,
  3.1 MB). Same 140-float features / DTW as ISL (`pose_features.py`), so no new TS matcher code.
- **Accuracy (all proxies, no real ASL sentence recordings exist to test on):**
  - signer-held-out (30% of signers never in refs), chained queries: top-1 76.8%, top-3 90.5%, top-5 94.0%
    (`build_bank.py eval`, 3 seeds); with word-edge cuts, 0.7-1.4x speed and jitter: 77.3 / 90.3 / 93.7% (`eval_hard.py`)
  - browser-style whole-clip queries (detection over the joined reference clip, rest pose between words):
    69.3 / 87.1 / 91.1% (`eval_clips.py`; its words are in the bank, so optimistic there)
  - Real fluent signing blends words; expect lower live. Needs a real-user test.
- Headless-Chrome smoke test: toggle -> ASL bank + 101-sentence catalog load, record, classify all work
  (fake-camera content loops, so its recognised sentence isn't meaningful).

## Update 14 (2026-10-01): ISL sentences -- tuned for NEW signers + "teach it your signing"

User report: ISL sentence results were wrong live. Root cause: every number so far was leave-one-out, where
the same signer's *other takes* of a sentence are in the bank. A realistic test (`eval_cross.py`: queries
from the 240x320 phone recordings, references only from the 1920x1080 camera recordings and vice versa --
different people) showed v2 at only **~11% top-1**. Worse, the two phone signers (different people, same room,
same phone) match each other's takes only ~7% of the time: people sign these sentences very differently.

1. **v3 features for ISL, tuned on the cross-setup test** (`pose_features.py`; experiments `dtw_exp_x*.py`):
   compact per-hand shape (10 joints rel. wrist / palm length) instead of 63 raw coords, much heavier arm
   weight (8), location 0.5, trim at 0.4. Cross-setup **10.8/18.2/24.2% -> 17.2/31.3/36.9%** (top-1/3/5);
   LOO drops 31.7 -> 27.7% top-1 (expected: less signer-specific detail). 54 floats/frame, ISL bank 22.5 -> 7.9 MB.
   Tried and rejected: hand-gap interpolation, L1/Euclid metrics, per-channel late fusion, per-clip centring.
2. **ASL keeps v2** (`pose_features_v2.py`): v3 dropped ASL from 77.9% to <=59% (ASL words differ by handshape
   + the two hands' relative position). Each `dtw_meta.json` now has `"features": {"version": 2|3, ...params}`
   and `features.ts clipFeatures(frames, w, h, spec)` dispatches on it. Parity re-verified for both banks
   (`check_pose_parity.py` + `webapp/scripts/check-pose-parity.mts`: max feature diff 4e-6, DTW rel. 2e-7).
3. **Teach it your signing** (`webapp/src/sentences/personal.ts`): after a result, "Correct -- remember it" or
   pick the right sentence + "Save as this"; "Practise this sentence" pre-selects that sentence. The recording's
   feature sequence is stored in localStorage (per language + feature recipe, newest 60) and matched as an
   extra reference (`classifySentence(frames, personal)`). Verified with `webapp/scripts/check-personal.mts`
   (a taught clip wins 7/7) and in headless Chrome (save, count, persists across reload). Its real-world gain
   can't be measured on this dataset (no signer IDs), but a same-person take is the closest reference there is.
4. The Sentence Builder's hand detector now uses min detection confidence 0.3 (as the banks were extracted);
   other pages keep 0.5.

## Update 15 (2026-10-01): sign-in + history

- **Sign-in** (`login.html`, `src/login.ts`, `src/auth.ts`): demo account **demo / demo1234** (username is
  case-insensitive). "Keep me signed in" -> localStorage, otherwise sessionStorage (ends with the browser).
  Header shows "Sign in" or the user's menu (Your history / Sign out); mobile menu too.
  The app pages -- translate, sentence, math, history -- are protected by an inline `<head>` script injected
  by `vite.config.ts` (`PROTECTED`), which redirects to `/login.html?next=...` before the page renders or asks
  for the camera; the page scripts also refuse to start without a session. `next` only accepts same-site paths.
  **Not real security:** there's no backend, so the check runs in the browser and the account list (salted
  SHA-256 hash, not the plain password) ships in the JS bundle. Fine for a demo / college project; a public
  deployment with real users needs a server-side login. To add a user: append to `ACCOUNTS` in `src/auth.ts`
  with `sha256("hastavani:<lowercase username>:<password>")`.
- **History** (`history.html`, `src/historyPage.ts`, `src/history.ts`): per-user list in localStorage (newest 500).
  Logged: translator text (when cleared, spoken, or the page is left; never the same text twice in a row),
  every sentence result (updated to "confirmed/corrected by you" when taught), calculator results and practice
  rounds. Page: grouped by day, filter (All/Translator/Sentences/Math), search, delete one, Clear history.
- Verified in headless Chrome: signed-out redirect with `next`, wrong password error, login -> back to the page,
  sentence result logged + corrected, delete, clear, sign-out, protected page redirect after sign-out;
  screenshots checked in light/dark and on mobile.
- **Follow-up (same day):** the whole site now requires sign-in -- every page except `/login.html` has the
  `<head>` guard (vite.config.ts `LOGIN_PAGE`), and the sign-in page gets a bare header/footer (logo + theme
  toggle only): the `<!--APP-->...<!--/APP-->` blocks in `partials/` (nav, menus, "Start translating",
  footer link columns) are stripped there. After signing in you land on `next` or the home page; signing
  out goes straight to the sign-in page.

## Update 16 (2026-10-01): content refresh (Home, About, How it works, Learn)
Home now presents all four tools (translator, sentences, math, learn), a sentence-builder spotlight with the
"teach it your signing" story, a "why" grid (incl. history + learns-your-signing), updated language cards and a
personal "Welcome back, <name>" badge (src/site.ts). About: name meaning (hasta + vāṇī), what it does, who it's
for, privacy, honest limits, open-data credits, expanded FAQ. How it works: letters / numbers / sentences with
jump links; new Numbers section (rule-based digits, 97% on 872 photos, 7/8 unvalidated); sentences section
restructured into 3 steps + stats + honest note. All numbers on the pages are the measured ones from Updates 2-14.

## Update 17 (2026-10-09): ISL sentences v3.1 -- smoothing + arm velocity (cross-setup top-1 17 -> 20%)
User asked for 90% sentence accuracy. Measured honestly that this is not reachable with ISL-CSLTR (687 clips,
~5-7 per sentence, people sign very differently): even with only 5 candidate sentences top-1 is ~60%
(`dtw_shortlist.py`). What was shipped instead, tuned on the cross-setup test (`eval_cross.py`):

| ISL, cross-setup | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| v3 | 16.9% | 30.9% | 36.8% |
| **v3.1: 5-frame moving average + arm velocity x5 (shipped)** | **20.0%** | **32.3%** | **41.3%** |

- `pose_features.add_motion()` <-> `features.ts addMotion()`; params `smooth`/`wVel` in `dtw_meta.json`
  (62 floats/frame, bank 9.1 MB). Parity re-verified (features 4.6e-6, DTW rel. 1.8e-7, incl. no-body path);
  `check-personal.mts` OK. v3 bank backed up in `tmp/sentence_model_v3_backup/`.
- Personal ("teach it") samples are keyed by feature recipe, so samples saved with v3 are no longer loaded --
  re-teach after this update.
- ASL unchanged: same idea gave no gain there (77.9 -> 78.2%, noise; `asl_sentences/eval_motion.py`).
- Experiments: `dtw_exp_y.py` (smoothing / velocity / v2+v3 fusion). Settings picked on this same test, so
  treat the gain as slightly optimistic.
- Per-sentence accuracy for both languages: `per_sentence_accuracy.py` -> `tmp/per_sentence_accuracy.json`
  (ASL: 46 sentences >= 90% top-1 on its proxy test; ISL: none >= 90%, 19 >= 50%).
