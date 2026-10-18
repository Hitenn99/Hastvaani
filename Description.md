# Hastavāṇī — Project Description

**A browser-based system for recognising American Sign Language (ASL) and Indian Sign Language (ISL) fingerspelling, digits and everyday sentences from a webcam.**

This is the complete technical record of the project, written as source material for a research paper. Every number below was measured in this project and comes from its scripts and logs; each result names the test it came from. Negative results (approaches that were tried and dropped) are included on purpose, because several of the design decisions only make sense in light of them.

---

## Contents

1. [Overview](#1-overview)
2. [Problem statement and objectives](#2-problem-statement-and-objectives)
3. [System architecture](#3-system-architecture)
4. [Technology stack and environment](#4-technology-stack-and-environment)
5. [Datasets](#5-datasets)
6. [Module A — ASL alphabet (fingerspelling)](#6-module-a--asl-alphabet-fingerspelling)
7. [Module B — ISL alphabet (fingerspelling)](#7-module-b--isl-alphabet-fingerspelling)
8. [Real-time letter pipeline (smoothing, speech)](#8-real-time-letter-pipeline-smoothing-speech)
9. [Module C — Digits 0–9 (Math with signs)](#9-module-c--digits-09-math-with-signs)
10. [Module D — ISL sentence recognition](#10-module-d--isl-sentence-recognition)
11. [Module E — ASL sentence recognition](#11-module-e--asl-sentence-recognition)
12. [Personalisation ("Teach it your signing")](#12-personalisation-teach-it-your-signing)
13. [Web application](#13-web-application)
14. [Verification and parity testing](#14-verification-and-parity-testing)
15. [Consolidated results](#15-consolidated-results)
16. [Key findings and lessons](#16-key-findings-and-lessons)
17. [Limitations](#17-limitations)
18. [Future work](#18-future-work)
19. [Reproducibility (files and commands)](#19-reproducibility-files-and-commands)
20. [Suggested paper structure](#20-suggested-paper-structure)
21. [References](#21-references)

---

## 1. Overview

**Name.** *Hastavāṇī* (हस्तवाणी) combines the Sanskrit words *hasta* (hand) and *vāṇī* (voice, speech): "the voice of the hand".

**What it does.** A user opens a website, allows camera access, and signs. The system recognises:

| Capability | Languages | Vocabulary | Method (final) |
|---|---|---|---|
| Fingerspelling (letters) | ASL, ISL | 26 letters each (A–Z) | Hand-landmark MLP classifier |
| Digits | ASL, ISL (one shared reader) | 0–9 | Rule-based geometry on hand landmarks (+ optional personal kNN) |
| Sentences | ISL | 101 everyday sentences | DTW nearest-neighbour matching on hand + body-pose features |
| Sentences | ASL | 101 everyday sentences | Same DTW matcher, references built by chaining WLASL word clips |

Recognised output is shown on screen, spoken aloud (Web Speech API) and saved to a per-user history.

**Main design properties.**
- **Runs entirely in the browser.** There is no backend server and no video upload; all recognition runs client-side in TypeScript.
- **Landmark-based, not pixel-based.** Every module works on MediaPipe hand (21 points per hand) and body-pose (33 points) landmarks, which is what made the system robust to background, lighting and skin tone (Section 6.4).
- **Honest evaluation.** For every module the project measured generalisation with a test that the model could not "cheat" on (cross-source photos, temporal splits, held-out signers, cross-setup tests), and published the lower, realistic number.

**Final headline numbers** (full detail in Section 15):

| Module | Realistic test | Top-1 | Top-3 |
|---|---|---|---|
| ASL letters (landmark MLP) | 672 photos from a different source, never trained on | **82.3%** | **93.0%** |
| ISL letters (landmark MLP) | temporal split, same signer | 100% *(not evidence of real-world accuracy)* | — |
| Digits (rules) | 872 real photos | **97%** overall | — |
| ISL sentences (DTW v3) | cross-setup (new signers), 101 classes | **17.2%** | **31.3%** (top-5 36.9%) |
| ASL sentences (DTW v2, word-chained) | held-out signers, chained queries | **76.8%** | **90.5%** (top-5 94.0%) |

Chance level for 101 sentences is ≈1%.

---

## 2. Problem statement and objectives

**Problem.** Sign languages are the primary language of millions of Deaf and hard-of-hearing people, yet few hearing people can read them. Most existing recognisers are research prototypes that need a GPU, a server, a special camera, or a controlled studio setting, and they often report accuracy on test sets that share signers or recording sessions with the training data. Those numbers do not tell you how the system will do for a new person on an ordinary webcam.

**Objectives.**
1. Recognise ASL and ISL fingerspelling (A–Z) live from an ordinary webcam, on a CPU-only laptop, inside a web browser.
2. Extend recognition beyond single letters: digits (for a signing calculator and practice game) and whole everyday sentences.
3. Keep the system private (no uploads) and deployable as a static website.
4. Measure accuracy honestly, with tests that reflect real use (new camera, new signer), and report failures as well as successes.
5. Wrap it in a complete, usable web application: learning resources, speech output, user sign-in and activity history.

---

## 3. System architecture

```
                  ┌──────────────────────────── Browser (client only) ───────────────────────────┐
 Webcam ──frames──►  MediaPipe HandLandmarker (≤2 hands × 21 pts)   MediaPipe PoseLandmarker (33 pts)
                  │            │                                              │ (sentences only)
                  │            ▼                                              ▼
                  │  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────────────────┐
                  │  │ Letter MLP    │  │ Digit rules      │  │ Clip features (hands + body)     │
                  │  │ ASL: 63→26    │  │ finger extension │  │ idle-trim → DTW vs reference bank│
                  │  │ ISL: 128→26   │  │ thumb, touches   │  │ (+ user's own recordings)        │
                  │  └──────┬────────┘  └────────┬─────────┘  └───────────────┬──────────────────┘
                  │         ▼                    ▼                            ▼
                  │  confidence gate + majority-vote smoothing       ranked sentence guesses
                  │         └──────────────► text on screen ◄───────────────────┘
                  │                          speech (Web Speech API)
                  │                          history (localStorage, per user)
                  └───────────────────────────────────────────────────────────────────────────────┘
 Offline (Python): dataset extraction → landmark extraction → training / tuning → export to JSON/binary
```

**Offline side (Python).** Extracts landmarks from the datasets, trains the small classifiers, tunes the sentence features, and exports everything to static files: JSON weights for the MLPs, and a binary reference bank plus JSON index for the sentence matcher.

**Online side (TypeScript, browser).** Runs MediaPipe on webcam frames, re-implements the feature code exactly (checked numerically against Python, Section 14), and runs the MLP forward pass and the DTW matcher in plain TypeScript. No ML runtime such as TensorFlow.js is needed in the final app.

---

## 4. Technology stack and environment

| Layer | Technology |
|---|---|
| Hand / body tracking | Google MediaPipe Tasks: HandLandmarker (float16 model), PoseLandmarker *lite*; `@mediapipe/tasks-vision` in the browser, `mediapipe` Python package offline |
| Training | Python 3.11, TensorFlow/Keras 2.x, NumPy, SciPy, OpenCV, Pillow, pyarrow |
| Fast experimentation | Numba (parallel JIT DTW; a full 687-video leave-one-out run takes ~13 s instead of 17.7 min in pure Python) |
| Model conversion (earlier image CNNs) | TensorFlow.js converter 4.22 |
| Front end | Vite 8 (multi-page), TypeScript 6, Tailwind CSS v4; plain TS forward passes (no TF.js in the final bundle) |
| Speech | Browser Web Speech API (`speechSynthesis`) |
| Storage | Browser `localStorage` / `sessionStorage` (session, history, personal samples, calibration) |
| Testing | Headless Chrome via `playwright-core` (end-to-end flows, fake-camera smoke tests); Node 24 `--experimental-strip-types` for TS↔Python parity scripts |
| Deployment | Static site (`npm run build` → `webapp/dist`); config files for Vercel and Netlify included |

**Hardware.** All training and evaluation ran on a laptop CPU: 12th-gen Intel Core i7-1255U, 10 cores, **no GPU**, Windows 11. The project was moved from the C: drive to E: midway because C: ran out of space; temporary files are redirected to `tmp/`.

---

## 5. Datasets

| # | Dataset | Used for | Size used | Notes |
|---|---|---|---|---|
| 1 | Kaggle **Sign Language MNIST** | ASL letters v1 (dropped) | 24 classes (A–Y without J, Z), 28×28 grayscale | Originally supplied pre-trained model; 99.80% on its own test set, unusable on webcam |
| 2 | **Marxulia/asl_sign_languages_alphabets_v03** (Hugging Face) | ASL letters training | 10,873 photos, 26 classes; 8,514 with a detected hand | Real, varied photos; inconsistent hand scale |
| 3 | **EitanG98/asl_letters** (Hugging Face) | ASL letters: *train* split for training; *validation* split = **held-out cross-source test** | validation: 720 images → 672 with a detected hand | Different photographer, camera and background; never trained on (validation split) |
| 4 | **Hemg/Indian_sign_language_dataset** (Hugging Face mirror of Kaggle "indian-sign-language-dataset") | ISL letters; part of the digit validation set | 42,745 photos, 35 classes → 31,945 letter images (A–Z) | Frames of one continuous recording per letter, **single signer**, dark background, mostly two-handed |
| 5 | **ISL-CSLTR** (Mendeley Data, kcmpdxky7p) | ISL sentences | 687 videos, 101 sentences (~5–7 per sentence) | Two native signers + volunteers; two recording setups: 492 videos at 1920×1080 (camera), 195 at 240×320 (phone, portrait) |
| 6 | **WLASL** (Kaggle "wlasl-processed") | ASL sentence references (word clips) | 11,980 videos, 2,000 glosses (~6 per gloss); 140 glosses / 1,050 clips used | Licence: C-UDA, academic/non-commercial use only; clips come from third-party sign websites |
| 7 | Digit charts supplied by the project owner + *Ekbote & Joshi (2017)* ISL numeral figure | Digit handshape definitions, Learn-page charts | 0–9 per language | Used to define the rules, not for training |

Datasets considered and rejected for ASL sentences: How2Sign (each sentence recorded roughly once, 80 h), ASL-Homework-RGBD (gated access, free-form content), ASLLRP (mostly unique utterances), RWTH-BOSTON-104 (old black-and-white cameras, unsuitable for MediaPipe), FLEURS-ASL (translation benchmark). No public ASL dataset has fixed sentences each recorded several times, which is why ASL sentences are built from word clips (Section 11).

---

## 6. Module A — ASL alphabet (fingerspelling)

The ASL letter model went through four versions. The progression is one of the project's main findings: **models that scored high on their own test sets failed on a new camera, and the fix was to change the input representation, not to add model capacity.**

### 6.1 v1 — Sign Language MNIST CNN (provided, then dropped)
- Pre-trained CNN on Sign Language MNIST: 24 classes, 28×28 grayscale.
- **99.80%** on the dataset's own test set. Conversion to TF.js verified (max abs diff 1.7e-5).
- **Live webcam: unusable.** Signs were recognised as the wrong letter almost every time.
- Two real bugs were found and fixed first:
  - The live hand crop used 35% padding, far looser than the dataset's edge-to-edge crops.
  - The letter-lock logic dropped its lock on a single noisy frame, so a held sign kept being re-appended ("HHHHHH").
- Accuracy stayed poor after the fixes. Root cause: the dataset is a controlled-lab distribution (grayscale, tightly cropped, almost no lighting or background variety) that a webcam never resembles.

### 6.2 v2 — MobileNetV2 transfer learning on real photos
- Dataset: Marxulia (26 classes, 10,873 photos).
- **From-scratch CNNs collapsed.** Loss stayed at ln(26) = 3.258 (chance level) through 6+ ablations: different learning rates, with and without spatial attention, GroupNorm or dropout, and even a raw NumPy/Keras loop.
- **Fix 1, consistent framing.** Every training photo was cropped to the MediaPipe hand box (the same box logic as the live app). 8,514 of 10,873 photos had a detectable hand.
- **Fix 2, transfer learning.** ImageNet-pretrained MobileNetV2 backbone, first frozen and then with its top ~30 layers fine-tuned, plus a spatial-attention head. Input 96×96 RGB.
- Results:
  - **93.5%** validation accuracy after the frozen phase.
  - **88.9%** held-out test accuracy (same source).
  - **Cross-source test (EitanG98 validation): only 27.2%.** The model had memorised Marxulia's camera and lighting signature; its confusion matrix was near-random.

### 6.3 v3 — Two sources + heavier augmentation
- Training set: Marxulia + EitanG98 *train* split combined, with per-batch colour and geometric augmentation and a gentler fine-tune (last 10 layers). Checkpointing was added after a crash.
- **Same-source test 73.0%; cross-source 53.2%** (125/235 images, 25 with no hand).
- Better than v2 on a new camera, but still weak and slow on the CPU backend.

### 6.4 v4 (final) — Hand-landmark MLP
- **Idea.** Classify the 21 MediaPipe hand landmarks the app already computes, instead of pixels. Landmarks carry no background, lighting or skin-tone information.
- **Features (63).** Landmarks (21×3) minus the wrist point, divided by the largest xy distance from the wrist (hand size), then flattened.
- **Training data.** 6,557 hand-cropped training photos with a detected hand (Marxulia + EitanG98 train), minus a 10% same-source hold-out.
- **Augmentation.** Each training sample gets 7 augmented copies: random mirror (50%), in-plane rotation (σ = 0.15 rad), anisotropic scale (σ = 0.08) and jitter (σ = 0.004).
- **Architecture.** Dense 256 → BatchNorm → ReLU → Dropout 0.3 → Dense 128 → BatchNorm → ReLU → Dropout 0.3 → Dense 26 softmax.
- **Training.** Adam (lr 1e-3), batch 128, up to 60 epochs, early stopping on validation accuracy (patience 8), ReduceLROnPlateau (×0.5, patience 3).
- **Results on 672 EitanG98 validation photos** (different source, never trained on, hand detected):

| Model | Top-1 | Top-3 |
|---|---|---|
| v3 image CNN | 53% | — |
| **v4 landmark MLP** | **82.3%** | **93.0%** |

  - Same-source hold-out: 91.8%.
  - Counting the 48 photos where no hand was detected as wrong: 76.8% top-1, 86.8% top-3.
- **Deployment.** Weights exported to JSON (`webapp/public/asl_landmarks/model.json`). The forward pass is plain TS matrix multiplication, so no TF.js is needed, and inference is instant on CPU.
- **Known limitation.** ASL J and Z involve motion. The model sees one frame at a time, so those two letters are static approximations.

---

## 7. Module B — ISL alphabet (fingerspelling)

### 7.1 Image CNN (dropped)
- Trained from scratch on the Hemg ISL photos: 31,945 images, 26 letters, 64×64 RGB. Conv blocks, GroupNorm, spatial attention, softmax; 80/10/10 stratified split.
- **Test accuracy 100.0%**. This was investigated rather than reported, because 100% on 26 classes is a red flag.
- **Leak found.** Within a class, the mean absolute pixel difference between images is only ~2–15 out of 255, even between frames 400+ apart. Each letter is one continuous recording of one person, so a random split puts near-duplicate frames into train and test alike.

### 7.2 Two-hand landmark MLP (final)
- **Landmarks.** Every 2nd frame, up to 2 hands, MediaPipe min detection confidence 0.3 → **15,733 samples**; 59.7% of them have both hands detected.
- **Features (128).** Hands sorted by wrist x. All present points are centred on their joint centroid and divided by the maximum xy distance from it. Two presence flags are appended (126 coordinates + 2 flags).
- **Honest(er) split.** Temporal, per class: train = first 70% of frames, validation = 72–80%, test = last 18%. The gaps prevent adjacent near-duplicate frames from straddling the split.
- **Architecture.** Dense 384 → BN → ReLU → Dropout 0.3 → Dense 192 → BN → ReLU → Dropout 0.3 → Dense 26. Six augmented copies of each sample; same optimiser and callbacks as ASL. The shipped model is refit on all frames.
- **Result.** Temporal-split test accuracy 100%. This is still one signer in one session, so it is **not** evidence of real-world accuracy; accuracy on other signers is unmeasured.
- **Deployment.** BatchNorm is folded into the Dense layers and the result exported to JSON. The TS forward pass matched Keras to a max probability difference of 6e-8.
- **Note.** The dataset's alphabet differs from the official ISLRTC chart for several letters; the app follows the dataset's chart.

---

## 8. Real-time letter pipeline (smoothing, speech)

Per video frame (inside `requestAnimationFrame`, so the UI never blocks):
1. MediaPipe HandLandmarker in VIDEO mode (1 hand for ASL, 2 for ISL).
2. Normalise the landmarks, run the MLP, take the arg-max.
3. **Confidence gate.** Only predictions with confidence ≥ 0.60 are accepted.
4. **Majority vote.** A letter is locked in when it wins at least 5 of the last 8 accepted predictions.
5. **Lock hysteresis.** The lock survives up to 10 consecutive missed or low-confidence frames. This fixes the "HHHHHH" repeat bug of v1.
6. Newly locked letters are appended to a word buffer (with Space, Backspace, Clear and Speak-word controls). Each letter is spoken at most once every 2 s.

---

## 9. Module C — Digits 0–9 (Math with signs)

**Why no neural network.** The digit handshapes on the charts the site teaches did not match the public digit datasets. Vedant3907 "Numbers-Sign-Language" is not real ASL numerals, and Hemg's ISL digits 6–9 differ from the chart. Training on them would have taught the wrong signs.

**Rule-based classifier** (`webapp/src/numbers/digits.ts`). The features are rotation- and scale-invariant (ratios and angles):
- **Finger extension** for index, middle, ring and little: joint angle at the PIP and a tip/base distance ratio (sigmoid-soft thresholds, e.g. angle midpoint 135°, ratio midpoint 1.6).
- **Thumb out vs. tucked** (midpoint 0.68).
- **Thumb touching fingertip** for index, middle, ring and little (midpoint 0.42). This is what separates 6, 7, 8 and 9.

A soft score per digit gives probabilities. ASL and ISL numerals coincide for most digits, so one reader serves both.

**Validation set.** 872 real photos whose signs match the charts (`model_assets/numbers/extract_digit_validation.py`). They come from verified-matching Hemg ISL classes and from ASL letter photos that share a digit handshape: O = 0, V/U = 2, W = 6, F = 9, B = 4.

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 83% (n=23) | 100% | 95% | 100% | 97% | 100% | 97% | n/a | n/a | 90% (n=29) | **97%** |

- 7 and 8 could not be validated because no matching photos were available; they use the same thumb-touch mechanism as 6 and 9.
- The thresholds were tuned on this same set, so real-world accuracy may be slightly lower.

**Personal calibration** (optional). The user records examples of each digit; only landmark vectors are stored, in localStorage, up to 80 per digit. Once ≥30 samples exist, a k-nearest-neighbour vote (k = 7) is blended with the rules.

**Page features.** A signed-digit calculator (+ − × ÷ and % as remainder), a 10-question practice game (easy/medium/hard, score, streaks, hints showing the needed sign, confetti), and digit charts for each language.

---

## 10. Module D — ISL sentence recognition

**Task.** Isolated sentence recognition over a **fixed catalog of 101 ISL sentences** (not open-vocabulary translation). The user records a clip (up to ~7.5 s); the system returns ranked guesses.

### 10.1 Data preparation
- ISL-CSLTR: 687 videos, 101 sentences.
- **Sampling.** Every 2nd frame of the 25 fps source, i.e. ~12.5 fps (one sample per 80 ms), capped at 90 frames.
- **Hands.** MediaPipe HandLandmarker, up to 2 hands, confidence 0.3; sorted by wrist x.
- **Body.** MediaPipe PoseLandmarker *lite* on exactly the same frames (added in v2); a body was found in 100% of frames.
- **Statistics.** Mean 46.8 sampled frames per clip, of which 34.8 have a hand. Only **31%** of frames have both hands detected. ~85% of clips have at least one mid-clip frame where the hand is lost (fast motion).

### 10.2 Approach history

| Version | Method | Test protocol | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| GRU | 2 stacked Bidirectional GRU + Dense, Keras Masking, heavy augmentation (14 copies; mirror/rotation/scale/jitter/time-resample) | one held-out video per sentence (99 videos) | 15.2% | 23.2% | 29.3% |
| kNN (fixed-length resample) | resample to fixed length, Euclidean kNN | full leave-one-out (LOO) | ~18% | ~28% | ~33% |
| **DTW v1** | DTW k-NN on 128 hand features (v1 letter features per frame), mirrored copies in bank | full LOO, 687 videos | 24.5% | 35.4% | 41.3% |
| DTW v2 | + Sakoe-Chiba band 0.2 | full LOO | 25.0% | 36.7% | 42.5% |
| DTW v2 | + body-relative hand location + pose elbows/wrists | full LOO | 28.0% | 39.7% | 45.6% |
| DTW v2 | + idle lead-in/lead-out trimming | full LOO | **31.7%** | **47.9%** | **53.7%** |
| **DTW v3 (final)** | compact hand shape, arms weighted ×8, trim 0.4 | **cross-setup (new signers)** | **17.2%** | **31.3%** | **36.9%** |

- The GRU reached **97% training accuracy** (memorisation): about 5 examples per class are too few for a 250K-parameter network.
- DTW needs no generalisation beyond "distance to the nearest real example".

### 10.3 The evaluation discovery: leave-one-out is optimistic
- In LOO, the same signer's *other takes* of a sentence remain in the reference bank. After the user reported poor live accuracy, a **cross-setup test** was built (`eval_cross.py`):
  - queries from the 240×320 phone recordings, references only from the 1920×1080 camera recordings, and vice versa;
  - the two setups contain different people, so this is the closest available proxy for "a new person on a new camera".
- **v2 dropped to 10.8% top-1 / 18.2% top-3 / 24.2% top-5** on this test.
- **Inter-signer variation is the dominant difficulty.** The two phone takes of each sentence are by two *different people*, filmed in the same room on the same phone. Matching one against the other gave only **6.8% top-1**.
- Within the camera group (where some signers repeat sentences), LOO gives 35.2% top-1 / 55.6% top-3 / 64.6% top-5.

### 10.4 Final feature set (v3, 54 floats per frame)
Computed per clip by `pose_features.clip_features` (Python), mirrored exactly in `features.ts clipFeaturesV3`:
1. **Body reference frame.**
   - Origin: median over the clip of the shoulder midpoint.
   - Scale: median shoulder width.
   - Computed in pixel coordinates, so portrait/landscape framing and camera distance cancel out.
2. **Hand shape (2 × 20).** For each hand, 5 fingertips + 5 middle joints relative to the wrist, divided by the palm length (wrist → middle-finger knuckle), × (w_shape / 3). This is coarser than v1's 63 raw coordinates per hand, which over-fitted each signer's exact handshape.
3. **Presence flags (2).**
4. **Hand location (2 × 2).** Palm centre relative to the body frame, × w_loc.
5. **Arms (2 × 2 × 2).** Pose elbows and wrists relative to the body frame, × w_arm. These are tracked even when the hand model loses the hand.
6. **Ordering.** Hands and arms are ordered by wrist x. The bank also stores an x-mirrored copy of every clip, to handle left-handed signers.
7. **Idle trimming.** Keep only frames from the first to the last frame in which a wrist is raised above `trimThr` shoulder-widths below the shoulder line. This drops the hands-in-lap lead-in and lead-out, whose length varies between clips.
8. **Frame filter.** A frame is kept if a hand or a body is detected.

| Parameter | v2 (ASL bank) | v3 (ISL bank) |
|---|---|---|
| Hand shape | v1 joint-normalised 126 coords + 2 flags | compact 40 + 2 flags |
| w_loc / w_arm | 1.0 / 2.0 | 0.5 / 8.0 (w_shape 2.0) |
| Trim threshold | 0.6 | 0.4 |
| Floats per frame | 140 | 54 |

**Tuning evidence (cross-setup, mean of both directions).**
- Removing trimming halves accuracy (10.8% → 5.7% top-1).
- Arm weight rising from 2 → 8 steadily helped.
- Compact shape beat raw coordinates.
- Rejected, with no gain on this test: hand-gap interpolation, velocity features, dropping z, L1 or per-frame-Euclidean costs, per-channel late fusion, per-clip centring, temporal smoothing, mean-of-k class aggregation.
- Per direction: phone→camera 17.9 / 32.3 / 37.4%; camera→phone 16.5 / 30.3 / 36.4%.
- v3 gives up some same-signer accuracy (LOO 31.7% → 27.7% top-1) in exchange for +60% relative on new signers.
- Smaller features also shrank the ISL reference bank from 22.5 MB to **7.9 MB**.

### 10.5 Matcher
- **DTW.**
  - Frame cost: squared Euclidean distance.
  - Recursion: D[i,j] = cost(i,j) + min(D[i−1,j−1], D[i−1,j], D[i,j−1]).
  - Sakoe-Chiba band of 0.2 × reference length around the relative diagonal.
  - Distance normalised by (T₁ + T₂).
- **Classification.** Class score = minimum distance over that class's references (1-NN per class). Returns the ranked list.
- **Display confidence.** Softmax over −3·(d − d_min)/d_min.
- **Bank.** 1,374 reference sequences (687 clips + mirrors). Binary float32 file with a JSON index; the index also stores the band and the feature recipe.

### 10.6 Live recording (browser)
- Hands are tracked every frame. The pose model runs only on 80 ms sample ticks, to save CPU.
- Hand detection confidence is 0.3, matching the extraction (other pages use 0.5).
- Features are computed on Stop, because the body frame and the idle trim are whole-clip statistics.
- The user is warned if their shoulders were not visible for most of the clip.

---

## 11. Module E — ASL sentence recognition

**Problem.** No public dataset records fixed ASL sentences several times over (Section 5).

**Solution: word-chained references.**
1. **Catalog.** `model_assets/asl_sentences/catalog.json` lists 101 everyday sentences: the ISL catalog's sentences wherever ASL has the words, plus common ASL phrases. Each is written in **ASL gloss order**, e.g. "What is your phone number?" → `YOUR PHONE NUMBER WHAT`. Only WLASL glosses are used (140 distinct words). *The glosses were authored by the project, not a Deaf ASL signer, and should be reviewed.*
2. **Landmarks.** Hands + pose for all 1,050 WLASL clips of those 140 words, sampled by time (80 ms), with the same MediaPipe settings as ISL.
3. **References.** Each word clip is featurised *on its own*: its own body frame, and its own idle trim, which cuts the rest pose before and after the word. The words are then concatenated in gloss order. Each sentence gets 7 random chains plus mirrored copies, for **1,414 references (31 MB)**.
4. **Features.** v2 (140 floats). Tested with the v3 structure, ASL fell to ≤59% top-1: ASL words are distinguished mainly by handshape and the two hands' relative position, which v2's full hand block keeps.

**Evaluation** (all proxies; no real recordings of whole ASL sentences exist to test on):

| Test | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Held-out signers (30% of signers never in references; 3 seeds), chained queries, 10 refs/sentence | 76.8% | 90.5% | 94.0% |
| Same, 7 refs/sentence (shipped) | 77.9% | 87.3% | 90.4% |
| Same, 5 / 3 refs/sentence | 73.1% / 65.8% | 86.4% / 80.2% | 90.3% / 86.1% |
| "Hard" queries: word edges cut 10–25%, speed 0.7–1.4×, jitter | 77.3% | 90.3% | 93.7% |
| Browser-style whole-clip queries (detection over the joined reference video, rest pose between words) | 69.3% | 87.1% | 91.1% |

Caveats:
- The queries are also chained word clips. Fluent signing blends words (co-articulation), so live accuracy will be lower.
- In the whole-clip test, the words used were also in the bank.

**Reference clips.** For each sentence, one WLASL clip per word (the largest file) was joined with ffmpeg: 426×320, 25 fps, H.264, 3.1 MB total.

**Licence.** WLASL is C-UDA (academic/computational use only, no commercial use). The clips carry third-party watermarks.

---

## 12. Personalisation ("Teach it your signing")

**Motivation.** Section 10.3 showed that different people sign the same sentence very differently, while a recording by the same person is the closest possible reference.

**Design** (`webapp/src/sentences/personal.ts`).
- After each result the user can press **"Correct — remember it"**, or choose the right sentence and **"Save as this"**. "Practise this sentence" from a reference clip pre-selects that sentence.
- The recording's exact feature sequence is stored in localStorage (base64 float32), keyed by language and feature recipe, keeping the newest 60.
- It is matched as an extra reference: `classifySentence(frames, personal)`.
- If the bank's feature recipe changes, old samples are ignored automatically.

**Verification.**
- Deterministic test: a clip not in the bank was taught under a deliberately wrong label (as a time-stretched, jittered second take); re-querying returned the taught label **7/7** times.
- In headless Chrome, saved samples persisted across a page reload.
- Its real-world gain could not be measured on this dataset, because the dataset has no signer IDs to build a same-person test.

---

## 13. Web application

### 13.1 Pages
| Page | Purpose |
|---|---|
| `login.html` | Sign-in (the only page reachable when signed out; bare header with no navigation) |
| `index.html` | Home: overview of the four tools, sentence spotlight, features, personalised greeting |
| `translate.html` | Live letter translator (ASL/ISL toggle), confidence meter, word builder, speech |
| `sentence.html` | Sentence builder (ISL/ASL toggle), recording, ranked results, reference clips, catalog search, "Teach it your signing" |
| `math.html` | Digit calculator, practice game, digit charts, personal calibration |
| `learn.html` | Photo guide to both alphabets with tips |
| `history.html` | Per-user activity history: grouped by day, filter (Translator/Sentences/Math), search, delete one, clear all |
| `how-it-works.html` | Technical explanation of letters, numbers and sentences, with measured accuracy |
| `about.html` | Mission, name meaning, audience, privacy, honest limits, data credits, FAQ |

### 13.2 Features
- **Sign-in.**
  - Demo account "demo / demo1234". Only a salted SHA-256 hash is stored in the code.
  - "Keep me signed in" uses localStorage; otherwise sessionStorage.
  - Every page except the sign-in page has an inline `<head>` guard that redirects to `/login.html?next=…` before the page renders or asks for the camera. `next` only accepts same-site paths.
  - *Client-side only; not real security.* A public deployment needs a server-side login.
- **History.**
  - Per user, newest 500 entries.
  - Logged: translator text (when cleared, spoken or the page is left; never the same text twice in a row), every sentence result (updated to "confirmed/corrected by you" when taught), calculator results and practice-round scores.
- **Speech.** Letters, words, answers and sentences are spoken via the Web Speech API, with a toggle on each tool.
- **UI.** Dark/light theme (OS default, remembered, no flash of the wrong theme), responsive layout down to 390 px, scroll-reveal animations, shared header and footer partials injected at build time.
- **Privacy.** No video leaves the device. The only network downloads are the MediaPipe models and runtime (CDN, cached) and the static reference banks.

### 13.3 Size and performance
- Final JS bundles are small (e.g. sentence page ~11 kB, shared hand-tracking chunk ~155 kB, gzip ~46 kB).
- Removing TF.js cut the translator bundle from ~1 MB to 164 kB.
- Reference banks: ISL 7.9 MB, ASL 31 MB, each loaded on first use of that language.

---

## 14. Verification and parity testing

Because features and models are implemented twice (Python for training, TypeScript for the browser), every port was checked numerically on real data:

| Component | Check | Result |
|---|---|---|
| ASL image model v2 (TF.js vs Keras) | same photo | same letter; confidence diff 4.2e-7 |
| ISL landmark MLP (TS vs Keras) | real samples | max probability diff 6e-8 |
| GRU sentence model (TS vs Keras) | 8 held-out clips | max probability diff ~4e-6, 8/8 top-1 agree |
| DTW v1 (TS vs Python) | 24 query/reference pairs on the shipped bank | ~1e-7 |
| Sentence features v2 + v3 and DTW with band (TS vs Python) | 16 real clips incl. the no-body fallback path, both banks | features max abs diff 3.8e-6; DTW relative diff 1.8e-7 |
| Personal samples | 7 deterministic cases | 7/7 |

**Bugs caught this way before shipping.**
- A broadcasting-axis bug in the hand sorting.
- A rotation-shape bug in the sequence augmentation.
- The GRU's masking had to be applied per timestep, not as a "first N frames" cutoff, because 85% of clips have mid-clip dropouts.

**End-to-end tests** (headless Chrome with a fake camera):
- The page boots, and ISL/ASL banks and catalogs load.
- Recording, classification, teach/save/reload, and history logging all work.
- Sign-in redirects, wrong-password error, sign-out, and protected-page redirects all work.

A lesson from these tests: fake-camera videos loop and are re-encoded several times, which degrades landmarks. Clean-data offline tests are the reliable check.

---

## 15. Consolidated results

| Module | Model / method | Test | Metric | Result |
|---|---|---|---|---|
| ASL letters v1 | SL-MNIST CNN | own test set | top-1 | 99.80% (unusable live) |
| ASL letters v2 | MobileNetV2 | same-source / cross-source | top-1 | 88.9% / 27.2% |
| ASL letters v3 | MobileNetV2, 2 sources | same-source / cross-source | top-1 | 73.0% / 53.2% |
| **ASL letters v4** | **landmark MLP** | **cross-source, 672 photos** | **top-1 / top-3** | **82.3% / 93.0%** |
| ISL letters (image) | CNN | random split | top-1 | 100% (leaky) |
| ISL letters (landmark) | 2-hand MLP | temporal split, one signer | top-1 | 100% (optimistic) |
| **Digits** | **rules** | **872 real photos** | **accuracy** | **97%** |
| ISL sentences | GRU | 99 held-out videos | top-1/3/5 | 15.2 / 23.2 / 29.3% |
| ISL sentences | DTW v1 | LOO 687 | top-1/3/5 | 24.5 / 35.4 / 41.3% |
| ISL sentences | DTW v2 | LOO 687 | top-1/3/5 | 31.7 / 47.9 / 53.7% |
| ISL sentences | DTW v2 | cross-setup | top-1/3/5 | 10.8 / 18.2 / 24.2% |
| **ISL sentences** | **DTW v3 (shipped)** | **cross-setup** | **top-1/3/5** | **17.2 / 31.3 / 36.9%** |
| ISL sentences | DTW v3 | LOO 687 | top-1/3/5 | 27.7 / 44.3 / 52.3% |
| **ASL sentences** | **DTW v2, word chains** | **held-out signers** | **top-1/3/5** | **76.8 / 90.5 / 94.0%** (7 refs shipped: 77.9 / 87.3 / 90.4%) |
| ASL sentences | same | whole-clip, browser-style | top-1/3/5 | 69.3 / 87.1 / 91.1% |

Chance for 26 letters ≈ 3.8%; for 10 digits 10%; for 101 sentences ≈ 1%.

---

## 16. Key findings and lessons

1. **In-distribution accuracy is not real-world accuracy.** Examples: 99.8% on Sign Language MNIST was unusable live; 88.9% same-source became 27.2% cross-source; 32% sentence LOO became 11% on new signers. Every module needed a harder, more honest test before it could be improved in the right direction.
2. **Representation matters more than model size.** Replacing pixels with hand landmarks raised cross-source ASL accuracy from 53% to 82%, and let a ~50K-parameter MLP run instantly in plain JavaScript on a CPU.
3. **Data leakage hides in video-derived datasets.** Frames of one recording are near-duplicates. The ISL 100% was a leak, detected by measuring pixel differences inside a class.
4. **With ~5 examples per class, nearest-neighbour matching beats training.** DTW beat a bidirectional GRU (24.5% vs 15.2%); the GRU memorised its training set (97%).
5. **Body context helps sentences.** Where the hands are relative to the body (face vs. chest) and the arm pose carry signer-independent information that hand shape alone lacks. Trimming idle frames is critical: without it, cross-setup accuracy halves.
6. **Inter-signer variation is the core difficulty of sentence recognition on small data.** Two signers in the same room matched each other only ~7% of the time. This motivates personalisation: letting users add their own references.
7. **Features must suit the language and vocabulary.** ISL sentences benefited from coarse shape + strong arm features, while ASL word-chains needed detailed hand shape (v3: 59% vs v2: 78%). The system therefore stores the feature recipe with each bank.
8. **Implementing twice requires parity testing.** Numerical Python↔TypeScript checks caught real bugs before users did.

---

## 17. Limitations

- **ISL letters.** One signer, one session; accuracy on other people is unmeasured. The dataset's alphabet differs from the official ISLRTC chart for several letters.
- **ASL J and Z** are static approximations of motion signs.
- **Digits.** 7 and 8 are unvalidated, and the thresholds were tuned on the validation set itself.
- **ISL sentences.** Only 17% top-1 for new signers (31% top-3). The dataset has few takes per sentence, few signers and no signer IDs, and the cross-setup test is only a proxy for a new signer.
- **ASL sentences.** References are synthetic chains of isolated words, with no co-articulation. All tests are proxies, and the glosses need review by a Deaf ASL signer.
- **Vocabulary.** Fixed catalogs only; no open-vocabulary translation and no facial grammar (non-manual markers).
- **Personalisation.** Gain not measured.
- **Security.** Sign-in is client-side only (hash visible in the JS); data lives only in the browser.
- **Licensing.** WLASL is non-commercial (C-UDA), and its clips come from third-party sites. The ISL digit chart comes from a published paper; republishing permission should be confirmed.
- **Recording conditions.** Accuracy depends on lighting, background and the whole upper body being in frame.

---

## 18. Future work

- Collect a small multi-signer dataset (e.g. 10+ signers × 101 sentences) with signer IDs, to measure signer-independent and personalised accuracy properly.
- Learned metric: train a frame or sequence embedding (e.g. contrastive / triplet loss, or a transformer on pose sequences) on a large corpus (How2Sign, YouTube-ASL), then use DTW or kNN on the embeddings.
- Continuous recognition with CTC-style models for open vocabulary, and adding facial landmarks for non-manual grammar.
- Motion-aware letters (J, Z) using short landmark sequences.
- Server-side authentication and optional cloud sync of history and personal samples.
- User study with Deaf signers to validate the glosses, the UI and the real-world accuracy.

---

## 19. Reproducibility (files and commands)

All scripts run **from the repository root** with the project venv (`.venv_tfjs/Scripts/python.exe`, Python 3.11). Install packages with `python.exe -m pip install …` (the venv's `pip.exe` still points at the old C: path). Set `TEMP` and `TMP` to `tmp/`.

| Area | Scripts (in order) | Output |
|---|---|---|
| ASL letters | `extract_asl2.py` → `crop_asl2_hands.py` / `extract_crop_eitan.py` → `landmarks_asl.py` → `train_asl_landmarks.py` → `export_asl_landmarks.py` | `webapp/public/asl_landmarks/model.json` |
| ASL image baselines | `train_asl2_model.py`, `crossval_asl2.py`, `convert_asl2_for_tfjs.py` | `tfjs_model_asl/` (reference only) |
| ISL letters | `extract.py` → `landmarks_isl.py` → `train_isl_landmarks.py` | `webapp/public/isl_landmarks/model.json` |
| Digits | `numbers/extract_digit_validation.py`, `numbers/crop_digit_charts.py` | validation set, chart images |
| ISL sentences | `sentences/extract_landmarks.py` → `sentences/extract_pose.py` → `sentences/export_dtw_bank.py`; clips `sentences/make_sample_clips.py` | `webapp/public/sentence_model/`, `sentence_clips/` |
| ISL sentence evaluation | `dtw_lab.py` (numba LOO harness), `eval_dtw_pose.py` (LOO), `eval_cross.py` (cross-setup), experiments `dtw_exp_*.py`, `dtw_trim*.py`, `dtw_grid.py` | printed metrics, `dtw_result.json` |
| ASL sentences | `asl_sentences/extract_words.py` → `asl_sentences/build_bank.py eval \| export` → `asl_sentences/make_sample_clips.py`; tests `eval_hard.py`, `eval_clips.py` | `webapp/public/sentence_model_asl/`, `sentence_clips_asl/` |
| Parity | `sentences/check_pose_parity.py` + `node --experimental-strip-types webapp/scripts/check-pose-parity.mts`; `webapp/scripts/check-personal.mts` | PARITY OK / PERSONAL OK |
| Web app | `cd webapp && npm install && npm run dev` (dev) / `npm run build` (static site in `webapp/dist`) | — |

Full chronological engineering log: `README.md` (Updates 1–16) and `PROJECT_NOTES.md`.

---

## 20. Suggested paper structure

1. **Title idea.** *"Hastavāṇī: Privacy-Preserving, Browser-Based Recognition of ASL and ISL Letters, Digits and Sentences from Webcam Landmarks"*
2. **Abstract.** Problem; landmark-based, client-only system; four modules. Headline results: ASL letters 82.3% cross-source, digits 97%, ISL sentences 17.2%/31.3% (top-1/top-3) for new signers against 1% chance, ASL word-chain sentences 76.8%. Main lesson: honest evaluation.
3. **Introduction.** Motivation, gap (server/GPU dependence, optimistic evaluation), contributions:
   - landmark pipeline running in the browser;
   - rule-based digits;
   - pose-aware DTW with idle trimming for small-data sentence recognition;
   - word-chaining for ASL sentences without sentence data;
   - personalisation;
   - an evaluation methodology exposing optimistic splits.
4. **Related work.** MediaPipe-based SLR, CNN fingerspelling (Sign Language MNIST), WLASL/MS-ASL isolated sign recognition, How2Sign/continuous SLR, DTW for gesture recognition, ISL recognition (Ekbote & Joshi).
5. **Datasets** (Section 5 of this file).
6. **Methodology** (Sections 6–12).
7. **Experiments and results** (Sections 10.2–10.4, 11, 15; include the ASL-letter progression table and the sentence ablation table).
8. **Discussion** (Section 16).
9. **Limitations and ethics** (Section 17; privacy, dataset licences, not a substitute for interpreters).
10. **Conclusion and future work** (Section 18).

Suggested figures: the architecture diagram (Section 3), the ASL-letter accuracy progression (bar chart), the sentence ablation (bar chart: GRU → DTW v1 → v2 → v3, LOO vs cross-setup), screenshots of the translator and sentence builder, and an example feature/trim illustration.

---

## 21. References

- Google MediaPipe — Hand Landmarker and Pose Landmarker (MediaPipe Tasks). https://ai.google.dev/edge/mediapipe
- Sign Language MNIST (Kaggle, datamunge). https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Marxulia, *asl_sign_languages_alphabets_v03* (Hugging Face). https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03
- EitanG98, *asl_letters* (Hugging Face). https://huggingface.co/datasets/EitanG98/asl_letters
- Hemg, *Indian_sign_language_dataset* (Hugging Face; mirror of Kaggle vaishnaviasonawane/indian-sign-language-dataset). https://huggingface.co/datasets/Hemg/Indian_sign_language_dataset
- ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition (Mendeley Data). https://data.mendeley.com/datasets/kcmpdxky7p/1
- D. Li, C. Rodriguez, X. Yu, H. Li, "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison" (WLASL), WACV 2020. https://dxli94.github.io/WLASL/
- J. Ekbote, M. Joshi, "Indian sign language recognition using ANN and SVM classifiers", 2017 (ISL numeral chart).
- M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
- H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition", IEEE Trans. ASSP, 1978 (DTW, Sakoe-Chiba band).
- A. Duarte et al., "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language", CVPR 2021 (considered, not used).
- *"Gesture Recognition in Sign Language Translation"* — the paper whose webcam → hand detection → CNN with spatial attention pipeline the first version reproduced (cite with full details from your copy).
