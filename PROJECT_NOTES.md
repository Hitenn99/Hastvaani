# Sign Language 1 — project handoff notes

Project folder was moved from its original location to another drive because the first drive was full.

## What this project is
Real-time sign language recognition web app (ASL + ISL).
- `webapp/` — Vite + TypeScript app (`npm run dev`, serves on http://localhost:5173). Source in `webapp/src/`
  (handTracking.ts, model.ts, preprocess.ts, smoothing.ts, speech.ts, main.ts). TF.js models in `webapp/public/tfjs_model_asl/` (model.json, shards, labels.json).
- `model_assets/` — Python training/conversion scripts and Keras models (train_asl2_model.py, train_isl_model.py,
  crossval_asl2.py, convert_*_for_tfjs.py, kaggle_*_model.keras, hand_landmarker.task).
- `asl_data/`, `isl_data/`, `crossval_data/` — datasets.
- `.venv_tfjs/` — Python 3.11 venv (works from E:; call `.venv_tfjs/Scripts/python.exe` directly).
- Nothing in the code hardcodes absolute paths (checked).

## State when we stopped
- A retraining run on the combined dataset crashed. Likely cause: C: disk was full (write failure), not a code bug.
- Known bug found and already reasoned about: augmentation layers were applied incorrectly in the parallel data pipeline
  (should be applied per batch, not per image across threads). Verify this in the training script and fix if not yet applied.
- TODO: add checkpointing (ModelCheckpoint) to training so a crash does not lose progress.
- TODO: when training, redirect TEMP/TMP (and TF cache) to a folder on E: since C: has < 1GB free.
- Copy to E: was verified (95,113 files, 0 failed; venv and dev server work from E:).

## Preferences
- User writes in Hinglish; reply in the same casual style, keep it short.
- Ask before deleting anything outside this project.

## State (2026-09-21, stopped for battery)
- ASL: DONE. Landmark MLP live (`webapp/public/asl_landmarks/`, `model.ts` predictLandmarks), 82.3% cross-source. User confirmed it works perfectly.
- ISL: DONE (landmark MLP trained+integrated, parity verified; needs live webcam check by user; only single-session data so real accuracy unknown). Old plan below kept for reference. Plan = same landmark approach, 2 hands (128 features). 
  1. `.venv_tfjs/Scripts/python.exe model_assets/landmarks_isl.py` (resumable: per-letter cache in `tmp/isl_lm/`, delete nothing; earlier 4-worker run hit MemoryError, now 2 workers). Produces `model_assets/landmarks_isl.npz`.
  2. `model_assets/train_isl_landmarks.py` (already written; temporal split; exports `webapp/public/isl_landmarks/model.json`).
  3. TODO: add ISL branch in `webapp/src/model.ts` (2-hand feature normalization identical to `normalize()` in train script: sort hands by wrist x, subtract centroid, divide by max xy dist, 126 floats + 2 presence flags) and use it in `main.ts` loop for mode 'isl' (pass all `result.landmarks`).
  4. Update README, keep ISL dataset chart (user's attached figure) as reference; do NOT mix the official ISLRTC PDF chart into training (different signs).
- Math page (digits 0-9) is DONE (see README "Update 6"). Sentence builder (ISL, 101-sentence catalog) is DONE, real held-out accuracy top-1 15/top-3 23/top-5 29% (see README "Update 9") -- low, honestly reported, not a bug; GRU forward pass in webapp/src/sentences/model.ts is numerically verified against Keras. Reference clips added (Update 10). IMPORTANT: use "python.exe -m pip install", never "Scripts/pip" directly -- pip.exe still points at the old C: venv path.
- If retraining the sentence model: re-run model_assets/sentences/export_sentence_model.py after any train_sentences.py change, and re-check parity (see model.ts header) since the export format is exact and unforgiving.
- Run scripts with TEMP/TMP set to `$PWD/tmp` (C: is full). Dev server: `cd webapp && npm run dev`.

## State (2026-09-24): sentence builder accuracy improved -- DTW replaced the GRU. DONE.
See README "Update 11" for the full story. Short version: user asked to improve sentence-builder accuracy;
DTW k-NN (classical, no training) beat the GRU on a rigorous full-687-video leave-one-out test --
**top-1 24.5%, top-3 35.4%, top-5 41.3%** (GRU was 15.2%/23.2%/29.3%). `sentence.ts` now uses
`webapp/src/sentences/dtw.ts` (loads `webapp/public/sentence_model/dtw_bank.bin` + `dtw_meta.json`, 23.3 MB).
Numerically verified against Python to ~1e-7 on real data, and a clean end-to-end query correctly recovered its
sentence at 100% confidence. `model.ts` (the GRU) is unused dead code now but deliberately left in the repo for
reference. `how-it-works.html`/`about.html` already updated with the real DTW numbers -- don't reintroduce the
old 15/23/29% figures.

One nuance worth remembering if this comes up again: a headless-browser smoke test that fed a *triple
re-encoded* video (mp4 -> ffmpeg frames -> fake camera -> re-detected landmarks) scored badly on that one clip
(rank 56/101) purely from accumulated compression artifacts across three re-encoding hops -- confirmed not a
bug by feeding clean, once-extracted landmark data through the same code path, which worked perfectly. A real
user's webcam is a single capture hop, so this shouldn't recur, but don't be alarmed if a similarly
multi-re-encoded test clip scores badly in the future -- check with clean data first before assuming a bug.

**Gotcha to remember:** `.venv_tfjs/Scripts/pip` (the .exe) still has the OLD drive path baked in — always
install with `.venv_tfjs/Scripts/python.exe -m pip install ...`, never `Scripts/pip install ...` directly.
`dtaidistance` was pip-installed (correctly, via `-m pip`) but ended up unused — the shipped DTW is a
hand-written O(T1*T2) numpy DP (`dtw_dist()` in `eval_dtw_full.py`) / TS port (`dtw.ts`), not that library, so
it's a safe uninstall later if anyone wants to trim the venv (not urgent).

Also: `model_assets/sentences/eval_dtw_full.py` and `eval_knn.py`/`eval_knn2.py` all define reusable functions
at module scope -- always guarded with `if __name__ == '__main__':` around the expensive part now (a real bug
hit this session: importing from an unguarded eval_dtw_full.py silently re-ran its whole 17-minute leave-one-out
loop as an import side effect). Keep that guard if these scripts are edited.

## State (2026-10-01): sentence builder v2 DONE (README "Update 12")
Top-1 24.5 -> 31.7%, top-3 35.4 -> 47.9%, top-5 41.3 -> 53.7% (full 687 LOO). Added MediaPipe pose
(body-relative hand position + elbows/wrists) and idle-frame trimming. Features: `pose_features.py` <->
`features.ts clipFeatures` -- change both + re-run export_dtw_bank.py + parity check together.
Fast experiments: `model_assets/sentences/dtw_lab.py` (numba, ~13 s per full LOO; numba installed in venv).
Needs user to test live on webcam (head + shoulders in view).

## State (2026-10-01): ASL sentence mode DONE (README "Update 13")
WLASL archive (downloaded from the WLASL site) was extracted to `asl_sentences_data/wlasl/`. ASL sentences = chained WLASL word
clips (catalog `model_assets/asl_sentences/catalog.json`, glosses need review by an ASL signer). Proxy accuracy
~77% top-1 on held-out signers; no real-sentence test possible. WLASL is C-UDA (non-commercial) -- user told.
To change the catalog: edit catalog.json -> extract_words.py (new words only, cached) -> build_bank.py export -> make_sample_clips.py.

## State (2026-10-01): ISL "not accurate" fix (README "Update 14")
Real new-signer accuracy was ~11% (LOO numbers were optimistic). ISL now v3 features (cross-setup 17/31/37%),
ASL stays v2 (per-bank "features" in dtw_meta.json). Added "Teach it your signing" (personal samples in
localStorage). Always judge sentence changes with eval_cross.py (ISL) / build_bank.py eval (ASL), not LOO.
Backups of v2 in tmp/*_v2_backup*. Needs a real webcam test by the user, incl. teaching 2-3 sentences.

## State (2026-10-01): sign-in + history DONE (README "Update 15")
Demo login demo / demo1234 (client-side only; user told it's not real security). WHOLE site needs sign-in; login page shows no nav (user asked). History per user in
localStorage; history.html has filters/search/delete/clear. Protected pages list lives in vite.config.ts.

## State (2026-10-09): ISL sentences v3.1 (README "Update 17")
User asked for 90% -- told them honestly it's not reachable with this data. Shipped smoothing + arm velocity
(ISL cross-setup top-1 17 -> 20%). ASL unchanged. Per-sentence lists: per_sentence_accuracy.py. Old v3 bank in
tmp/sentence_model_v3_backup/. Needs a live webcam check; personal samples must be re-taught.
