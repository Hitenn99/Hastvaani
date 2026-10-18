// Sentence classifier: k-nearest-neighbour matching by Dynamic Time Warping (DTW) against a
// bank of every recorded training video's real (hand-detected) per-frame landmark sequence
// (plus a mirrored copy of each) -- no trained model, no weights, just "which reference
// clip's motion is closest". Chosen over a trained GRU sequence model after both were
// measured on the SAME rigorous leave-one-out test across all 687 training videos: this
// beat the GRU (see model_assets/sentences/eval_dtw_full.py and README "Update 11") because
// with only ~5 videos per sentence, a neural net has too little data to learn a general
// decision boundary, while comparing directly against real examples doesn't need to
// generalise past memorising -- it only needs a good distance metric.
//
// The DTW recursion here MUST stay numerically identical to `dtw_nb()` in
// model_assets/sentences/dtw_lab.py (checked by model_assets/sentences/check_pose_parity.py).
// Since v2 the bank holds hands + body-pose features (features.ts clipFeatures /
// pose_features.py) and the alignment is limited to a Sakoe-Chiba band (dtw_meta.json "band").

import type { FeatureSpec } from './features';

export interface DtwEntry {
  offset: number; // frame index into the shared Float32Array bank (NOT byte offset)
  length: number; // number of frames (timesteps) in this reference sequence
  classIndex: number;
}

export interface DtwBank {
  featureLen: number;
  bank: Float32Array; // all reference sequences' frames concatenated, each frame featureLen floats
  entries: DtwEntry[];
  labels: string[];
  band: number; // Sakoe-Chiba band as a fraction of the reference length (<= 0: none)
  features: FeatureSpec; // how queries must be featurised to match this bank
}

// One bank per language folder ('/sentence_model' = ISL, '/sentence_model_asl' = ASL), each
// fetched once; `bank` is whichever was loaded last, and is what classifySentence() uses.
const banks = new Map<string, Promise<DtwBank>>();
let bank: DtwBank | null = null;

async function fetchBank(base: string): Promise<DtwBank> {
  const [meta, buf, labels] = await Promise.all([
    fetch(`${base}/dtw_meta.json`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch DTW reference index (${r.status})`);
      return r.json() as Promise<{ featureLen: number; band?: number; features?: FeatureSpec; entries: DtwEntry[] }>;
    }),
    fetch(`${base}/dtw_bank.bin`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch DTW reference bank (${r.status})`);
      return r.arrayBuffer();
    }),
    fetch(`${base}/labels.json`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch sentence labels (${r.status})`);
      return r.json() as Promise<string[]>;
    }),
  ]);
  return {
    featureLen: meta.featureLen,
    bank: new Float32Array(buf),
    entries: meta.entries,
    labels,
    band: meta.band ?? 0,
    features: meta.features ?? { version: 2 },
  };
}

/** Loads (once) and activates the reference bank in `base`. */
export async function loadDtwBank(base = '/sentence_model'): Promise<DtwBank> {
  if (!banks.has(base)) {
    const p = fetchBank(base);
    banks.set(base, p);
    p.catch(() => banks.delete(base)); // allow a retry after a failed download
  }
  bank = null;
  const loaded = await banks.get(base)!;
  bank = loaded;
  return loaded;
}

export function dtwBankReady(): boolean {
  return bank !== null;
}

/** Feature recipe of the active bank (pass to features.ts clipFeatures). */
export function activeFeatureSpec(): FeatureSpec {
  if (!bank) throw new Error('DTW reference bank not loaded yet');
  return bank.features;
}

/** The `frameIndex`-th frame (featureLen floats) of a sequence that starts at `seqOffset`
 * frames into `flat` -- `seqOffset` is in FRAMES, not floats (multiplied in here). */
function frameAt(flat: Float32Array, featureLen: number, seqOffset: number, frameIndex: number): Float32Array {
  const start = (seqOffset + frameIndex) * featureLen;
  return flat.subarray(start, start + featureLen);
}

/**
 * DTW distance between two variable-length sequences of equal-dimension frame vectors,
 * normalised by path length (so longer clips aren't unfairly penalised). Same recursion as
 * the Python reference: cost(i,j) = ||a_i - b_j||^2 (squared Euclidean per frame),
 * D[i,j] = cost(i,j) + min(D[i-1,j-1], D[i-1,j], D[i,j-1]).
 * `a`/`b` are the flat arrays their sequence lives in; `aOffset`/`bOffset` (in frames, not
 * floats) locate where that particular sequence starts within them.
 */
function dtwDistance(
  a: Float32Array, aOffset: number, aLen: number,
  b: Float32Array, bOffset: number, bLen: number,
  featureLen: number,
  band: number,
): number {
  if (aLen === 0 || bLen === 0) return 1e9;
  // D is (aLen+1) x (bLen+1), but only two rows are ever needed at once.
  let prevRow = new Float64Array(bLen + 1).fill(Infinity);
  let curRow = new Float64Array(bLen + 1).fill(Infinity);
  prevRow[0] = 0;
  for (let i = 1; i <= aLen; i++) {
    curRow.fill(Infinity);
    // Only cells near the (relative) diagonal: same int() truncation as the Python.
    let lo = 1, hi = bLen;
    if (band > 0) {
      const centre = (i * bLen) / aLen;
      lo = Math.max(1, Math.trunc(centre - band * bLen));
      hi = Math.min(bLen, Math.trunc(centre + band * bLen) + 1);
    }
    const ai = frameAt(a, featureLen, aOffset, i - 1);
    for (let j = lo; j <= hi; j++) {
      const bj = frameAt(b, featureLen, bOffset, j - 1);
      let cost = 0;
      for (let k = 0; k < featureLen; k++) {
        const diff = ai[k] - bj[k];
        cost += diff * diff;
      }
      const best = Math.min(prevRow[j - 1], prevRow[j], curRow[j - 1]);
      curRow[j] = cost + best;
    }
    const tmp = prevRow;
    prevRow = curRow;
    curRow = tmp;
  }
  return prevRow[bLen] / (aLen + bLen);
}

export interface SentencePrediction {
  sentence: string;
  index: number;
  confidence: number;
}

/** Test-only: raw DTW distance from a query to one specific reference bank entry (by its
 * position in dtw_meta.json's `entries` array), for numeric parity-checking against Python's
 * dtw_dist() on the exact shipped data/algorithm -- not used by the app itself. */
export function _distanceToEntryForTest(frames: Float32Array[], entryIndex: number): number {
  if (!bank) throw new Error('DTW reference bank not loaded yet');
  const { featureLen, bank: flat, entries } = bank;
  const entry = entries[entryIndex];
  const query = new Float32Array(frames.length * featureLen);
  frames.forEach((f, i) => query.set(f, i * featureLen));
  return dtwDistance(query, 0, frames.length, flat, entry.offset, entry.length, featureLen, bank.band);
}

/**
 * `frames`: the query clip's per-frame feature vectors from features.ts `clipFeatures()` with
 * this bank's activeFeatureSpec() -- the same recipe (ported from Python) that built the bank.
 * `personal`: the user's own saved recordings (personal.ts), matched as extra references.
 */
export function classifySentence(
  frames: Float32Array[],
  personal: { classIndex: number; frames: Float32Array[] }[] = [],
): SentencePrediction[] {
  if (!bank) throw new Error('DTW reference bank not loaded yet');
  const { featureLen, bank: flat, entries, labels } = bank;
  const query = new Float32Array(frames.length * featureLen);
  frames.forEach((f, i) => query.set(f, i * featureLen));

  const distByClass = new Map<number, number>();
  const consider = (classIndex: number, dist: number) => {
    const best = distByClass.get(classIndex);
    if (best === undefined || dist < best) distByClass.set(classIndex, dist);
  };
  for (const entry of entries) {
    consider(entry.classIndex, dtwDistance(query, 0, frames.length, flat, entry.offset, entry.length, featureLen, bank.band));
  }
  for (const p of personal) {
    if (!p.frames.length || p.frames[0].length !== featureLen) continue;
    const ref = new Float32Array(p.frames.length * featureLen);
    p.frames.forEach((f, i) => ref.set(f, i * featureLen));
    consider(p.classIndex, dtwDistance(query, 0, frames.length, ref, 0, p.frames.length, featureLen, bank.band));
  }

  const ranked = [...distByClass.entries()].sort((a, b) => a[1] - b[1]);
  // Turn distances into a display-friendly "confidence": softmax over negative distance,
  // scaled so the gap between a clear best match and the rest is visually meaningful.
  const dists = ranked.map(([, d]) => d);
  const minD = Math.min(...dists);
  const scale = Math.max(minD, 1e-6);
  const weights = dists.map((d) => Math.exp(-((d - minD) / scale) * 3));
  const total = weights.reduce((a, b) => a + b, 0);

  return ranked.map(([classIndex], i) => ({
    sentence: labels[classIndex],
    index: classIndex,
    confidence: weights[i] / total,
  }));
}

