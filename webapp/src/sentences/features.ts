// Per-frame two-hand landmark features for the sentence recognizer.
//
// MUST stay numerically identical to `normalize()` in model_assets/sentences/train_sentences.py
// (parity is checked offline before every model export -- see that script's header comment).
// This is the SAME convention as the single-frame ISL letter features in model.ts
// (predictIslLandmarks), just applied to every frame of a recorded sequence.

export interface Pt {
  x: number;
  y: number;
  z: number;
}

export const FEATURE_LEN = 128;

/** Sorts up to two hands by wrist x; a missing hand becomes an all-zero, absent slot. */
function sortedPair(hands: Pt[][]): { pts: number[][][]; mask: [number, number] } {
  const sorted = [...hands].sort((a, b) => a[0].x - b[0].x).slice(0, 2);
  const pts: number[][][] = [];
  const mask: [number, number] = [0, 0];
  for (let slot = 0; slot < 2; slot++) {
    const hand = sorted[slot];
    if (hand) {
      pts.push(hand.map((p) => [p.x, p.y, p.z]));
      mask[slot] = 1;
    } else {
      pts.push(Array.from({ length: 21 }, () => [0, 0, 0]));
    }
  }
  return { pts, mask };
}

/**
 * One video frame's hands -> a 128-float feature vector: [hand0 (63), hand1 (63),
 * present0, present1]. A frame with no hands returns an all-zero vector (matches the
 * padding convention the training data and the Masking(mask_value=0) layer both use).
 */
export function frameFeatures(hands: Pt[][]): Float32Array {
  const out = new Float32Array(FEATURE_LEN);
  if (hands.length === 0) return out;
  const { pts, mask } = sortedPair(hands);
  if (mask[0] === 0 && mask[1] === 0) return out;

  let cx = 0, cy = 0, cz = 0, count = 0;
  for (let slot = 0; slot < 2; slot++) {
    if (!mask[slot]) continue;
    for (const p of pts[slot]) {
      cx += p[0]; cy += p[1]; cz += p[2]; count++;
    }
  }
  cx /= count; cy /= count; cz /= count;

  let scale = 1e-6;
  const centered: number[][][] = [[], []];
  for (let slot = 0; slot < 2; slot++) {
    for (const p of pts[slot]) {
      const q = mask[slot] ? [p[0] - cx, p[1] - cy, p[2] - cz] : [0, 0, 0];
      centered[slot].push(q);
      if (mask[slot]) scale = Math.max(scale, Math.hypot(q[0], q[1]));
    }
  }

  for (let slot = 0; slot < 2; slot++) {
    for (let i = 0; i < 21; i++) {
      const q = centered[slot][i];
      out[slot * 63 + i * 3] = q[0] / scale;
      out[slot * 63 + i * 3 + 1] = q[1] / scale;
      out[slot * 63 + i * 3 + 2] = q[2] / scale;
    }
  }
  out[126] = mask[0];
  out[127] = mask[1];
  return out;
}

/** A recorded clip's per-frame hand snapshots -> (length, 128) feature sequence. */
export function sequenceFeatures(frames: Pt[][][]): Float32Array[] {
  return frames.map(frameFeatures);
}

// ---------- Clip features for the DTW banks ----------
//
// Two versions, picked by the bank's dtw_meta.json "features.version":
//   v2 (ASL bank): clip_features() in model_assets/sentences/pose_features_v2.py
//   v3 (ISL bank): clip_features() in model_assets/sentences/pose_features.py -- see that file
//                  for why (tuned for signers the bank has never seen)
// Both MUST stay numerically identical to their Python versions (checked by
// model_assets/sentences/check_pose_parity.py).

export interface FeatureSpec {
  version: 2 | 3;
  wShape?: number;
  wLoc?: number;
  wArm?: number;
  trimThr?: number;
  smooth?: number; // v3.1 moving-average window over kept frames (1 / absent = off)
  wVel?: number; // v3.1 weight of the appended arm velocity (0 / absent = off)
}

/** A whole recorded clip -> its feature sequence for a bank built with `spec`. */
export function clipFeatures(frames: RawFrame[], width: number, height: number, spec: FeatureSpec): Float32Array[] {
  return spec.version === 3 ? clipFeaturesV3(frames, width, height, spec) : clipFeaturesV2(frames, width, height);
}

export const CLIP_FEATURE_LEN = 128 + 4 + 8;
const W_LOC = 1.0;
const W_ARM = 2.0;
const TRIM_THR = 0.6;
const L_SH = 11, R_SH = 12, L_EL = 13, R_EL = 14, L_WR = 15, R_WR = 16;
const PALM = [0, 5, 9, 13, 17];

/** One sampled video frame: every detected hand, and the body landmarks (null if no body). */
export interface RawFrame {
  hands: Pt[][];
  pose: Pt[] | null;
}

/** numpy.median semantics (mean of the two middle values for an even count). */
function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** v2: a whole recorded clip -> its (n, CLIP_FEATURE_LEN) feature sequence (may be empty). */
function clipFeaturesV2(frames: RawFrame[], width: number, height: number): Float32Array[] {
  const T = frames.length;
  const sorted = frames.map((f) => sortedPair(f.hands));
  const px = (p: Pt | number[]): [number, number] =>
    Array.isArray(p) ? [p[0] * width, p[1] * height] : [p.x * width, p.y * height];

  // Body reference: shoulder-midpoint origin + shoulder-width scale (medians over the clip).
  const bodyIdx = frames.map((f, t) => (f.pose ? t : -1)).filter((t) => t >= 0);
  let ox: number, oy: number, scale: number;
  if (bodyIdx.length) {
    const mx: number[] = [], my: number[] = [], widths: number[] = [];
    for (const t of bodyIdx) {
      const [lx, ly] = px(frames[t].pose![L_SH]);
      const [rx, ry] = px(frames[t].pose![R_SH]);
      mx.push((lx + rx) / 2);
      my.push((ly + ry) / 2);
      widths.push(Math.hypot(lx - rx, ly - ry));
    }
    ox = median(mx);
    oy = median(my);
    scale = Math.max(median(widths), 1e-3);
  } else {
    let sx = 0, sy = 0, cnt = 0;
    const sizes: number[] = [];
    for (const { pts, mask } of sorted) {
      for (let h = 0; h < 2; h++) {
        if (!mask[h]) continue;
        for (const p of pts[h]) {
          const [x, y] = px(p);
          sx += x; sy += y; cnt++;
        }
        const [wx, wy] = px(pts[h][0]);
        const [mx9, my9] = px(pts[h][9]);
        sizes.push(Math.hypot(mx9 - wx, my9 - wy));
      }
    }
    ox = cnt ? sx / cnt : 0;
    oy = cnt ? sy / cnt : 0;
    scale = Math.max(4.0 * (sizes.length ? median(sizes) : 1.0), 1e-3);
  }

  // Trim to the "active" window: first..last frame where a wrist is raised.
  let start = 0, end = T;
  if (bodyIdx.length) {
    const active: number[] = [];
    for (const t of bodyIdx) {
      const pose = frames[t].pose!;
      const wy = Math.min(px(pose[L_WR])[1] - oy, px(pose[R_WR])[1] - oy) / scale;
      if (wy < TRIM_THR) active.push(t);
    }
    if (active.length >= 4) {
      start = active[0];
      end = active[active.length - 1] + 1;
    }
  }

  const out: Float32Array[] = [];
  for (let t = start; t < end; t++) {
    const { pts, mask } = sorted[t];
    const pose = frames[t].pose;
    if (!mask[0] && !mask[1] && !pose) continue;
    const f = new Float32Array(CLIP_FEATURE_LEN);
    f.set(frameFeatures(frames[t].hands), 0);
    for (let h = 0; h < 2; h++) {
      if (!mask[h]) continue;
      let sx = 0, sy = 0;
      for (const i of PALM) {
        const [x, y] = px(pts[h][i]);
        sx += x; sy += y;
      }
      f[128 + h * 2] = (W_LOC * (sx / PALM.length - ox)) / scale;
      f[128 + h * 2 + 1] = (W_LOC * (sy / PALM.length - oy)) / scale;
    }
    if (pose) {
      // arms (elbow, wrist) ordered by wrist x, like the hands; stable on ties
      let arms = [[pose[L_EL], pose[L_WR]], [pose[R_EL], pose[R_WR]]];
      if (px(arms[1][1])[0] < px(arms[0][1])[0]) arms = [arms[1], arms[0]];
      let k = 132;
      for (const arm of arms) {
        for (const p of arm) {
          const [x, y] = px(p);
          f[k++] = (W_ARM * (x - ox)) / scale;
          f[k++] = (W_ARM * (y - oy)) / scale;
        }
      }
    }
    out.push(f);
  }
  return out;
}

// ---------- v3 ----------
const V3_LEN = 40 + 2 + 4 + 8;
const SHAPE_PTS = [4, 8, 12, 16, 20, 3, 6, 10, 14, 18];

/** Body reference (origin + scale) and idle-trim window for v3 (same logic as v2's, inline above). */
function bodyFrameAndTrim(
  frames: RawFrame[],
  sorted: { pts: number[][][]; mask: [number, number] }[],
  px: (p: Pt | number[]) => [number, number],
  trimThr: number,
) {
  const T = frames.length;
  const bodyIdx = frames.map((f, t) => (f.pose ? t : -1)).filter((t) => t >= 0);
  let ox: number, oy: number, scale: number;
  if (bodyIdx.length) {
    const mx: number[] = [], my: number[] = [], widths: number[] = [];
    for (const t of bodyIdx) {
      const [lx, ly] = px(frames[t].pose![L_SH]);
      const [rx, ry] = px(frames[t].pose![R_SH]);
      mx.push((lx + rx) / 2);
      my.push((ly + ry) / 2);
      widths.push(Math.hypot(lx - rx, ly - ry));
    }
    ox = median(mx);
    oy = median(my);
    scale = Math.max(median(widths), 1e-3);
  } else {
    let sx = 0, sy = 0, cnt = 0;
    const sizes: number[] = [];
    for (const { pts, mask } of sorted) {
      for (let h = 0; h < 2; h++) {
        if (!mask[h]) continue;
        for (const p of pts[h]) {
          const [x, y] = px(p);
          sx += x; sy += y; cnt++;
        }
        const [wx, wy] = px(pts[h][0]);
        const [mx9, my9] = px(pts[h][9]);
        sizes.push(Math.hypot(mx9 - wx, my9 - wy));
      }
    }
    ox = cnt ? sx / cnt : 0;
    oy = cnt ? sy / cnt : 0;
    scale = Math.max(4.0 * (sizes.length ? median(sizes) : 1.0), 1e-3);
  }
  let start = 0, end = T;
  if (bodyIdx.length) {
    const active: number[] = [];
    for (const t of bodyIdx) {
      const pose = frames[t].pose!;
      const wy = Math.min(px(pose[L_WR])[1] - oy, px(pose[R_WR])[1] - oy) / scale;
      if (wy < trimThr) active.push(t);
    }
    if (active.length >= 4) {
      start = active[0];
      end = active[active.length - 1] + 1;
    }
  }
  return { ox, oy, scale, start, end };
}

function clipFeaturesV3(frames: RawFrame[], width: number, height: number, spec: FeatureSpec): Float32Array[] {
  const wShape = spec.wShape ?? 2.0, wLoc = spec.wLoc ?? 0.5, wArm = spec.wArm ?? 8.0, trimThr = spec.trimThr ?? 0.4;
  const sorted = frames.map((f) => sortedPair(f.hands));
  const px = (p: Pt | number[]): [number, number] =>
    Array.isArray(p) ? [p[0] * width, p[1] * height] : [p.x * width, p.y * height];
  const { ox, oy, scale, start, end } = bodyFrameAndTrim(frames, sorted, px, trimThr);

  const out: Float32Array[] = [];
  for (let t = start; t < end; t++) {
    const { pts, mask } = sorted[t];
    const pose = frames[t].pose;
    if (!mask[0] && !mask[1] && !pose) continue;
    const f = new Float32Array(V3_LEN);
    for (let h = 0; h < 2; h++) {
      if (!mask[h]) continue;
      // hand shape: fingertips + middle joints relative to the wrist, / palm length
      const [wx, wy] = px(pts[h][0]);
      const [kx, ky] = px(pts[h][9]);
      const palmLen = Math.hypot(kx - wx, ky - wy) + 1e-6;
      SHAPE_PTS.forEach((i, j) => {
        const [x, y] = px(pts[h][i]);
        f[h * 20 + j * 2] = ((wShape / 3) * (x - wx)) / palmLen;
        f[h * 20 + j * 2 + 1] = ((wShape / 3) * (y - wy)) / palmLen;
      });
      // palm centre relative to the body
      let sx = 0, sy = 0;
      for (const i of PALM) {
        const [x, y] = px(pts[h][i]);
        sx += x; sy += y;
      }
      f[42 + h * 2] = (wLoc * (sx / PALM.length - ox)) / scale;
      f[42 + h * 2 + 1] = (wLoc * (sy / PALM.length - oy)) / scale;
    }
    f[40] = mask[0];
    f[41] = mask[1];
    if (pose) {
      // arms (elbow, wrist) ordered by wrist x, like the hands; stable on ties
      let arms = [[pose[L_EL], pose[L_WR]], [pose[R_EL], pose[R_WR]]];
      if (px(arms[1][1])[0] < px(arms[0][1])[0]) arms = [arms[1], arms[0]];
      let k = 46;
      for (const arm of arms) {
        for (const p of arm) {
          const [x, y] = px(p);
          f[k++] = (wArm * (x - ox)) / scale;
          f[k++] = (wArm * (y - oy)) / scale;
        }
      }
    }
    out.push(f);
  }
  return addMotion(out, spec.smooth ?? 1, spec.wVel ?? 0);
}

const ARM_FROM = 46, ARM_LEN = 8;

/** v3.1, = add_motion() in pose_features.py: edge-padded moving average over `w` frames (clips
 * shorter than `w` unchanged), then the arm features' velocity x wVel appended. */
function addMotion(frames: Float32Array[], w: number, wVel: number): Float32Array[] {
  const T = frames.length;
  let f = frames;
  if (w > 1 && T >= w) {
    const left = w >> 1;
    f = frames.map((_, t) => {
      const acc = new Float64Array(V3_LEN);
      for (let m = 0; m < w; m++) {
        const src = frames[Math.min(Math.max(t - left + m, 0), T - 1)];
        for (let k = 0; k < V3_LEN; k++) acc[k] += src[k];
      }
      return Float32Array.from(acc, (v) => v / w);
    });
  }
  if (!wVel) return f;
  return f.map((row, t) => {
    const o = new Float32Array(V3_LEN + ARM_LEN);
    o.set(row, 0);
    if (T >= 2) {
      const a = t === 0 ? 0 : t === T - 1 ? T - 2 : t - 1;
      const b = t === 0 ? 1 : t === T - 1 ? T - 1 : t + 1;
      const div = t === 0 || t === T - 1 ? 1 : 2;
      for (let k = 0; k < ARM_LEN; k++) o[V3_LEN + k] = (wVel * (f[b][ARM_FROM + k] - f[a][ARM_FROM + k])) / div;
    }
    return o;
  });
}
