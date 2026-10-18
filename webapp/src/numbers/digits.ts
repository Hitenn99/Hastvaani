// Digit (0-9) recognition from MediaPipe hand landmarks -- rule based, no training data needed.
//
// This module is intentionally separate from the alphabet recognizer (model.ts and
// friends): nothing here is shared with, or changes, the A-Z models.
//
// How it works: each digit sign is described by which fingers are extended, whether the
// thumb is out or tucked, and which fingertip (if any) the thumb touches. Those measurements
// are rotation/scale invariant (ratios and angles), so the hand can be tilted or at any distance.
// Digit handshapes follow the ASL and ISL numeral charts (they coincide for 0-9).

export interface Pt {
  x: number;
  y: number;
  z: number;
}

export interface DigitResult {
  digit: number | null; // null = no confident digit
  confidence: number; // 0..1
  probs: number[]; // length 10
  features: Features;
}

export interface Features {
  ext: [number, number, number, number]; // index, middle, ring, little: 0 folded .. 1 straight
  thumbOut: number; // 0 tucked .. 1 stretched out
  touch: [number, number, number, number]; // thumb tip touching index..little fingertip: 0..1
}

const sig = (v: number) => 1 / (1 + Math.exp(-v));
type V3 = [number, number, number];
const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function angleAt(a: V3, b: V3, c: V3): number {
  const v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const n = Math.hypot(v1[0], v1[1], v1[2]) * Math.hypot(v2[0], v2[1], v2[2]) + 1e-9;
  return (Math.acos(Math.max(-1, Math.min(1, dot / n))) * 180) / Math.PI;
}

// Landmark indices of [MCP, PIP, DIP, TIP] per finger.
const FINGERS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

// Tunable thresholds (validated on real photos, see model_assets/numbers/).
export const T = {
  angleMid: 135,
  angleSoft: 14,
  ratioMid: 1.6,
  ratioSoft: 0.1,
  thumbMid: 0.68,
  thumbSoft: 0.07,
  touchMid: 0.42,
  touchSoft: 0.08,
};

/** width/height convert normalized landmark coordinates into an isotropic space. */
export function extractFeatures(lm: Pt[], width: number, height: number): Features {
  const P: V3[] = lm.map((p) => [p.x * width, p.y * height, p.z * width]);
  const palm = dist(P[0], P[9]) + 1e-6;

  const ext = FINGERS.map(([mcp, pip, dip, tip]) => {
    const straight = (angleAt(P[mcp], P[pip], P[dip]) + angleAt(P[pip], P[dip], P[tip])) / 2;
    const a = sig((straight - T.angleMid) / T.angleSoft);
    const r = dist(P[tip], P[0]) / (dist(P[mcp], P[0]) + 1e-6);
    const b = sig((r - T.ratioMid) / T.ratioSoft);
    return (a + b) / 2;
  }) as Features['ext'];

  const thumbOut = sig((dist(P[4], P[5]) / palm - T.thumbMid) / T.thumbSoft);
  const touch = FINGERS.map(([, , , tip]) =>
    sig((T.touchMid - dist(P[4], P[tip]) / palm) / T.touchSoft),
  ) as Features['touch'];

  return { ext, thumbOut, touch };
}

interface Template {
  ext: [0 | 1, 0 | 1, 0 | 1, 0 | 1];
  thumb: 'out' | 'in' | 'any';
  touch: 0 | 1 | 2 | 3 | null; // which fingertip the thumb must touch
}

const TEMPLATES: Template[] = [
  /* 0 */ { ext: [0, 0, 0, 0], thumb: 'any', touch: 0 }, // O shape: fingertips meet the thumb
  /* 1 */ { ext: [1, 0, 0, 0], thumb: 'in', touch: null },
  /* 2 */ { ext: [1, 1, 0, 0], thumb: 'in', touch: null },
  /* 3 */ { ext: [1, 1, 0, 0], thumb: 'out', touch: null },
  /* 4 */ { ext: [1, 1, 1, 1], thumb: 'in', touch: null },
  /* 5 */ { ext: [1, 1, 1, 1], thumb: 'out', touch: null },
  /* 6 */ { ext: [1, 1, 1, 0], thumb: 'any', touch: 3 }, // thumb meets little finger
  /* 7 */ { ext: [1, 1, 0, 1], thumb: 'any', touch: 2 }, // thumb meets ring finger
  /* 8 */ { ext: [1, 0, 1, 1], thumb: 'any', touch: 1 }, // thumb meets middle finger
  /* 9 */ { ext: [0, 1, 1, 1], thumb: 'any', touch: 0 }, // thumb meets index finger
];

const MIN_MATCH = 0.5;

export function classifyFeatures(f: Features): DigitResult {
  const scores = TEMPLATES.map((t) => {
    let logSum = 0;
    let n = 0;
    const add = (p: number) => {
      logSum += Math.log(Math.max(p, 1e-4));
      n++;
    };
    t.ext.forEach((e, i) => add(e ? f.ext[i] : 1 - f.ext[i]));
    // A thumb resting on the folded ring/little finger counts as "in" even if it is a bit away from the palm.
    const hold = Math.max(f.touch[2], f.touch[3]);
    if (t.thumb === 'out') add(f.thumbOut * (1 - (t.ext[2] === 0 ? hold : 0)));
    else if (t.thumb === 'in') add(t.ext[2] === 0 ? Math.max(1 - f.thumbOut, hold) : 1 - f.thumbOut);
    if (t.touch !== null) add(f.touch[t.touch]);
    return Math.exp(logSum / n); // geometric mean agreement, 0..1
  });

  const powered = scores.map((s) => Math.pow(s, 4));
  const total = powered.reduce((a, b) => a + b, 0) + 1e-9;
  const probs = powered.map((p) => p / total);
  let best = 0;
  for (let i = 1; i < 10; i++) if (scores[i] > scores[best]) best = i;

  const ok = scores[best] >= MIN_MATCH;
  return {
    digit: ok ? best : null,
    confidence: ok ? Math.min(1, scores[best] * 0.5 + probs[best] * 0.5) : 0,
    probs,
    features: f,
  };
}

export function classifyDigit(lm: Pt[], width: number, height: number): DigitResult {
  return classifyFeatures(extractFeatures(lm, width, height));
}
