// Optional personal calibration for digit recognition.
//
// The user can record a few examples of each digit with their own hand and camera. Those
// samples (landmark vectors only, no images) live in this browser's localStorage and are blended
// with the rule-based classifier via a small nearest-neighbour vote. Nothing is uploaded.

import type { Pt } from './digits';

const KEY = 'hastavani.digitCalibration.v1';
const MAX_PER_DIGIT = 80;
const MIN_TOTAL_TO_USE = 30;
const K = 7;

interface Sample {
  d: number;
  v: number[];
}

let samples: Sample[] = load();

function load(): Sample[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Sample[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(samples));
  } catch {
    /* storage unavailable or full: calibration just won't persist */
  }
}

/** Wrist-centred, hand-size-normalised 63-vector (same idea as the alphabet features). */
export function normVector(lm: Pt[], width: number, height: number): number[] {
  const pts = lm.map((p) => [p.x * width, p.y * height, p.z * width]);
  const [wx, wy, wz] = pts[0];
  let scale = 1e-6;
  for (const p of pts) {
    p[0] -= wx;
    p[1] -= wy;
    p[2] -= wz;
    scale = Math.max(scale, Math.hypot(p[0], p[1]));
  }
  const out: number[] = [];
  for (const p of pts) out.push(p[0] / scale, p[1] / scale, p[2] / scale);
  return out;
}

export function counts(): number[] {
  const c = new Array(10).fill(0);
  for (const s of samples) c[s.d]++;
  return c;
}

export function addSample(digit: number, v: number[]) {
  samples.push({ d: digit, v: v.map((x) => Math.round(x * 1000) / 1000) });
  const mine = samples.filter((s) => s.d === digit);
  if (mine.length > MAX_PER_DIGIT) {
    const drop = mine.length - MAX_PER_DIGIT;
    let removed = 0;
    samples = samples.filter((s) => (s.d === digit && removed++ < drop ? false : true));
  }
  persist();
}

export function clearDigit(digit: number) {
  samples = samples.filter((s) => s.d !== digit);
  persist();
}

export function clearAll() {
  samples = [];
  persist();
}

export function isActive(): boolean {
  return samples.length >= MIN_TOTAL_TO_USE;
}

/** k-nearest-neighbour class probabilities over digits, or null when there is too little data. */
export function knnProbs(v: number[]): { probs: number[]; nearest: number } | null {
  if (!isActive()) return null;
  const dists = samples.map((s) => {
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      const d = v[i] - s.v[i];
      sum += d * d;
    }
    return { d: s.d, dist: Math.sqrt(sum) };
  });
  dists.sort((a, b) => a.dist - b.dist);
  const probs = new Array(10).fill(0);
  let total = 0;
  for (const n of dists.slice(0, K)) {
    const w = 1 / (n.dist + 0.05);
    probs[n.d] += w;
    total += w;
  }
  return { probs: probs.map((p) => p / total), nearest: dists[0].dist };
}
