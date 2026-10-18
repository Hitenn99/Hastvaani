import type { ModeConfig, SignMode } from './config';

export interface Prediction {
  letter: string;
  confidence: number;
  probabilities: Float32Array;
}

// ---------- ASL: MLP on MediaPipe hand landmarks (plain JS, no TF.js) ----------
// Trained by model_assets/train_asl_landmarks.py; feature normalization here
// MUST match `normalize()` in that script (wrist at origin, scaled by hand size).
interface DenseLayer {
  W: number[][]; // (out, in)
  b: number[];
  act: 'relu' | 'softmax';
}

const mlps: Partial<Record<SignMode, { layers: DenseLayer[]; labels: string[] }>> = {};

export async function loadForMode(mode: SignMode, config: ModeConfig): Promise<void> {
  if (mlps[mode]) return;
  const url = mode === 'asl' ? '/asl_landmarks/model.json' : '/isl_landmarks/model.json';
  const [m, labels] = await Promise.all([
    fetch(url).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch ${mode.toUpperCase()} landmark model (${r.status})`);
      return r.json() as Promise<{ layers: DenseLayer[] }>;
    }),
    fetch(config.labelsUrl).then((r) => r.json() as Promise<string[]>),
  ]);
  mlps[mode] = { layers: m.layers, labels };
}

export function landmarkModelReady(mode: SignMode): boolean {
  return mlps[mode] !== undefined;
}

type Landmark = { x: number; y: number; z: number };

function runMlp(mode: SignMode, input: Float32Array): Prediction {
  const mlp = mlps[mode]!;
  let h = input;
  for (const layer of mlp.layers) {
    const out = new Float32Array(layer.b.length);
    for (let o = 0; o < out.length; o++) {
      const row = layer.W[o];
      let s = layer.b[o];
      for (let i = 0; i < row.length; i++) s += row[i] * h[i];
      out[o] = layer.act === 'relu' ? Math.max(0, s) : s;
    }
    if (layer.act === 'softmax') {
      let max = -Infinity;
      for (const v of out) if (v > max) max = v;
      let sum = 0;
      for (let i = 0; i < out.length; i++) {
        out[i] = Math.exp(out[i] - max);
        sum += out[i];
      }
      for (let i = 0; i < out.length; i++) out[i] /= sum;
    }
    h = out;
  }
  let maxIdx = 0;
  for (let i = 1; i < h.length; i++) if (h[i] > h[maxIdx]) maxIdx = i;
  return { letter: mlp.labels[maxIdx] ?? '?', confidence: h[maxIdx], probabilities: h };
}

/**
 * ISL: up to two hands. Feature layout MUST match `normalize()` in
 * model_assets/train_isl_landmarks.py: hands sorted by wrist x, all present
 * points centered on their joint centroid and divided by the max xy distance
 * from it, then [hand0 (63), hand1 (63), present0, present1] = 128 floats
 * (an absent hand is all zeros).
 */
export function predictIslLandmarks(hands: Landmark[][], width: number, height: number): Prediction {
  const sorted = [...hands].sort((a, b) => a[0].x - b[0].x).slice(0, 2);
  const pts = sorted.map((hand) => hand.map((p) => [p.x * width, p.y * height, p.z * width]));
  let cx = 0, cy = 0, cz = 0, count = 0;
  for (const hand of pts) for (const p of hand) { cx += p[0]; cy += p[1]; cz += p[2]; count++; }
  cx /= count; cy /= count; cz /= count;
  let scale = 1e-6;
  for (const hand of pts) for (const p of hand) {
    p[0] -= cx; p[1] -= cy; p[2] -= cz;
    scale = Math.max(scale, Math.hypot(p[0], p[1]));
  }
  const input = new Float32Array(128);
  pts.forEach((hand, hi) => {
    hand.forEach((p, i) => {
      input[hi * 63 + i * 3] = p[0] / scale;
      input[hi * 63 + i * 3 + 1] = p[1] / scale;
      input[hi * 63 + i * 3 + 2] = p[2] / scale;
    });
    input[126 + hi] = 1;
  });
  return runMlp('isl', input);
}

export function predictLandmarks(landmarks: Landmark[], width: number, height: number): Prediction {
  const n = landmarks.length;
  // Pixel-space coordinates (isotropic); z is on roughly the same scale as x.
  const pts = landmarks.map((p) => [p.x * width, p.y * height, p.z * width]);
  const [wx, wy, wz] = pts[0];
  let scale = 1e-6;
  for (const p of pts) {
    p[0] -= wx;
    p[1] -= wy;
    p[2] -= wz;
    scale = Math.max(scale, Math.hypot(p[0], p[1]));
  }
  const input = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    input[i * 3] = pts[i][0] / scale;
    input[i * 3 + 1] = pts[i][1] / scale;
    input[i * 3 + 2] = pts[i][2] / scale;
  }
  return runMlp('asl', input);
}
