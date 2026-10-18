// Sentence classifier: two stacked Bidirectional-GRU layers + a small MLP head, run as
// plain TypeScript matrix math (no TF.js), reading exported weights from
// /sentence_model/model.json. The forward-pass formulas here were verified against Keras
// output numerically (max diff ~3e-8) before being written -- see
// model_assets/sentences/export_sentence_model.py, which every retrain must re-run so this
// stays in sync. In particular:
//   - GRU gate order is [update z, reset r, candidate h], "reset_after" style: the reset
//     gate is applied AFTER the recurrent-kernel matmul, and input/recurrent biases are
//     kept separate (Keras's default GRU, matching cuDNN).
//   - Masked (padded) timesteps use an EXPLICIT real-frame-count, not "is this row all
//     zero" -- Keras's own masking works the same way, and inferring it from data would
//     break the second GRU layer (its input is the first layer's hidden state, which is
//     not all-zero at padded positions even though Keras's own sequence *output* is).

export interface GruWeights {
  units: number;
  kernel: number[][]; // (units*3, inputDim) -- ROW-MAJOR PER OUTPUT UNIT, see export script
  recurrentKernel: number[][]; // (units*3, units)
  inputBias: number[]; // units*3
  recurrentBias: number[]; // units*3
}

export interface BiGruLayer {
  forward: GruWeights;
  backward: GruWeights;
  returnSequences: boolean;
}

export interface DenseLayer {
  W: number[][]; // (out, in)
  b: number[];
  act: 'relu' | 'softmax';
}

export interface SentenceModel {
  maxLen: number;
  featureLen: number;
  gru: BiGruLayer[];
  dense: DenseLayer[];
  labels: string[];
}

let model: SentenceModel | null = null;

export async function loadSentenceModel(): Promise<SentenceModel> {
  if (model) return model;
  const [m, labels] = await Promise.all([
    fetch('/sentence_model/model.json').then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch sentence model (${r.status})`);
      return r.json() as Promise<Omit<SentenceModel, 'labels'>>;
    }),
    fetch('/sentence_model/labels.json').then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch sentence labels (${r.status})`);
      return r.json() as Promise<string[]>;
    }),
  ]);
  model = { ...m, labels };
  return model;
}

export function sentenceModelReady(): boolean {
  return model !== null;
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * One GRU direction over a (T, inputDim) sequence, with an explicit per-timestep mask
 * (matching Keras's Masking layer, computed once from the raw input and reused for every
 * layer -- see the header comment). `mask[t] === false` means: don't update the hidden
 * state at t, and that timestep's sequence output is exactly 0. This is NOT just a
 * trailing-padding thing -- roughly 85% of the real recorded clips this model is trained
 * on have at least one frame in the MIDDLE of the sentence where MediaPipe briefly lost
 * the hand(s) (fast motion, hand leaving frame, blur), so masking must be checked at every
 * timestep independently, not derived from a single "first n are real" cutoff.
 */
function gruForward(x: Float32Array[], mask: boolean[], w: GruWeights, backwards: boolean): { seq: Float32Array[]; last: Float32Array } {
  const { units, kernel, recurrentKernel, inputBias, recurrentBias } = w;
  const T = x.length;
  const order: number[] = [];
  if (backwards) for (let t = T - 1; t >= 0; t--) order.push(t);
  else for (let t = 0; t < T; t++) order.push(t);

  let h = new Float32Array(units);
  const seq: Float32Array[] = Array.from({ length: T }, () => new Float32Array(units));

  for (const t of order) {
    if (!mask[t]) continue; // masked: seq[t] stays all-zero, h unchanged
    const xt = x[t];
    const mx = new Float32Array(units * 3);
    const mh = new Float32Array(units * 3);
    for (let o = 0; o < units * 3; o++) {
      let sx = inputBias[o];
      const krow = kernel[o];
      for (let i = 0; i < xt.length; i++) sx += krow[i] * xt[i];
      mx[o] = sx;
      let sh = recurrentBias[o];
      const rrow = recurrentKernel[o];
      for (let i = 0; i < units; i++) sh += rrow[i] * h[i];
      mh[o] = sh;
    }
    const hNext = new Float32Array(units);
    for (let u = 0; u < units; u++) {
      const z = sigmoid(mx[u] + mh[u]);
      const r = sigmoid(mx[units + u] + mh[units + u]);
      const hcand = Math.tanh(mx[units * 2 + u] + r * mh[units * 2 + u]);
      hNext[u] = (1 - z) * hcand + z * h[u];
    }
    h = hNext;
    seq[t] = h;
  }
  return { seq, last: h };
}

function biGru(x: Float32Array[], mask: boolean[], layer: BiGruLayer): Float32Array[] {
  const fwd = gruForward(x, mask, layer.forward, false);
  const bwd = gruForward(x, mask, layer.backward, true);
  if (layer.returnSequences) {
    return fwd.seq.map((f, t) => Float32Array.of(...f, ...bwd.seq[t]));
  }
  return [Float32Array.of(...fwd.last, ...bwd.last)];
}

function dense(x: Float32Array, layer: DenseLayer): Float32Array {
  const out = new Float32Array(layer.b.length);
  for (let o = 0; o < out.length; o++) {
    let s = layer.b[o];
    const row = layer.W[o];
    for (let i = 0; i < x.length; i++) s += row[i] * x[i];
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
  return out;
}

export interface SentencePrediction {
  sentence: string;
  index: number;
  confidence: number;
}

/** `frames`: real per-frame feature vectors (length = actual recorded frame count, NOT
 * pre-padded). Padding to the model's fixed input length happens here. A frame is "masked"
 * whenever its feature vector is exactly all-zero -- true for the padding this function
 * adds, and also true for any real recorded frame where no hand was detected that instant
 * (frameFeatures() returns an all-zero vector then too), matching how the training data
 * and Keras's Masking layer both define it. */
export function predictSentence(frames: Float32Array[]): SentencePrediction[] {
  if (!model) throw new Error('Sentence model not loaded yet');
  const n = Math.min(frames.length, model.maxLen);
  const padded: Float32Array[] = Array.from({ length: model.maxLen }, (_, t) =>
    t < n ? frames[t] : new Float32Array(model!.featureLen),
  );
  const mask = padded.map((row) => row.some((v) => v !== 0));

  let x = padded;
  for (const layer of model.gru) {
    x = biGru(x, mask, layer);
  }

  let h = x[0];
  for (const layer of model.dense) h = dense(h, layer);

  const order = Array.from(h, (_, i) => i).sort((a, b) => h[b] - h[a]);
  return order.map((i) => ({ sentence: model!.labels[i], index: i, confidence: h[i] }));
}
