// TS side of model_assets/sentences/check_pose_parity.py: runs the shipped clipFeatures() +
// DTW on the dumped raw clips and compares with the Python reference.
// Run from the repo root: node --experimental-strip-types webapp/scripts/check-pose-parity.mts
import { readFileSync } from 'node:fs';
import { clipFeatures, type RawFrame, type FeatureSpec } from '../src/sentences/features.ts';
import { loadDtwBank, _distanceToEntryForTest, classifySentence } from '../src/sentences/dtw.ts';

globalThis.fetch = (async (url: string) => {
  const buf = readFileSync('webapp/public' + url);
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
}) as unknown as typeof fetch;

interface Clip { base: string; spec: FeatureSpec; W: number; H: number; frames: RawFrame[]; features: number[][]; entries: number[]; dists: number[] }
const clips: Clip[] = JSON.parse(readFileSync('tmp/pose_parity.json', 'utf8'));

let maxFeat = 0, maxDistRel = 0, lenMismatch = 0;
for (const c of clips) {
  await loadDtwBank(c.base);
  const f = clipFeatures(c.frames, c.W, c.H, c.spec);
  if (f.length !== c.features.length) { lenMismatch++; console.log('length mismatch', f.length, c.features.length); continue; }
  f.forEach((row, t) => row.forEach((v, k) => (maxFeat = Math.max(maxFeat, Math.abs(v - c.features[t][k])))));
  c.entries.forEach((e, i) => {
    const d = _distanceToEntryForTest(f, e);
    maxDistRel = Math.max(maxDistRel, Math.abs(d - c.dists[i]) / Math.max(c.dists[i], 1e-9));
  });
  const top = classifySentence(f)[0];
  console.log(`${c.base} v${c.spec.version}: clip ${c.features.length} frames -> top: ${top.sentence} (${(top.confidence * 100).toFixed(0)}%)`);
}
console.log({ clips: clips.length, lenMismatch, maxFeatureAbsDiff: maxFeat, maxDistRelDiff: maxDistRel });
if (lenMismatch || maxFeat > 1e-4 || maxDistRel > 1e-5) { console.error('PARITY FAILED'); process.exit(1); }
console.log('PARITY OK');
