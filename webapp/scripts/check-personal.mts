// Checks that personal (user-taught) samples take part in matching: a real ISL clip (so NOT in
// the ASL bank it is matched against) is taught
// under a DIFFERENT, deliberately wrong sentence (as a time-stretched + jittered copy, like a
// second take); querying the clip again must then return the taught sentence first.
// Run from the repo root (after check_pose_parity.py wrote tmp/pose_parity.json):
//   node --experimental-strip-types webapp/scripts/check-personal.mts
import { readFileSync } from 'node:fs';
import { clipFeatures, type RawFrame, type FeatureSpec } from '../src/sentences/features.ts';
import { loadDtwBank, classifySentence } from '../src/sentences/dtw.ts';

globalThis.fetch = (async (url: string) => {
  const buf = readFileSync('webapp/public' + url);
  return { ok: true, json: async () => JSON.parse(buf.toString('utf8')), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
}) as unknown as typeof fetch;

interface Clip { base: string; spec: FeatureSpec; W: number; H: number; frames: RawFrame[] }
const clips: Clip[] = JSON.parse(readFileSync('tmp/pose_parity.json', 'utf8')).filter((c: Clip) => c.base === '/sentence_model_asl');
const bank = await loadDtwBank('/sentence_model_asl');
let pass = 0;
for (const c of clips.slice(0, 7)) {
  const q = clipFeatures(c.frames, c.W, c.H, c.spec);
  const before = classifySentence(q)[0];
  const wrong = (before.index + 37) % bank.labels.length;
  // "second take": resample to 85% length + small noise
  const m = Math.max(4, Math.round(q.length * 0.85));
  const take = Array.from({ length: m }, (_, i) => q[Math.round((i * (q.length - 1)) / (m - 1))].map((v) => v + (Math.random() - 0.5) * 0.05));
  const after = classifySentence(q, [{ classIndex: wrong, frames: take }])[0];
  const ok = after.index === wrong;
  pass += Number(ok);
  console.log(`${ok ? 'ok ' : 'BAD'} before: ${before.sentence} -> taught as "${bank.labels[wrong]}" -> now: ${after.sentence}`);
}
console.log(pass === 7 ? 'PERSONAL OK' : 'PERSONAL FAILED');
