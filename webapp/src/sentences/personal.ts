// "Teach it your signing": the user's own recordings, saved in this browser and matched
// alongside the reference bank.
//
// Why: the reference banks come from a handful of signers, and different people sign the same
// sentence quite differently -- on the ISL data, two signers recorded in the same room on the
// same phone match each other's takes only ~7% of the time (see README "Update 14"). A
// recording by the same person is by far the closest reference there is, so every confirmed
// or corrected result is stored (as the exact feature sequence the matcher compares) and used
// as an extra reference for that sentence.
//
// Stored per language AND feature recipe (a bank rebuilt with different features makes old
// samples incomparable, so they're simply not loaded). localStorage, newest MAX_SAMPLES kept.

import type { FeatureSpec } from './features';

export interface PersonalSample {
  classIndex: number;
  frames: Float32Array[];
}

interface StoredSample {
  c: number; // class index
  n: number; // feature length per frame
  d: string; // base64 of the frames' float32 bytes, back to back
}

const MAX_SAMPLES = 60;

function storageKey(lang: string, spec: FeatureSpec): string {
  return `sentence-personal-${lang}-${JSON.stringify(spec)}`;
}

function toBase64(frames: Float32Array[]): string {
  const len = frames.length ? frames[0].length : 0;
  const flat = new Float32Array(frames.length * len);
  frames.forEach((f, i) => flat.set(f, i * len));
  const bytes = new Uint8Array(flat.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

function fromBase64(s: StoredSample): Float32Array[] {
  const bin = atob(s.d);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const flat = new Float32Array(bytes.buffer);
  const out: Float32Array[] = [];
  for (let i = 0; i + s.n <= flat.length; i += s.n) out.push(flat.slice(i, i + s.n));
  return out;
}

function readStored(lang: string, spec: FeatureSpec): StoredSample[] {
  try {
    const raw = localStorage.getItem(storageKey(lang, spec));
    return raw ? (JSON.parse(raw) as StoredSample[]) : [];
  } catch {
    return [];
  }
}

export function loadPersonal(lang: string, spec: FeatureSpec): PersonalSample[] {
  return readStored(lang, spec).map((s) => ({ classIndex: s.c, frames: fromBase64(s) }));
}

/** Saves one recording as a reference for `classIndex`. Returns false if storage failed. */
export function savePersonal(lang: string, spec: FeatureSpec, classIndex: number, frames: Float32Array[]): boolean {
  if (!frames.length) return false;
  const all = readStored(lang, spec);
  all.push({ c: classIndex, n: frames[0].length, d: toBase64(frames) });
  // drop the oldest until it fits (both the count cap and the browser's storage quota)
  while (all.length) {
    const kept = all.slice(-MAX_SAMPLES);
    try {
      localStorage.setItem(storageKey(lang, spec), JSON.stringify(kept));
      return true;
    } catch {
      all.shift();
    }
  }
  return false;
}

export function clearPersonal(lang: string, spec: FeatureSpec): void {
  try {
    localStorage.removeItem(storageKey(lang, spec));
  } catch {
    /* ignore */
  }
}
