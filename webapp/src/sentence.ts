import './style.css';
import { getHandLandmarker, detectHands } from './handTracking';
import { clipFeatures, type Pt, type RawFrame } from './sentences/features';
import { getPoseLandmarker, detectPose } from './sentences/pose';
import { loadDtwBank, classifySentence, dtwBankReady, activeFeatureSpec, type SentencePrediction } from './sentences/dtw';
import { loadCatalog, type CatalogEntry } from './sentences/catalog';
import { loadPersonal, savePersonal, clearPersonal, type PersonalSample } from './sentences/personal';
import { currentUser } from './auth';
import { addHistory, updateHistory } from './history';

// Sentence Builder page: records a short clip (per-frame two-hand landmarks + body pose), then
// classifies it by DTW nearest-neighbour against a bank of real training clips, on Stop (see
// dtw.ts's header for why this replaced an earlier trained GRU model -- it measured better on
// the same leave-one-out test). Separate from the alphabet and math pages; only the shared
// MediaPipe hand tracker is reused (plus a pose tracker used only here).

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const video = $<HTMLVideoElement>('webcam');
const overlay = $<HTMLCanvasElement>('overlay');
const octx = overlay.getContext('2d')!;
const statusOverlay = $('status-overlay');
const statusMessage = $('status-message');
const recBadge = $('rec-badge');
const recTimer = $('rec-timer');
const recProgress = $('rec-progress');
const recHint = $('rec-hint');
const btnRecord = $<HTMLButtonElement>('btn-record');
const btnStop = $<HTMLButtonElement>('btn-stop');
const btnRetry = $<HTMLButtonElement>('btn-retry');
const speechBtn = $('sent-speech');
const resultSentence = $('result-sentence');
const resultGloss = $('result-gloss');
const resultFill = $('result-fill');
const resultPct = $('result-pct');
const resultOthers = $('result-others');
const teachBox = $('teach');
const teachCorrect = $<HTMLButtonElement>('teach-correct');
const teachSelect = $<HTMLSelectElement>('teach-select');
const teachSave = $<HTMLButtonElement>('teach-save');
const teachStatus = $('teach-status');
const teachCount = $('teach-count');

function setStatus(msg: string | null) {
  statusOverlay.classList.toggle('hidden', msg === null);
  if (msg !== null) statusMessage.textContent = msg;
}

let speechOn = true;
speechBtn.addEventListener('click', () => {
  speechOn = !speechOn;
  speechBtn.textContent = speechOn ? '🔊 Voice on' : '🔇 Voice off';
  if (!speechOn && 'speechSynthesis' in window) window.speechSynthesis.cancel();
});
function say(text: string) {
  if (!speechOn || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.9;
  window.speechSynthesis.speak(u);
}

// ---------- Hand overlay drawing (up to 2 hands) ----------
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
function drawHands(hands: Pt[][]) {
  const w = overlay.width;
  const h = overlay.height;
  for (const lm of hands) {
    octx.lineWidth = 3;
    octx.strokeStyle = '#3ee0b5';
    octx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      octx.moveTo(lm[a].x * w, lm[a].y * h);
      octx.lineTo(lm[b].x * w, lm[b].y * h);
    }
    octx.stroke();
    octx.fillStyle = '#ff6b3d';
    for (const p of lm) {
      octx.beginPath();
      octx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
      octx.fill();
    }
  }
}

// ---------- Recording state ----------
// DTW references are variable-length, so there's no fixed sequence length to match --
// MAX_QUERY_FRAMES is the same cap the training clips were extracted with (90 frames, ~7s).
const MAX_QUERY_FRAMES = 90;
const SAMPLE_INTERVAL_MS = 80; // ~12.5fps, matching the training data's frame density
const MIN_FRAMES = 10; // ~1s of actual hand-detected frames: shorter clips are unreliable
const MAX_RECORD_MS = 7500;

type Phase = 'idle' | 'recording' | 'processing' | 'result';
let phase: Phase = 'idle';
// Raw per-sample detections (frames where neither a hand nor a body was seen are skipped);
// turned into features only on Stop, because the body reference (shoulder midpoint/width) and
// the idle lead-in/lead-out trim are computed over the whole clip -- exactly like the bank.
let buffer: RawFrame[] = [];
let bodySamples = 0;
let recordStart = 0;
let lastSampleAt = 0;
let handsSeenDuringRecording = 0;
let modelReady = false;

// ---------- Language (ISL / ASL): each has its own reference bank, catalog and clips ----------
type Lang = 'isl' | 'asl';
const LANGS: Record<Lang, { base: string; clips: string; name: string; clipSource: string }> = {
  isl: {
    base: '/sentence_model',
    clips: '/sentence_clips',
    name: 'ISL',
    clipSource: 'Reference clip from the ISL-CSLTR training set, not your camera.',
  },
  asl: {
    base: '/sentence_model_asl',
    clips: '/sentence_clips_asl',
    name: 'ASL',
    clipSource: 'Reference clip: real WLASL word clips joined in ASL word order, not your camera.',
  },
};
const LANG_KEY = 'sentence-lang';
let lang: Lang = 'isl';
try {
  if (localStorage.getItem(LANG_KEY) === 'asl') lang = 'asl';
} catch {
  /* storage blocked: default to ISL */
}
const langBtns: Record<Lang, HTMLButtonElement> = {
  isl: $<HTMLButtonElement>('lang-isl'),
  asl: $<HTMLButtonElement>('lang-asl'),
};

function setPhase(next: Phase) {
  phase = next;
  btnRecord.classList.toggle('hidden', next === 'recording');
  btnStop.classList.toggle('hidden', next !== 'recording');
  btnRetry.classList.toggle('hidden', next === 'idle' || next === 'recording');
  recBadge.classList.toggle('hidden', next !== 'recording');
  recBadge.classList.toggle('flex', next === 'recording');
  btnRecord.disabled = !modelReady;
}

function startRecording() {
  if (!modelReady) return;
  buffer = [];
  bodySamples = 0;
  recordStart = performance.now();
  lastSampleAt = 0;
  handsSeenDuringRecording = 0;
  recProgress.style.width = '0%';
  recHint.textContent = 'Recording… sign your sentence, then press Stop.';
  setPhase('recording');
}

function stopRecording() {
  if (phase !== 'recording') return;
  if (buffer.length < MIN_FRAMES) {
    recHint.textContent = `Too short (${buffer.length} frames captured). Hold a bit longer and try again.`;
    setPhase('idle');
    return;
  }
  const feats = clipFeatures(buffer, video.videoWidth, video.videoHeight, activeFeatureSpec());
  if (feats.length < MIN_FRAMES) {
    recHint.textContent = 'Not enough signing captured. Raise your hands into view while signing and try again.';
    setPhase('idle');
    return;
  }
  const noBody = bodySamples < buffer.length / 2;
  const clipLang = lang;
  setPhase('processing');
  recHint.textContent = 'Reading the sentence…';
  // Give the "processing" state a paint frame before the (synchronous) inference call.
  requestAnimationFrame(() =>
    setTimeout(() => {
      if (clipLang !== lang) return; // language switched mid-recording: result would be for the wrong bank
      const preds = classifySentence(feats, personal);
      lastFeats = feats;
      showResult(preds);
      lastHistoryId = addHistory({
        kind: 'sentence',
        lang,
        text: preds[0].sentence,
        detail: `${Math.round(preds[0].confidence * 100)}% match`,
      });
      showTeach(preds[0].index);
      setPhase('result');
      recHint.textContent = noBody
        ? 'Tip: your shoulders were not visible most of the time -- sit a little further back so your head and shoulders are in frame, it makes matching more accurate.'
        : 'Press "Try again" to record another sentence.';
    }, 0),
  );
}

btnRecord.addEventListener('click', startRecording);
btnStop.addEventListener('click', stopRecording);
btnRetry.addEventListener('click', () => {
  setPhase('idle');
  recHint.textContent = 'Sit back so your head, shoulders and both hands are in view, then press record and sign your sentence at a natural pace.';
});

function showResult(preds: SentencePrediction[]) {
  const top = preds[0];
  resultSentence.textContent = top.sentence;
  resultGloss.textContent = glossFor(top.sentence);
  const pct = Math.round(top.confidence * 100);
  resultFill.style.width = `${pct}%`;
  resultPct.textContent = `${pct}%`;
  resultOthers.innerHTML = preds
    .slice(1, 4)
    .map(
      (p) => `<div class="flex items-center gap-3 text-sm">
        <span class="flex-1 truncate text-ink/70">${p.sentence}</span>
        <div class="confidence-bar-track !h-1.5 w-24"><div class="confidence-bar-fill !bg-ink/30" style="width:${Math.round(p.confidence * 100)}%"></div></div>
        <span class="w-9 text-right text-xs tabular-nums text-ink/50">${Math.round(p.confidence * 100)}%</span>
      </div>`,
    )
    .join('');
  say(top.sentence);
}

let glosses: Record<string, string> = {};
function glossFor(sentence: string): string {
  const g = glosses[sentence];
  return g ? `${LANGS[lang].name} gloss: ${g}` : '';
}

// ---------- Camera + detection loop ----------
let lastVideoTime = -1;
let running = false;

function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }
  octx.clearRect(0, 0, overlay.width, overlay.height);

  const now = performance.now();
  const result = detectHands(video, now);
  const hands = (result?.landmarks ?? []) as Pt[][];
  if (hands.length) {
    drawHands(hands);
    handsSeenDuringRecording++;
  }

  if (phase === 'recording') {
    const elapsed = now - recordStart;
    recProgress.style.width = `${Math.min(100, (elapsed / MAX_RECORD_MS) * 100)}%`;
    recTimer.textContent = `${(elapsed / 1000).toFixed(1)}s`;
    // Sample at the training data's frame rate. The pose tracker only runs on these sample
    // ticks (not every video frame) to keep the page responsive.
    if (now - lastSampleAt >= SAMPLE_INTERVAL_MS && buffer.length < MAX_QUERY_FRAMES) {
      lastSampleAt = now;
      const body = detectPose(video, now)?.landmarks?.[0] ?? null;
      if (hands.length || body) {
        buffer.push({ hands, pose: body as Pt[] | null });
        if (body) bodySamples++;
      }
    }
    if (elapsed >= MAX_RECORD_MS || buffer.length >= MAX_QUERY_FRAMES) {
      stopRecording();
    }
  }
}

function resetResult() {
  teachBox.classList.add('hidden');
  lastFeats = null;
  resultSentence.textContent = 'Record a sentence to begin';
  resultGloss.textContent = '';
  resultFill.style.width = '0%';
  resultPct.textContent = '–';
  resultOthers.innerHTML = '';
}

/** Switches the page to `next`'s reference bank + catalog (each is downloaded once). */
async function setLang(next: Lang) {
  lang = next;
  try {
    localStorage.setItem(LANG_KEY, next);
  } catch {
    /* ignore */
  }
  for (const l of Object.keys(langBtns) as Lang[]) {
    langBtns[l].classList.toggle('active', l === next);
    langBtns[l].setAttribute('aria-selected', String(l === next));
  }
  $('lang-short').textContent = LANGS[next].name;
  $('lang-name').textContent = LANGS[next].name;
  $('lang-note-isl').classList.toggle('hidden', next !== 'isl');
  $('lang-note-asl').classList.toggle('hidden', next !== 'asl');
  $('clip-source').textContent = LANGS[next].clipSource;
  if (phase === 'recording') setPhase('idle');
  modelReady = false;
  setPhase(phase === 'result' ? 'idle' : phase);
  resetResult();
  recHint.textContent = `Loading the ${LANGS[next].name} sentences…`;
  const [catalog] = await Promise.all([loadCatalog(LANGS[next].base), loadDtwBank(LANGS[next].base)]);
  if (lang !== next) return; // user switched again while this was loading
  modelReady = dtwBankReady();
  personal = loadPersonal(next, activeFeatureSpec());
  teachCount.textContent = String(personal.length);
  setCatalog(catalog);
  glosses = Object.fromEntries(catalog.filter((c) => c.gloss).map((c) => [c.sentence, c.gloss as string]));
  $('catalog-count').textContent = String(catalog.length);
  setPhase('idle');
  recHint.textContent = 'Sit back so your head, shoulders and both hands are in view, then press record and sign your sentence at a natural pace.';
}

for (const l of Object.keys(langBtns) as Lang[]) {
  langBtns[l].addEventListener('click', () => {
    if (l !== lang) setLang(l).catch((err) => setStatus(`Could not load the ${LANGS[l].name} sentences (${err instanceof Error ? err.message : err}).`));
  });
}

async function boot() {
  if (!currentUser()) return; // the <head> guard is already redirecting to /login.html
  try {
    setStatus('Loading hand tracker…');
    await Promise.all([getHandLandmarker(2, 0.3), getPoseLandmarker(), setLang(lang)]);

    setStatus('Starting camera…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise<void>((resolve) => (video.onloadedmetadata = () => resolve()));
    await video.play();

    setStatus(null);
    running = true;
    setPhase('idle');
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    const message = err instanceof Error ? err.message : String(err);
    setStatus(
      modelReady
        ? `Camera unavailable: ${message}. Allow camera access and reload.`
        : `Could not load the sentence reference data (${message}). It may not have finished exporting yet — please check back later.`,
    );
  }
}

// ---------- Catalog browser + reference clip dialog ----------
const clipDialog = $<HTMLDialogElement>('clip-dialog');
const clipVideo = $<HTMLVideoElement>('clip-video');
const clipSentence = $('clip-sentence');
const clipGloss = $('clip-gloss');

let openClipIndex: number | null = null;
function openClip(c: CatalogEntry) {
  openClipIndex = c.index;
  clipVideo.src = `${LANGS[lang].clips}/${c.index}.mp4`;
  clipVideo.currentTime = 0;
  clipSentence.textContent = c.sentence;
  clipGloss.textContent = c.gloss ? `${LANGS[lang].name} gloss: ${c.gloss}` : '';
  clipDialog.showModal();
  clipVideo.play().catch(() => {
    /* autoplay may be blocked before the user interacts with <video> controls; that's fine */
  });
}
// Runs regardless of how the dialog closed (button, backdrop click, or Escape).
clipDialog.addEventListener('close', () => {
  clipVideo.pause();
  clipVideo.removeAttribute('src');
  clipVideo.load();
});
$('clip-close').addEventListener('click', () => clipDialog.close());
clipDialog.addEventListener('click', (e) => {
  if (e.target === clipDialog) clipDialog.close();
});
$('clip-practice').addEventListener('click', () => {
  practiceTarget = openClipIndex;
  clipDialog.close();
  document.getElementById('btn-record')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

let catalogRows: CatalogEntry[] = [];
const catalogSearch = $<HTMLInputElement>('catalog-search');
catalogSearch.addEventListener('input', renderCatalog);

function setCatalog(catalog: CatalogEntry[]) {
  catalogRows = catalog;
  teachSelect.innerHTML = catalog
    .map((c) => `<option value="${c.index}">${c.sentence}</option>`)
    .join('');
  renderCatalog();
}

function renderCatalog() {
  const list = $('catalog-list');
  const q = catalogSearch.value.trim().toLowerCase();
  const rows = catalogRows.filter((c) => !q || c.sentence.toLowerCase().includes(q));
  list.innerHTML = '';
  for (const c of rows) {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.className =
      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left odd:bg-sand hover:bg-brand-soft';
    btn.innerHTML = `
      <span class="grid size-7 shrink-0 place-items-center rounded-full bg-ink text-paper">
        <svg viewBox="0 0 24 24" class="size-3.5 translate-x-px" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
      </span>
      <span class="min-w-0 flex-1">
        <span class="block text-ink">${c.sentence}</span>
        ${c.gloss ? `<span class="text-xs text-ink/40">${c.gloss}</span>` : ''}
      </span>`;
    btn.addEventListener('click', () => openClip(c));
    li.appendChild(btn);
    list.appendChild(li);
  }
}

// ---------- Teach it your signing (personal references, see sentences/personal.ts) ----------
let personal: PersonalSample[] = [];
let lastFeats: Float32Array[] | null = null; // the last classified recording, until saved/reset
let practiceTarget: number | null = null; // sentence chosen via "Practise this sentence"
let lastHistoryId: string | null = null; // history entry of the result on screen

function showTeach(topIndex: number) {
  teachBox.classList.remove('hidden');
  teachSelect.value = String(practiceTarget ?? topIndex);
  teachCorrect.disabled = teachSave.disabled = false;
  teachStatus.textContent = practiceTarget !== null ? 'You were practising the sentence selected below.' : '';
}

function teach(classIndex: number) {
  if (!lastFeats) return;
  const ok = savePersonal(lang, activeFeatureSpec(), classIndex, lastFeats);
  if (!ok) {
    teachStatus.textContent = 'Could not save (browser storage is full or blocked).';
    return;
  }
  personal = loadPersonal(lang, activeFeatureSpec());
  teachCount.textContent = String(personal.length);
  const sentence = catalogRows.find((c) => c.index === classIndex)?.sentence ?? '';
  if (lastHistoryId) {
    const corrected = sentence !== resultSentence.textContent;
    updateHistory(lastHistoryId, corrected ? { text: sentence, detail: 'corrected by you' } : { detail: 'confirmed by you' });
  }
  teachStatus.textContent = `Saved as "${sentence}". Recording it 2–3 times helps even more.`;
  teachCorrect.disabled = teachSave.disabled = true;
  lastFeats = null;
  practiceTarget = null;
}

teachCorrect.addEventListener('click', () => {
  const top = resultSentence.textContent;
  const entry = catalogRows.find((c) => c.sentence === top);
  if (entry) teach(entry.index);
});
teachSave.addEventListener('click', () => teach(Number(teachSelect.value)));
$('teach-clear').addEventListener('click', () => {
  if (!dtwBankReady()) return;
  if (!confirm(`Delete your ${personal.length} saved ${LANGS[lang].name} recordings from this browser?`)) return;
  clearPersonal(lang, activeFeatureSpec());
  personal = [];
  teachCount.textContent = '0';
});

boot();
