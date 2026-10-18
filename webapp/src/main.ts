import './style.css';
import { MODE_CONFIGS, CONFIDENCE_THRESHOLD, type SignMode } from './config';
import { getHandLandmarker, detectHands, computeUnionBBox } from './handTracking';
import { loadForMode, predictLandmarks, predictIslLandmarks } from './model';
import { PredictionSmoother } from './smoothing';
import { SpeechAnnouncer } from './speech';
import { currentUser } from './auth';
import { addHistory } from './history';

// ---------- DOM references ----------
const video = document.getElementById('webcam') as HTMLVideoElement;
const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const overlayCtx = overlay.getContext('2d')!;
const statusOverlay = document.getElementById('status-overlay')!;
const statusMessage = document.getElementById('status-message')!;
const letterDisplay = document.getElementById('letter-display')!;
const confidenceFill = document.getElementById('confidence-fill') as HTMLDivElement;
const confidenceText = document.getElementById('confidence-text')!;
const muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;
const wordOutput = document.getElementById('word-output')!;
const modelInfo = document.getElementById('model-info')!;
const btnSpace = document.getElementById('btn-space') as HTMLButtonElement;
const btnBackspace = document.getElementById('btn-backspace') as HTMLButtonElement;
const btnClear = document.getElementById('btn-clear') as HTMLButtonElement;
const btnSpeakWord = document.getElementById('btn-speak-word') as HTMLButtonElement;
const modeAslBtn = document.getElementById('mode-asl') as HTMLButtonElement;
const modeIslBtn = document.getElementById('mode-isl') as HTMLButtonElement;

// ---------- State ----------
let currentMode: SignMode = 'asl';
const smoother = new PredictionSmoother();
const speech = new SpeechAnnouncer();
let wordBuffer = '';
let running = false;
let modelReady = false;
let lastVideoTime = -1;
let noHandStatusShown = false;
let inferenceErrorShown = false;

function setStatus(message: string | null) {
  if (message === null) {
    statusOverlay.classList.add('hidden');
    statusMessage.textContent = '';
  } else {
    statusOverlay.classList.remove('hidden');
    statusMessage.textContent = message;
  }
}

function updateWordDisplay() {
  wordOutput.textContent = wordBuffer.length > 0 ? wordBuffer : ' ';
}

function updateLetterDisplay(letter: string | null, confidence: number) {
  letterDisplay.textContent = letter ?? '–';
  const pct = Math.round(confidence * 100);
  confidenceFill.style.width = `${pct}%`;
  confidenceText.textContent = `${pct}%`;
}

// ---------- Word builder buttons ----------
btnSpace.addEventListener('click', () => {
  wordBuffer += ' ';
  updateWordDisplay();
});
btnBackspace.addEventListener('click', () => {
  wordBuffer = wordBuffer.slice(0, -1);
  updateWordDisplay();
});
// History: the built-up text is logged when it's "done" -- cleared, spoken, or the page is left --
// and never twice in a row.
let lastLogged = '';
function logWords() {
  const text = wordBuffer.replace(/\s+/g, ' ').trim();
  if (!text || text === lastLogged) return;
  lastLogged = text;
  addHistory({ kind: 'translate', lang: currentMode, text, detail: 'Fingerspelled' });
}
window.addEventListener('pagehide', logWords);

btnClear.addEventListener('click', () => {
  logWords();
  wordBuffer = '';
  updateWordDisplay();
});
btnSpeakWord.addEventListener('click', () => {
  logWords();
  speech.speakText(wordBuffer);
});

muteBtn.addEventListener('click', () => {
  const nowMuted = !speech.isMuted();
  speech.setMuted(nowMuted);
  muteBtn.textContent = nowMuted ? '🔇 Speech off' : '🔊 Speech on';
});

// ---------- Mode switching ----------
async function switchMode(mode: SignMode) {
  if (mode === currentMode) return;
  currentMode = mode;
  modeAslBtn.classList.toggle('active', mode === 'asl');
  modeAslBtn.setAttribute('aria-selected', String(mode === 'asl'));
  modeIslBtn.classList.toggle('active', mode === 'isl');
  modeIslBtn.setAttribute('aria-selected', String(mode === 'isl'));

  smoother.reset();
  updateLetterDisplay(null, 0);
  modelReady = false;

  const config = MODE_CONFIGS[mode];
  modelInfo.textContent = `Loading ${config.label} model…`;
  setStatus(`Loading ${config.label} model…`);

  try {
    await Promise.all([loadForMode(mode, config), getHandLandmarker(config.numHands)]);
    modelReady = true;
    modelInfo.innerHTML = `<strong>${config.label}</strong> model loaded. Detecting up to <strong>${config.numHands}</strong> hand${config.numHands > 1 ? 's' : ''}. Hold a steady, clear sign in view.`;
    setStatus(null);
  } catch (err) {
    console.error(err);
    modelInfo.textContent = `Failed to load the ${config.label} model.`;
    setStatus(
      `Could not load the ${config.label} model. It may not have finished training/converting yet — please check back later.`,
    );
  }
}

modeAslBtn.addEventListener('click', () => switchMode('asl'));
modeIslBtn.addEventListener('click', () => switchMode('isl'));

// ---------- Camera setup ----------
async function startCamera(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise<void>((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
    await video.play();
    return true;
  } catch (err) {
    console.error('Camera error:', err);
    setStatus(
      'Camera permission denied or unavailable. Please allow camera access in your browser settings and reload the page.',
    );
    return false;
  }
}

// ---------- Main detection loop (requestAnimationFrame, never blocks the UI thread) ----------
function resizeOverlay() {
  overlay.width = video.videoWidth || video.clientWidth;
  overlay.height = video.videoHeight || video.clientHeight;
}

function loop() {
  if (!running) return;
  requestAnimationFrame(loop);

  if (!modelReady) return;
  if (video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    resizeOverlay();
  }

  const config = MODE_CONFIGS[currentMode];
  const timestampMs = performance.now();
  const result = detectHands(video, timestampMs);

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (!result || !result.landmarks || result.landmarks.length === 0) {
    if (!noHandStatusShown) {
      setStatus('No hand detected. Show your hand clearly to the camera.');
      noHandStatusShown = true;
    }
    const { changed } = smoother.push(null);
    if (changed) updateLetterDisplay(null, 0);
    return;
  }

  if (noHandStatusShown) {
    setStatus(null);
    noHandStatusShown = false;
  }

  const bbox = computeUnionBBox(result, video.videoWidth, video.videoHeight, config.bboxPadding);
  if (!bbox || bbox.width <= 0 || bbox.height <= 0) {
    return;
  }

  // Draw the bounding box overlay.
  overlayCtx.strokeStyle = '#33d69f';
  overlayCtx.lineWidth = 3;
  overlayCtx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

  try {
    const prediction =
      currentMode === 'asl'
        ? predictLandmarks(result.landmarks[0], video.videoWidth, video.videoHeight)
        : predictIslLandmarks(result.landmarks, video.videoWidth, video.videoHeight);

    if (prediction.confidence >= CONFIDENCE_THRESHOLD) {
      const { locked, changed } = smoother.push(prediction.letter);
      updateLetterDisplay(locked, prediction.confidence);
      if (changed && locked) {
        wordBuffer += locked;
        updateWordDisplay();
        speech.announce(locked);
      }
    } else {
      const { locked } = smoother.push(null);
      updateLetterDisplay(locked, prediction.confidence);
    }
    if (inferenceErrorShown) {
      inferenceErrorShown = false;
      setStatus(null);
    }
  } catch (err) {
    // Surface inference errors on-screen instead of failing silently every
    // frame -- without this, a thrown error here (e.g. a backend or tensor
    // shape issue) would leave the letter/confidence frozen at their initial
    // placeholder values forever, with no visible indication anything is
    // wrong, since requestAnimationFrame(loop) is already scheduled before
    // this code runs and happily keeps calling loop() again next frame.
    console.error('Inference error:', err);
    if (!inferenceErrorShown) {
      inferenceErrorShown = true;
      const message = err instanceof Error ? err.message : String(err);
      setStatus(`Prediction error: ${message} (see browser console for details)`);
    }
  }
}

// Surface any error we didn't explicitly catch (e.g. inside MediaPipe's WASM
// glue, or a model-loading failure after the initial load) instead of it
// only ever showing up in a devtools console the user may not have open.
window.addEventListener('error', (e) => {
  console.error('Uncaught error:', e.error ?? e.message);
  setStatus(`Unexpected error: ${e.message} (see browser console for details)`);
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('Unhandled promise rejection:', e.reason);
  const message = e.reason instanceof Error ? e.reason.message : String(e.reason);
  setStatus(`Unexpected error: ${message} (see browser console for details)`);
});

// ---------- Boot ----------
async function init() {
  if (!currentUser()) return; // the <head> guard is already redirecting to /login.html
  const [cameraOk] = await Promise.all([
    startCamera(),
    (async () => {
      const config = MODE_CONFIGS[currentMode];
      modelInfo.textContent = `Loading ${config.label} model…`;
      try {
        await Promise.all([loadForMode(currentMode, config), getHandLandmarker(config.numHands)]);
        modelReady = true;
        modelInfo.innerHTML = `<strong>${config.label}</strong> model loaded. Detecting up to <strong>${config.numHands}</strong> hand${config.numHands > 1 ? 's' : ''}. Hold a steady, clear sign in view.`;
      } catch (err) {
        console.error(err);
        modelInfo.textContent = `Failed to load the ${config.label} model.`;
        setStatus(`Could not load the ${config.label} model.`);
      }
    })(),
  ]);

  if (cameraOk) {
    resizeOverlay();
    running = true;
    if (modelReady) setStatus(null);
    requestAnimationFrame(loop);
  }
}

updateWordDisplay();
init();
