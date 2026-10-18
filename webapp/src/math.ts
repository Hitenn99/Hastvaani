import './style.css';
import { getHandLandmarker, detectHands } from './handTracking';
import { classifyFeatures, extractFeatures, type Pt } from './numbers/digits';
import * as calib from './numbers/calibration';
import { currentUser } from './auth';
import { addHistory } from './history';
import { GUIDE, GUIDE_BY_LANG, GUIDE_NOTES, SHARED_NOTE, fingerDiagram, type GuideCard } from './numbers/guide';

// Math-with-signs page. Completely separate from the alphabet recogniser (main.ts / model.ts):
// it only reuses the shared MediaPipe hand tracker.

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// ---------- DOM ----------
const video = $<HTMLVideoElement>('webcam');
const overlay = $<HTMLCanvasElement>('overlay');
const octx = overlay.getContext('2d')!;
const statusOverlay = $('status-overlay');
const statusMessage = $('status-message');
const digitDisplay = $('digit-display');
const ringFg = document.getElementById('ring-fg') as unknown as SVGCircleElement;
const digitHint = $('digit-hint');

// ---------- Small helpers ----------
const RING = 276.5;
let speechOn = true;

function setStatus(msg: string | null) {
  statusOverlay.classList.toggle('hidden', msg === null);
  if (msg !== null) statusMessage.textContent = msg;
}

function say(text: string) {
  if (!speechOn || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
}

const rand = (a: number, b: number) => a + Math.floor(Math.random() * (b - a + 1));

// ---------- Modes / tabs ----------
type Mode = 'calc' | 'practice' | 'guide';
let mode: Mode = 'calc';
const tabs: Record<Mode, HTMLElement> = { calc: $('tab-calc'), practice: $('tab-practice'), guide: $('tab-guide') };
const views: Record<Mode, HTMLElement> = { calc: $('view-calc'), practice: $('view-practice'), guide: $('view-guide') };

function setMode(next: Mode) {
  mode = next;
  (Object.keys(tabs) as Mode[]).forEach((m) => {
    tabs[m].setAttribute('aria-selected', String(m === next));
    views[m].classList.toggle('hidden', m !== next);
  });
  resetStabilizer();
}
(Object.keys(tabs) as Mode[]).forEach((m) => tabs[m].addEventListener('click', () => setMode(m)));

// ---------- Language toggle (guide text only; the digit handshapes are the same in both charts) ----------
type Lang = 'asl' | 'isl';
let lang: Lang = 'asl';
const langBtns: Record<Lang, HTMLElement[]> = {
  asl: [$('num-lang-asl'), $('guide-lang-asl')],
  isl: [$('num-lang-isl'), $('guide-lang-isl')],
};
function setLang(next: Lang) {
  lang = next;
  (Object.keys(langBtns) as Lang[]).forEach((l) =>
    langBtns[l].forEach((b) => b.setAttribute('aria-pressed', String(l === next))),
  );
  renderGuide();
}
(Object.keys(langBtns) as Lang[]).forEach((l) =>
  langBtns[l].forEach((b) => b.addEventListener('click', () => setLang(l))),
);

const speechBtn = $('num-speech');
speechBtn.addEventListener('click', () => {
  speechOn = !speechOn;
  speechBtn.textContent = speechOn ? '🔊 Voice on' : '🔇 Voice off';
  if (!speechOn && 'speechSynthesis' in window) window.speechSynthesis.cancel();
});

// ---------- Digit recognition + stabiliser ----------
function recognise(lm: Pt[], w: number, h: number): { digit: number | null; conf: number } {
  const rule = classifyFeatures(extractFeatures(lm, w, h));
  const knn = calib.knnProbs(calib.normVector(lm, w, h));
  if (!knn) return { digit: rule.digit, conf: rule.confidence };
  // Personal samples only count when the hand is actually close to something the user recorded.
  const weight = 0.6 * Math.max(0, Math.min(1, 1 - (knn.nearest - 0.2) / 0.8));
  const probs = rule.probs.map((p, i) => (1 - weight) * p + weight * knn.probs[i]);
  let best = 0;
  for (let i = 1; i < 10; i++) if (probs[i] > probs[best]) best = i;
  const ok = probs[best] > 0.4 && (rule.digit !== null || weight > 0.2);
  return { digit: ok ? best : null, conf: probs[best] };
}

const HIST = 14;
const NEED = 10;
const hist: (number | null)[] = [];
let lastCommitted: number | null = null;
let absent = 0;

function resetStabilizer() {
  hist.length = 0;
  showProgress(null, 0);
}

function showProgress(leader: number | null, progress: number) {
  digitDisplay.textContent = leader === null ? '–' : String(leader);
  ringFg.style.strokeDashoffset = String(RING * (1 - Math.min(1, progress)));
}

function pushFrame(digit: number | null) {
  hist.push(digit);
  if (hist.length > HIST) hist.shift();
  const counts = new Array(10).fill(0);
  for (const d of hist) if (d !== null) counts[d]++;
  let leader = 0;
  for (let i = 1; i < 10; i++) if (counts[i] > counts[leader]) leader = i;
  const n = counts[leader];
  if (n < 3) {
    showProgress(null, 0);
    return;
  }
  showProgress(leader, n / NEED);
  if (n >= NEED && !recording) {
    if (leader !== lastCommitted) {
      lastCommitted = leader;
      commitDigit(leader);
      hist.length = 0;
      pulseRing();
    }
  }
}

function pulseRing() {
  digitDisplay.animate([{ transform: 'scale(1.35)' }, { transform: 'scale(1)' }], { duration: 260, easing: 'ease-out' });
}

function commitDigit(d: number) {
  if (mode === 'calc') calcDigit(d);
  else if (mode === 'practice') practiceDigit(d);
}

// ---------- Camera loop ----------
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function drawHand(lm: Pt[]) {
  const w = overlay.width;
  const h = overlay.height;
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

let lastVideoTime = -1;
let running = false;
let frameCount = 0;

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

  const result = detectHands(video, performance.now());
  const lm = result?.landmarks?.[0] as Pt[] | undefined;
  if (!lm) {
    absent++;
    if (absent >= 6) lastCommitted = null; // hand lowered: the same digit may be entered again
    pushFrame(null);
    digitHint.textContent = 'No hand detected. Show one hand clearly to the camera.';
    return;
  }
  absent = 0;
  drawHand(lm);

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (recording) {
    recordFrame(lm, w, h);
    return;
  }
  const { digit, conf } = recognise(lm, w, h);
  pushFrame(digit);
  digitHint.textContent =
    digit === null
      ? 'Not sure yet. Adjust your fingers so the sign is clear.'
      : `Looks like ${digit} (${Math.round(conf * 100)}%). Hold steady until the ring fills.`;
}

async function startCamera(): Promise<void> {
  try {
    setStatus('Loading hand tracker…');
    await getHandLandmarker(1);
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
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus(
      'Camera or hand tracker unavailable. Allow camera access and reload. You can still type digits on your keyboard.',
    );
  }
}

// ---------- Calculator ----------
type Op = '+' | '−' | '×' | '÷' | '%';
const OP_WORDS: Record<Op, string> = { '+': 'plus', '−': 'minus', '×': 'times', '÷': 'divided by', '%': 'mod' };
const MAX_DIGITS = 6;

const calcExpr = $('calc-expr');
const calcMain = $('calc-main');
const calcMsg = $('calc-msg');
const calcHistoryEl = $('calc-history');

let acc: number | null = null;
let op: Op | null = null;
let cur = '';
let justEvaluated = false;
const history: string[] = [];

const fmt = (n: number) => String(Math.round(n * 10000) / 10000);

function apply(a: number, o: Op, b: number): number | null {
  switch (o) {
    case '+': return a + b;
    case '−': return a - b;
    case '×': return a * b;
    case '÷': return b === 0 ? null : a / b;
    case '%': return b === 0 ? null : a % b;
  }
}

function calcMessage(msg: string) {
  calcMsg.textContent = msg;
  if (msg) setTimeout(() => calcMsg.textContent === msg && (calcMsg.textContent = ''), 2200);
}

function renderCalc() {
  calcExpr.textContent = acc !== null && op ? `${fmt(acc)} ${op}` : ' ';
  calcMain.textContent = cur !== '' ? cur : acc !== null ? fmt(acc) : '0';
  calcHistoryEl.innerHTML = history.length
    ? history.map((h) => `<li>${h}</li>`).join('')
    : '<li class="text-ink/40">Your calculations will appear here.</li>';
}

function calcDigit(d: number) {
  if (justEvaluated && op === null) {
    acc = null;
    justEvaluated = false;
  }
  if (cur.length >= MAX_DIGITS) return calcMessage(`Max ${MAX_DIGITS} digits`);
  cur = cur === '0' ? String(d) : cur + d;
  renderCalc();
}

function calcOperator(next: Op) {
  if (cur !== '') {
    if (acc !== null && op) {
      const r = apply(acc, op, Number(cur));
      if (r === null) return calcMessage("Can't divide by 0");
      acc = r;
    } else {
      acc = Number(cur);
    }
    cur = '';
  } else if (acc === null) {
    acc = 0;
  }
  op = next;
  justEvaluated = false;
  renderCalc();
}

function calcEquals() {
  if (acc === null || !op || cur === '') return calcMessage('Enter a number after the operator');
  const b = Number(cur);
  const r = apply(acc, op, b);
  if (r === null) return calcMessage("Can't divide by 0");
  history.unshift(`${fmt(acc)} ${op} ${fmt(b)} = <strong class="text-ink">${fmt(r)}</strong>`);
  if (history.length > 8) history.pop();
  say(`${fmt(acc)} ${OP_WORDS[op]} ${fmt(b)} equals ${fmt(r)}`);
  addHistory({ kind: 'math', lang, text: `${fmt(acc)} ${op} ${fmt(b)} = ${fmt(r)}`, detail: 'Calculator' });
  acc = r;
  op = null;
  cur = '';
  justEvaluated = true;
  renderCalc();
}

document.querySelectorAll<HTMLButtonElement>('.op-btn').forEach((b) =>
  b.addEventListener('click', () => calcOperator(b.dataset.op as Op)),
);
$('btn-eq').addEventListener('click', calcEquals);
$('btn-c').addEventListener('click', () => {
  acc = null;
  op = null;
  cur = '';
  justEvaluated = false;
  renderCalc();
});
$('btn-back').addEventListener('click', () => {
  cur = cur.slice(0, -1);
  renderCalc();
});
$('btn-clear-history').addEventListener('click', () => {
  history.length = 0;
  renderCalc();
});

// ---------- Practice ----------
type Level = 'easy' | 'medium' | 'hard';
interface Question {
  text: string;
  answer: number;
}
const LEVEL_NOTES: Record<Level, string> = {
  easy: 'Addition and subtraction with single-digit answers.',
  medium: 'Adds multiplication, so answers can have two digits.',
  hard: 'Adds division and remainder (%), with bigger numbers.',
};

function makeQuestion(level: Level): Question {
  const ops: Op[] = level === 'easy' ? ['+', '−'] : level === 'medium' ? ['+', '−', '×'] : ['+', '−', '×', '÷', '%'];
  const o = ops[rand(0, ops.length - 1)];
  let a = 0;
  let b = 0;
  switch (o) {
    case '+':
      if (level === 'hard') { a = rand(10, 49); b = rand(10, 49); }
      else if (level === 'easy') { a = rand(0, 9); b = rand(0, 9 - a); }
      else { a = rand(0, 9); b = rand(0, 9); }
      break;
    case '−':
      a = level === 'hard' ? rand(20, 99) : rand(0, 9);
      b = rand(0, level === 'hard' ? 19 : a);
      break;
    case '×':
      a = level === 'hard' ? rand(2, 9) : rand(0, 9);
      b = level === 'hard' ? rand(2, 9) : rand(0, 9);
      break;
    case '÷': {
      b = rand(1, 9);
      const q = rand(0, 9);
      a = b * q;
      break;
    }
    case '%':
      a = rand(10, 99);
      b = rand(2, 9);
      break;
  }
  const answer = apply(a, o, b) as number;
  return { text: `${a} ${o} ${b}`, answer };
}

const ROUND = 10;
const pScore = $('p-score');
const pStreak = $('p-streak');
const pBest = $('p-best');
const pProgress = $('p-progress');
const pCount = $('p-count');
const pQuestion = $('p-question');
const pSlots = $('p-slots');
const pFeedback = $('p-feedback');
const pStart = $('p-start');
const pHint = $('p-hint');
const pClear = $('p-clear');
const pSkip = $('p-skip');
const pHintCard = $('p-hint-card');
const pSummary = $('p-summary');
const pStage = $('p-stage');
const pConfetti = $('p-confetti');
const levelBtns = Array.from(document.querySelectorAll<HTMLButtonElement>('.level-btn'));

let level: Level = 'easy';
let phase: 'idle' | 'asking' | 'feedback' | 'done' = 'idle';
let question: Question | null = null;
let entered = '';
let qIndex = 0;
let score = 0;
let streak = 0;
let bestStreak = 0;
let correctCount = 0;
let lastText = '';

const bestKey = (l: Level) => `hastavani.mathBest.${l}`;
function loadBest(): number {
  try {
    return Number(localStorage.getItem(bestKey(level)) ?? 0) || 0;
  } catch {
    return 0;
  }
}
function saveBest(v: number) {
  try {
    localStorage.setItem(bestKey(level), String(v));
  } catch {
    /* ignore */
  }
}

function renderStats() {
  pScore.textContent = String(score);
  pStreak.textContent = String(streak);
  pBest.textContent = String(loadBest());
  pProgress.style.width = `${(qIndex / ROUND) * 100}%`;
}

function renderSlots() {
  const len = question ? String(question.answer).length : 1;
  pSlots.innerHTML = '';
  for (let i = 0; i < len; i++) {
    const ch = entered[i] ?? '';
    const box = document.createElement('div');
    box.className = `grid size-16 place-items-center rounded-2xl border-2 font-display text-4xl font-extrabold ${
      ch ? 'border-brand bg-brand-soft text-brand-dark' : 'border-dashed border-ink/25 text-ink/25'
    }`;
    box.textContent = ch || '?';
    pSlots.appendChild(box);
  }
}

function setLevel(l: Level) {
  level = l;
  levelBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.level === l)));
  $('p-level-note').textContent = LEVEL_NOTES[l];
  resetRound();
}
levelBtns.forEach((b) => b.addEventListener('click', () => setLevel(b.dataset.level as Level)));

function resetRound() {
  phase = 'idle';
  question = null;
  entered = '';
  qIndex = 0;
  score = 0;
  streak = 0;
  bestStreak = 0;
  correctCount = 0;
  pQuestion.textContent = '? + ?';
  pCount.textContent = `${ROUND} questions. Ready?`;
  pFeedback.textContent = '';
  pFeedback.className = 'mt-5 h-7 text-lg font-bold';
  pSlots.innerHTML = '';
  pSummary.classList.add('hidden');
  pSummary.classList.remove('flex');
  pHintCard.classList.add('hidden');
  pStart.classList.remove('hidden');
  [pHint, pClear, pSkip].forEach((b) => b.classList.add('hidden'));
  renderStats();
}

function nextQuestion() {
  if (qIndex >= ROUND) return finishRound();
  let q = makeQuestion(level);
  for (let i = 0; i < 6 && q.text === lastText; i++) q = makeQuestion(level);
  lastText = q.text;
  question = q;
  entered = '';
  phase = 'asking';
  pQuestion.innerHTML = `${q.text} <span class="text-ink/40">=</span> <span class="text-brand">?</span>`;
  pCount.textContent = `Question ${qIndex + 1} of ${ROUND}`;
  pFeedback.textContent = 'Sign your answer, one digit at a time.';
  pFeedback.className = 'mt-5 h-7 text-lg font-semibold text-ink/60';
  pHintCard.classList.add('hidden');
  pStart.classList.add('hidden');
  [pHint, pClear, pSkip].forEach((b) => b.classList.remove('hidden'));
  renderSlots();
  renderStats();
}

function startRound() {
  resetRound();
  say('Let us begin');
  nextQuestion();
}

function cardHtml(c: GuideCard, compact: boolean) {
  const visual = c.img
    ? `<div class="rounded-2xl bg-white p-2"><img src="${c.img}" alt="${lang.toUpperCase()} sign for ${c.digit}" loading="lazy" class="mx-auto ${compact ? 'h-32' : 'h-40'} w-auto object-contain" /></div>`
    : fingerDiagram(GUIDE[c.digit]);
  return `${visual}<p class="mt-2 text-xs leading-snug text-ink/70">${c.text}</p>${
    c.note ? `<p class="mt-1 text-[11px] font-medium leading-snug text-accent">${c.note}</p>` : ''
  }`;
}

function showHint() {
  if (!question) return;
  const digits = Array.from(new Set(String(question.answer).split('').map(Number)));
  pHintCard.innerHTML = digits
    .map((d) => {
      const c = GUIDE_BY_LANG[lang].find((x) => x.digit === d) as GuideCard;
      return `<div class="mb-3 last:mb-0"><p class="mb-1 text-xs font-bold uppercase tracking-widest text-ink/50">${lang.toUpperCase()} digit ${d}</p>${cardHtml(c, true)}</div>`;
    })
    .join('');
  pHintCard.classList.remove('hidden');
}

function confetti() {
  const colors = ['#ff6b3d', '#2fb79c', '#f5c542', '#4f8cff', '#e8558a'];
  for (let i = 0; i < 32; i++) {
    const s = document.createElement('span');
    s.className = 'confetti';
    s.style.left = `${Math.random() * 100}%`;
    s.style.background = colors[i % colors.length];
    s.style.animationDelay = `${Math.random() * 0.25}s`;
    s.style.setProperty('--dx', `${(Math.random() - 0.5) * 160}px`);
    pConfetti.appendChild(s);
    setTimeout(() => s.remove(), 1600);
  }
}

function answerQuestion() {
  if (!question || phase !== 'asking') return;
  phase = 'feedback';
  const ok = Number(entered) === question.answer;
  qIndex++;
  if (ok) {
    correctCount++;
    score += 10 + Math.min(streak, 5) * 2;
    streak++;
    bestStreak = Math.max(bestStreak, streak);
    pFeedback.textContent = streak >= 3 ? `Correct! ${streak} in a row 🔥` : 'Correct! ✓';
    pFeedback.className = 'mt-5 h-7 text-lg font-bold text-brand';
    pStage.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.02)' }, { transform: 'scale(1)' }], { duration: 300 });
    confetti();
    say('Correct');
  } else {
    streak = 0;
    pFeedback.textContent = `Not quite. The answer is ${question.answer}.`;
    pFeedback.className = 'mt-5 h-7 text-lg font-bold text-accent';
    pSlots.animate(
      [{ transform: 'translateX(0)' }, { transform: 'translateX(-10px)' }, { transform: 'translateX(10px)' }, { transform: 'translateX(-6px)' }, { transform: 'translateX(0)' }],
      { duration: 350 },
    );
    showHint();
    say(`The answer is ${question.answer}`);
  }
  if (score > loadBest()) saveBest(score);
  renderStats();
  setTimeout(nextQuestion, ok ? 1400 : 3600);
}

function practiceDigit(d: number) {
  if (phase !== 'asking' || !question) return;
  const len = String(question.answer).length;
  if (entered.length >= len) return;
  entered += d;
  renderSlots();
  if (entered.length === len) setTimeout(answerQuestion, 350);
}

function finishRound() {
  phase = 'done';
  [pHint, pClear, pSkip].forEach((b) => b.classList.add('hidden'));
  const pct = Math.round((correctCount / ROUND) * 100);
  const msg = pct === 100 ? 'Perfect round! 🌟' : pct >= 70 ? 'Great work!' : pct >= 40 ? 'Getting there. Keep practising!' : 'Good start. Try the digit guide, then go again.';
  pSummary.innerHTML = `
    <p class="text-xs font-semibold uppercase tracking-widest text-ink/50">Round complete</p>
    <p class="mt-2 font-display text-6xl font-extrabold text-brand">${score}</p>
    <p class="text-sm text-ink/60">points</p>
    <p class="mt-4 text-lg font-bold">${msg}</p>
    <p class="mt-1 text-sm text-ink/60">${correctCount} of ${ROUND} correct · best streak ${bestStreak}</p>
    <div class="mt-6 flex gap-2"><button id="p-again" class="btn-accent">Play again</button></div>`;
  pSummary.classList.remove('hidden');
  pSummary.classList.add('flex');
  $('p-again').addEventListener('click', startRound);
  if (pct >= 70) confetti();
  say(`Round complete. You scored ${score} points.`);
  addHistory({
    kind: 'math',
    lang,
    text: `Practice round: ${score} points`,
    detail: `${correctCount} of ${ROUND} correct · ${level}`,
  });
}

pStart.addEventListener('click', startRound);
pHint.addEventListener('click', showHint);
pClear.addEventListener('click', () => {
  if (phase !== 'asking') return;
  entered = '';
  renderSlots();
});
pSkip.addEventListener('click', () => {
  if (phase !== 'asking') return;
  streak = 0;
  qIndex++;
  nextQuestion();
});

// Keyboard fallback: number keys act like signed digits, so the page is usable without a camera.
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input, textarea, select')) return;
  if (/^[0-9]$/.test(e.key)) commitDigit(Number(e.key));
  else if (mode === 'calc') {
    const map: Record<string, Op> = { '+': '+', '-': '−', '*': '×', '/': '÷', '%': '%' };
    if (map[e.key]) calcOperator(map[e.key]);
    else if (e.key === 'Enter' || e.key === '=') calcEquals();
    else if (e.key === 'Backspace') { cur = cur.slice(0, -1); renderCalc(); }
  }
});

// ---------- Digit guide + calibration ----------
const guideGrid = $('guide-grid');
const guideNote = $('guide-note');

function renderGuide() {
  guideNote.innerHTML = `${GUIDE_NOTES[lang]} <span class="text-ink/50">${SHARED_NOTE}</span>`;
  guideGrid.innerHTML = GUIDE_BY_LANG[lang]
    .map(
      (c) => `<div class="card !p-4 text-center ${c.referenceOnly ? 'opacity-70' : ''}">
        <p class="font-display text-4xl font-extrabold ${c.referenceOnly ? 'text-ink/40' : 'text-brand'}">${c.digit}</p>
        <div class="mt-1">${cardHtml(c, false)}</div>
      </div>`,
    )
    .join('');
}

const calChips = $('cal-chips');
const calTarget = $('cal-target');
const calStatus = $('cal-status');
const calRecord = $('cal-record') as HTMLButtonElement;
let calDigit = 0;
let recording = false;
let recorded = 0;
const REC_TARGET = 40;

function renderCalChips() {
  const c = calib.counts();
  calChips.innerHTML = '';
  for (let d = 0; d < 10; d++) {
    const b = document.createElement('button');
    b.className = `rounded-full border px-3 py-1.5 text-sm font-semibold transition ${
      d === calDigit ? 'border-brand bg-brand-soft text-brand-dark' : 'border-ink/15 hover:border-ink'
    }`;
    b.innerHTML = `${d} <span class="text-xs font-medium text-ink/50">${c[d]}</span>`;
    b.addEventListener('click', () => {
      calDigit = d;
      calTarget.textContent = String(d);
      renderCalChips();
    });
    calChips.appendChild(b);
  }
  calStatus.textContent = calib.isActive()
    ? 'Personal samples are active.'
    : `Record at least ${30} samples in total to activate.`;
}

calRecord.addEventListener('click', () => {
  if (!running) return (calStatus.textContent = 'Camera is not running.');
  recording = true;
  recorded = 0;
  frameCount = 0;
  calRecord.disabled = true;
  calStatus.textContent = `Hold digit ${calDigit} and move your hand slightly… 0/${REC_TARGET}`;
});

function recordFrame(lm: Pt[], w: number, h: number) {
  frameCount++;
  if (frameCount % 2) return;
  calib.addSample(calDigit, calib.normVector(lm, w, h));
  recorded++;
  calStatus.textContent = `Recording digit ${calDigit}… ${recorded}/${REC_TARGET}`;
  if (recorded >= REC_TARGET) {
    recording = false;
    calRecord.disabled = false;
    renderCalChips();
    calStatus.textContent = `Saved ${REC_TARGET} samples for digit ${calDigit}. ${calStatus.textContent}`;
  }
}

$('cal-reset').addEventListener('click', () => {
  calib.clearAll();
  renderCalChips();
});

// ---------- Boot ----------
renderCalc();
renderGuide();
renderCalChips();
setLevel('easy');
setMode('calc');
if (currentUser()) startCamera(); // signed out: the <head> guard is redirecting to /login.html
