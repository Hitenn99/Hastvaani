import './style.css';
import { ASL, ISL, type LetterInfo } from './alphabet';
import { currentUser, signOut } from './auth';

// Shared behavior for every page: mobile menu, scroll reveal, Learn page, copy button.
// (The recognizer itself lives in main.ts and is untouched by any of this.)

// ---------- Theme toggle (initial theme is set by an inline script in <head>) ----------
const themeBtn = document.getElementById('theme-btn');
function syncThemeBtn() {
  const dark = document.documentElement.dataset.theme === 'dark';
  themeBtn?.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
}
syncThemeBtn();
themeBtn?.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('theme', next);
  } catch {
    /* storage unavailable: theme just won't persist */
  }
  syncThemeBtn();
});

// ---------- Account (header): "Sign in", or the signed-in user's menu ----------
const authSlot = document.getElementById('auth-slot');
const mobileAuth = document.getElementById('mobile-auth');
const user = currentUser();
const here = encodeURIComponent(location.pathname + location.search);
if (authSlot && mobileAuth) {
  if (!user) {
    if (!location.pathname.startsWith('/login')) {
      authSlot.innerHTML = `<a href="/login.html?next=${here}" class="btn-ghost !px-4">Sign in</a>`;
      mobileAuth.innerHTML = `<a href="/login.html?next=${here}" class="mobile-link">Sign in</a>`;
    }
  } else {
    const initial = user.displayName.charAt(0).toUpperCase();
    authSlot.innerHTML = `
      <button id="user-btn" class="flex items-center gap-2 rounded-full border border-ink/15 py-1 pl-1 pr-3 transition hover:bg-ink/5" aria-haspopup="menu" aria-expanded="false" aria-controls="user-menu">
        <span class="grid size-8 place-items-center rounded-full bg-brand font-display text-sm font-bold text-white">${initial}</span>
        <span class="hidden text-sm font-semibold sm:inline">${user.displayName}</span>
      </button>
      <div id="user-menu" class="absolute right-0 top-12 hidden w-56 overflow-hidden rounded-2xl border border-ink/10 bg-surface py-2 shadow-xl" role="menu">
        <p class="px-4 pb-2 pt-1 text-xs text-ink/50">Signed in as <strong class="text-ink">${user.displayName}</strong></p>
        <a href="/history.html" class="block px-4 py-2.5 text-sm hover:bg-ink/5" role="menuitem">Your history</a>
        <button id="sign-out" class="block w-full px-4 py-2.5 text-left text-sm text-accent hover:bg-ink/5" role="menuitem">Sign out</button>
      </div>`;
    mobileAuth.innerHTML = `
      <p class="px-3 pb-1 pt-2 text-xs text-ink/50">Signed in as <strong class="text-ink">${user.displayName}</strong></p>
      <a href="/history.html" class="mobile-link">Your history</a>
      <button id="sign-out-mobile" class="mobile-link w-full text-left text-accent">Sign out</button>`;
    const userBtn = document.getElementById('user-btn')!;
    const userMenu = document.getElementById('user-menu')!;
    const setOpen = (open: boolean) => {
      userMenu.classList.toggle('hidden', !open);
      userBtn.setAttribute('aria-expanded', String(open));
    };
    userBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      setOpen(userMenu.classList.contains('hidden'));
    });
    document.addEventListener('click', (e) => {
      if (!userMenu.contains(e.target as Node)) setOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') setOpen(false);
    });
    const doSignOut = () => {
      signOut();
      location.href = '/login.html';
    };
    document.getElementById('sign-out')!.addEventListener('click', doSignOut);
    document.getElementById('sign-out-mobile')!.addEventListener('click', doSignOut);
  }
}

// Home page greeting
const hello = document.getElementById('hello-user');
if (hello && user) hello.textContent = `Welcome back, ${user.displayName} · ASL & ISL`;

// ---------- Mobile menu ----------
const menuBtn = document.getElementById('menu-btn');
const mobileMenu = document.getElementById('mobile-menu');
menuBtn?.addEventListener('click', () => {
  const open = mobileMenu?.classList.toggle('hidden') === false;
  menuBtn.setAttribute('aria-expanded', String(open));
});

// ---------- Scroll reveal ----------
const revealEls = document.querySelectorAll<HTMLElement>('.reveal');
if ('IntersectionObserver' in window) {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12 },
  );
  revealEls.forEach((el) => io.observe(el));
} else {
  revealEls.forEach((el) => el.classList.add('in'));
}

// ---------- Copy button (translator page) ----------
const copyBtn = document.getElementById('btn-copy');
copyBtn?.addEventListener('click', async () => {
  const text = (document.getElementById('word-output')?.textContent ?? '').trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = 'Copied!';
  } catch {
    copyBtn.textContent = 'Copy failed';
  }
  setTimeout(() => (copyBtn.textContent = 'Copy text'), 1500);
});

// ---------- Learn page ----------
const grid = document.getElementById('alphabet-grid');
if (grid) {
  const tabAsl = document.getElementById('tab-asl') as HTMLButtonElement;
  const tabIsl = document.getElementById('tab-isl') as HTMLButtonElement;
  const search = document.getElementById('letter-search') as HTMLInputElement;
  const note = document.getElementById('lang-note')!;
  const empty = document.getElementById('no-results')!;
  const dialog = document.getElementById('letter-dialog') as HTMLDialogElement;
  const dLetter = document.getElementById('dialog-letter')!;
  const dImg = document.getElementById('dialog-img') as HTMLImageElement;
  const dHands = document.getElementById('dialog-hands')!;
  const dTip = document.getElementById('dialog-tip')!;
  const dClose = document.getElementById('dialog-close')!;

  let lang: 'asl' | 'isl' = 'asl';

  const handsLabel = (l: LetterInfo) =>
    l.hands === 'either' ? 'One or two hands' : l.hands === 2 ? 'Two hands' : 'One hand';

  function render() {
    const q = search.value.trim().toUpperCase();
    const data = (lang === 'asl' ? ASL : ISL).filter((l) => !q || l.letter === q);
    grid!.innerHTML = '';
    for (const l of data) {
      const b = document.createElement('button');
      const src = `/signs/${lang}/${l.letter}.jpg`;
      b.className =
        'group overflow-hidden rounded-3xl border border-ink/10 bg-surface text-left shadow-sm transition hover:-translate-y-1 hover:border-brand hover:shadow-lg';
      b.setAttribute('aria-label', `${lang.toUpperCase()} letter ${l.letter}`);
      b.innerHTML = `
        <span class="relative block aspect-square overflow-hidden bg-ink/5">
          <img src="${src}" alt="${lang.toUpperCase()} sign for the letter ${l.letter}" loading="lazy"
               class="size-full object-cover transition duration-500 group-hover:scale-105" />
          <span class="absolute left-3 top-3 grid size-9 place-items-center rounded-xl bg-surface font-display text-xl font-extrabold shadow">${l.letter}</span>
        </span>
        <span class="flex items-center justify-between px-4 py-3 text-xs font-semibold text-ink/60">
          <span>${handsLabel(l)}</span>${l.motion ? '<span class="rounded-full bg-accent-soft px-2 py-0.5 text-accent">motion</span>' : ''}
        </span>`;
      b.addEventListener('click', () => {
        dLetter.textContent = l.letter;
        dImg.src = src;
        dImg.alt = `${lang.toUpperCase()} sign for the letter ${l.letter}`;
        dHands.textContent = handsLabel(l) + (l.motion ? ' · involves motion' : '');
        dTip.textContent = l.tip;
        dialog.showModal();
      });
      grid!.appendChild(b);
    }
    empty.classList.toggle('hidden', data.length > 0);
    note.textContent =
      lang === 'asl'
        ? 'American Sign Language fingerspelling uses one hand. J and Z are traced in the air, so a still camera reads them less reliably.'
        : 'Indian Sign Language fingerspelling mostly uses two hands, so keep both fully in view. Hand counts here come from the photos the ISL model was trained on.';
  }

  function setLang(next: 'asl' | 'isl') {
    lang = next;
    tabAsl.setAttribute('aria-selected', String(next === 'asl'));
    tabIsl.setAttribute('aria-selected', String(next === 'isl'));
    render();
  }

  tabAsl.addEventListener('click', () => setLang('asl'));
  tabIsl.addEventListener('click', () => setLang('isl'));
  search.addEventListener('input', render);
  dClose.addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });
  render();
}
