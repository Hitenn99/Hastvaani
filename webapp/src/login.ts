import './style.css';
import { currentUser, safeNext, signIn } from './auth';

// Sign-in page: /login.html?next=/sentence.html sends you back where you were going.

const next = safeNext(new URLSearchParams(location.search).get('next'));
if (currentUser()) location.replace(next);

const form = document.getElementById('login-form') as HTMLFormElement;
const username = document.getElementById('username') as HTMLInputElement;
const password = document.getElementById('password') as HTMLInputElement;
const remember = document.getElementById('remember') as HTMLInputElement;
const errorBox = document.getElementById('login-error')!;
const submit = document.getElementById('login-submit') as HTMLButtonElement;
const togglePw = document.getElementById('toggle-pw') as HTMLButtonElement;

function showError(msg: string | null) {
  errorBox.classList.toggle('hidden', msg === null);
  errorBox.textContent = msg ?? '';
}

togglePw.addEventListener('click', () => {
  const show = password.type === 'password';
  password.type = show ? 'text' : 'password';
  togglePw.textContent = show ? 'Hide' : 'Show';
  togglePw.setAttribute('aria-pressed', String(show));
  togglePw.setAttribute('aria-label', show ? 'Hide password' : 'Show password');
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  showError(null);
  if (!username.value.trim() || !password.value) {
    showError('Enter your username and password.');
    (username.value.trim() ? password : username).focus();
    return;
  }
  submit.disabled = true;
  submit.textContent = 'Signing in…';
  try {
    const session = await signIn(username.value, password.value, remember.checked);
    if (session) {
      location.replace(next);
      return;
    }
    showError('Incorrect username or password.');
    password.value = '';
    password.focus();
  } catch {
    showError('Sign-in is not available in this browser (it needs a secure https:// or localhost page).');
  }
  submit.disabled = false;
  submit.textContent = 'Sign in';
});

username.focus();
