// Sign-in for the site. There is NO backend: the account list lives here and the session is kept
// in the browser (localStorage with "Remember me", otherwise sessionStorage, so it ends when the
// browser closes). Passwords are stored only as salted SHA-256 hashes, but anyone can read this
// file, so this keeps casual visitors out of the app pages -- it is not real security. A real
// deployment needs a server that checks the password and issues the session.
//
// Every page except /login.html is also guarded by an inline <head> script injected by
// vite.config.ts, which redirects to /login.html before the page renders.

export const SESSION_KEY = 'hv-session';

interface Account {
  username: string;
  displayName: string;
  /** hex SHA-256 of `hastavani:<lowercased username>:<password>` */
  passwordHash: string;
}

const ACCOUNTS: Account[] = [
  // Public demo account: demo / demo1234 -- replace with your own before deploying.
  { username: 'demo', displayName: 'Demo User', passwordHash: '8fb3f044a03f74efd5d1e19cb8ebd2e8ad87fed7bf258ba867008cb06be14268' },
];

export interface Session {
  username: string;
  displayName: string;
  signedInAt: number;
}

async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function readSession(store: Storage): Session | null {
  try {
    const raw = store.getItem(SESSION_KEY);
    const s = raw ? (JSON.parse(raw) as Session) : null;
    return s && ACCOUNTS.some((a) => a.username === s.username) ? s : null;
  } catch {
    return null;
  }
}

/** The signed-in user, or null. */
export function currentUser(): Session | null {
  return readSession(localStorage) ?? readSession(sessionStorage);
}

/** Checks the credentials; on success stores the session and returns it, otherwise null. */
export async function signIn(username: string, password: string, remember: boolean): Promise<Session | null> {
  const name = username.trim().toLowerCase();
  const account = ACCOUNTS.find((a) => a.username === name);
  const hash = await sha256Hex(`hastavani:${name}:${password}`);
  if (!account || hash !== account.passwordHash) return null;
  const session: Session = { username: account.username, displayName: account.displayName, signedInAt: Date.now() };
  signOut(); // never leave a session in both stores
  try {
    (remember ? localStorage : sessionStorage).setItem(SESSION_KEY, JSON.stringify(session));
  } catch {
    /* storage blocked: the sign-in only lasts for this page */
  }
  return session;
}

export function signOut(): void {
  for (const store of [localStorage, sessionStorage]) {
    try {
      store.removeItem(SESSION_KEY);
    } catch {
      /* ignore */
    }
  }
}

/** Only same-site paths are allowed as a post-login destination (no open redirects). */
export function safeNext(next: string | null): string {
  return next && next.startsWith('/') && !next.startsWith('//') && !next.startsWith('/login') ? next : '/index.html';
}
