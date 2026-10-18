// Activity history for the signed-in user: what the translator, sentence builder and math page
// recognised. Stored in this browser (localStorage), one list per user, newest first.

import { currentUser } from './auth';

export type HistoryKind = 'translate' | 'sentence' | 'math';

export interface HistoryItem {
  id: string;
  at: number; // epoch ms
  kind: HistoryKind;
  lang?: 'asl' | 'isl';
  text: string; // the main thing that was recognised / computed
  detail?: string; // e.g. confidence, score
}

const MAX_ITEMS = 500;

function key(): string | null {
  const u = currentUser();
  return u ? `hv-history-${u.username}` : null;
}

export function loadHistory(): HistoryItem[] {
  const k = key();
  if (!k) return [];
  try {
    const raw = localStorage.getItem(k);
    return raw ? (JSON.parse(raw) as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

function save(items: HistoryItem[]) {
  const k = key();
  if (!k) return;
  try {
    localStorage.setItem(k, JSON.stringify(items.slice(0, MAX_ITEMS)));
  } catch {
    /* storage full/blocked: history just isn't kept */
  }
}

/** Records one entry for the signed-in user (no-op when signed out). Returns its id. */
export function addHistory(item: Omit<HistoryItem, 'id' | 'at'>): string | null {
  if (!key() || !item.text.trim()) return null;
  const entry: HistoryItem = { ...item, text: item.text.trim(), id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`, at: Date.now() };
  save([entry, ...loadHistory()]);
  return entry.id;
}

/** Changes an existing entry (e.g. a sentence result the user corrected). */
export function updateHistory(id: string, patch: Partial<Omit<HistoryItem, 'id' | 'at'>>): void {
  save(loadHistory().map((h) => (h.id === id ? { ...h, ...patch } : h)));
}

export function deleteHistory(id: string): void {
  save(loadHistory().filter((h) => h.id !== id));
}

export function clearHistory(): void {
  const k = key();
  if (!k) return;
  try {
    localStorage.removeItem(k);
  } catch {
    /* ignore */
  }
}
