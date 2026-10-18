import './style.css';
import { currentUser } from './auth';
import { clearHistory, deleteHistory, loadHistory, type HistoryItem, type HistoryKind } from './history';

// History page: the signed-in user's activity, grouped by day, filterable and clearable.

const list = document.getElementById('history-list')!;
const empty = document.getElementById('history-empty')!;
const summary = document.getElementById('history-summary')!;
const search = document.getElementById('history-search') as HTMLInputElement;
const clearBtn = document.getElementById('history-clear') as HTMLButtonElement;
const tabs = [...document.querySelectorAll<HTMLButtonElement>('[data-filter]')];

const user = currentUser();
document.getElementById('history-user')!.textContent = user?.displayName ?? 'you';

let filter: 'all' | HistoryKind = 'all';

const KIND: Record<HistoryKind, { label: string; icon: string; tint: string }> = {
  translate: { label: 'Translator', icon: '🔤', tint: 'bg-brand-soft text-brand-dark' },
  sentence: { label: 'Sentence', icon: '💬', tint: 'bg-accent-soft text-accent' },
  math: { label: 'Math', icon: '➗', tint: 'bg-sand text-ink' },
};

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

function dayLabel(t: number): string {
  const d = new Date(t);
  const today = new Date();
  const yesterday = new Date(Date.now() - 864e5);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
}

function row(h: HistoryItem): string {
  const k = KIND[h.kind];
  const time = new Date(h.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return `<li class="flex items-start gap-4 px-5 py-4">
    <span class="grid size-10 shrink-0 place-items-center rounded-2xl ${k.tint}" aria-hidden="true">${k.icon}</span>
    <div class="min-w-0 flex-1">
      <p class="break-words font-semibold text-ink">${esc(h.text)}</p>
      <p class="mt-0.5 text-xs text-ink/50">${k.label}${h.lang ? ` · ${h.lang.toUpperCase()}` : ''}${h.detail ? ` · ${esc(h.detail)}` : ''} · ${time}</p>
    </div>
    <button class="grid size-9 shrink-0 place-items-center rounded-full text-ink/40 transition hover:bg-ink/5 hover:text-accent" data-delete="${h.id}" aria-label="Delete this entry" title="Delete">
      <svg viewBox="0 0 24 24" class="size-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>
    </button>
  </li>`;
}

function render() {
  const all = loadHistory();
  const q = search.value.trim().toLowerCase();
  const items = all.filter((h) => (filter === 'all' || h.kind === filter) && (!q || `${h.text} ${h.detail ?? ''}`.toLowerCase().includes(q)));

  clearBtn.disabled = all.length === 0;
  clearBtn.classList.toggle('opacity-40', all.length === 0);
  empty.classList.toggle('hidden', all.length > 0);
  summary.textContent = all.length
    ? items.length === all.length
      ? `${all.length} ${all.length === 1 ? 'entry' : 'entries'}`
      : `${items.length} of ${all.length} entries`
    : '';

  const groups = new Map<string, HistoryItem[]>();
  for (const h of items) {
    const label = dayLabel(h.at);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(h);
  }
  list.innerHTML =
    [...groups.entries()]
      .map(
        ([label, hs]) => `<section>
          <h2 class="text-xs font-semibold uppercase tracking-widest text-ink/50">${label}</h2>
          <ul class="card mt-3 divide-y divide-ink/10 !p-0">${hs.map(row).join('')}</ul>
        </section>`,
      )
      .join('') || (all.length ? '<p class="text-sm text-ink/60">No entries match.</p>' : '');
}

list.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-delete]');
  if (!btn) return;
  deleteHistory(btn.dataset.delete!);
  render();
});

clearBtn.addEventListener('click', () => {
  if (!confirm('Delete your whole history from this browser? This cannot be undone.')) return;
  clearHistory();
  render();
});

tabs.forEach((t) =>
  t.addEventListener('click', () => {
    filter = t.dataset.filter as typeof filter;
    tabs.forEach((b) => b.setAttribute('aria-selected', String(b === t)));
    render();
  }),
);
search.addEventListener('input', render);
// Another tab (translator, sentence page) may add entries while this one is open.
window.addEventListener('storage', render);

render();
