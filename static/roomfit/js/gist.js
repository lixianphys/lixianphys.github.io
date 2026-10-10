// Sync through one private GitHub Gist holding the encrypted plan. Used by the GitHub Pages
// build. The Gist only ever contains ciphertext plus the (non-secret) salt and a timestamp.
import { seal, unseal } from './vault.js';

const API = 'https://api.github.com';
export const GIST_FILE = 'roomfit-plan.json';
const GIST_DESC = 'RoomFit plan (encrypted)';
// Every write makes a Gist revision, and GitHub limits how many an account may make, so
// writes wait for a pause in editing and are spaced out. Leaving the page flushes at once.
const PUSH_DELAY = 4000;
const MIN_GAP = 20000;
const POLL_MS = 30000;

export class GitHubError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function gh(token, path, { method = 'GET', body, keepalive = false } = {}) {
  const res = await fetch(API + path, {
    method,
    cache: 'no-store',
    keepalive,
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    // GitHub limits how often an account may write ("secondary rate limit", 403 or 429)
    if (res.status === 429 || (res.status === 403 && (/rate limit/i.test(text) || res.headers.get('x-ratelimit-remaining') === '0'))) {
      const after = parseInt(res.headers.get('retry-after'), 10);
      const reset = parseInt(res.headers.get('x-ratelimit-reset'), 10);
      const wait = after > 0 ? after * 1000 : reset > 0 && res.headers.get('x-ratelimit-remaining') === '0' ? reset * 1000 - Date.now() : 0;
      const err = new GitHubError(429, 'GitHub asked RoomFit to slow down.');
      err.wait = Math.max(wait, 60000);
      throw err;
    }
    const msg = res.status === 401 ? 'GitHub rejected the token. It may have expired or been revoked.'
      : res.status === 403 ? 'The token can’t access Gists. Create one with the “gist” scope.'
      : res.status === 404 ? 'The synced plan’s Gist wasn’t found. It may have been deleted, or the token can’t access Gists.'
      : `GitHub answered ${res.status}.`;
    throw new GitHubError(res.status, msg);
  }
  return res.status === 204 ? null : res.json();
}

export async function findGist(token) {
  for (let page = 1; page <= 5; page++) {
    const list = await gh(token, `/gists?per_page=100&page=${page}`);
    const hit = list.find((g) => g.files && g.files[GIST_FILE] && g.description === GIST_DESC);
    if (hit) return hit.id;
    if (list.length < 100) break;
  }
  return null;
}

export async function readGist(token, id) {
  const g = await gh(token, `/gists/${id}`);
  const f = g.files[GIST_FILE];
  if (!f) throw new GitHubError(404, 'The RoomFit Gist has no plan file.');
  const text = f.truncated ? await (await fetch(f.raw_url, { cache: 'no-store' })).text() : f.content;
  return JSON.parse(text);
}

export async function createGist(token, payload) {
  const g = await gh(token, '/gists', {
    method: 'POST',
    body: { description: GIST_DESC, public: false, files: { [GIST_FILE]: { content: JSON.stringify(payload) } } },
  });
  return g.id;
}

export async function writeGist(token, id, payload, { keepalive = false } = {}) {
  await gh(token, `/gists/${id}`, { method: 'PATCH', keepalive, body: { files: { [GIST_FILE]: { content: JSON.stringify(payload) } } } });
}

export async function sealPlan(key, salt, iterations, plan) {
  return { app: 'roomfit', v: 1, salt, iterations, updatedAt: plan.updatedAt || 0, ...(await seal(key, JSON.stringify(plan))) };
}

export async function openPlan(key, payload) {
  return JSON.parse(await unseal(key, payload));
}

// The plan without its save time, to tell real changes from re-saves of the same content.
const content = (plan) => JSON.stringify({ ...plan, updatedAt: 0 });

// Same contract as sync.js startSync, for the Pages build.
export function gistSync({ token, gistId, key, salt, iterations }) {
  return function startGistSync(store, setStatus, { onRemote } = {}) {
    let timer = null, busy = false, again = false, stopped = false;
    let synced = null;      // the plan content in the Gist, as last read or written
    let lastWrite = 0;      // when this device last wrote
    let written = 0;        // updatedAt of the plan it wrote
    let pausedUntil = 0;    // GitHub asked us to wait until then
    let strikes = 0;        // refusals in a row, to back off further each time
    let failed = false;
    const dirty = () => synced === null || content(store.plan) !== synced;
    const ok = () => setStatus('synced', 'Synced to your private GitHub Gist (encrypted)');

    function schedule(delay = PUSH_DELAY) {
      clearTimeout(timer);
      const at = Math.max(Date.now() + delay, lastWrite + MIN_GAP, pausedUntil);
      timer = setTimeout(sync, at - Date.now());
    }

    // Read, then write only if this device has changes and the Gist has nothing newer.
    async function sync({ flush = false } = {}) {
      clearTimeout(timer);
      timer = null;
      if (stopped) return;
      if (busy) { again = true; return; }
      if (store.dragging) return schedule(1000);
      if (Date.now() < pausedUntil) return schedule(0);
      busy = true;
      failed = false;
      try {
        if (!flush) {
          const remote = await readGist(token, gistId);
          // older than what this device last wrote: a stale read, ignore it
          if (remote.updatedAt >= written) {
            const plan = await openPlan(key, remote);
            if (remote.updatedAt > (store.plan.updatedAt || 0)) {
              if (store.dragging) { again = true; return; }
              synced = content(plan);
              store.applyRemote(plan);
              setStatus('synced', 'Updated from another device');
              onRemote?.();
              return;
            }
            synced = content(plan);
          }
        }
        if (!dirty()) return ok();
        const text = content(store.plan), at = store.plan.updatedAt || 0;
        setStatus('syncing', 'Syncing…');
        lastWrite = Date.now();
        await writeGist(token, gistId, await sealPlan(key, salt, iterations, store.plan), { keepalive: flush });
        synced = text;
        written = at;
        strikes = 0;
        if (!dirty()) ok();
      } catch (e) {
        failed = true;
        fail(e);
      } finally {
        busy = false;
        if (again) { again = false; schedule(0); }
        else if (!stopped && !failed && dirty() && !timer) schedule();
      }
    }

    function fail(e) {
      if (e instanceof GitHubError && e.status === 401) {
        stopped = true;
        setStatus('error', `${e.message} Lock RoomFit, choose “Reset this device” and set it up again with a new token.`);
      } else if (e instanceof GitHubError && e.status === 429) {
        pausedUntil = Date.now() + Math.min(Math.max(e.wait, 60000 * 2 ** strikes), 30 * 60000);
        strikes++;
        const at = new Date(pausedUntil).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        setStatus('pending', `${e.message} Your changes are saved on this device and sync at about ${at}.`);
        schedule(0);
      } else if (e instanceof GitHubError) {
        setStatus('error', e.message);
      } else if (e?.name === 'OperationError') {
        stopped = true;
        setStatus('error', 'The synced plan was encrypted with a different password.');
      } else {
        setStatus('error', 'Offline. Changes are kept here and sync when you’re back online.');
      }
    }

    store.onSave = () => {
      setStatus('pending', 'Unsynced changes');
      schedule();
    };
    const flush = () => { if (synced !== null && dirty() && !busy && Date.now() >= pausedUntil) sync({ flush: true }); };
    const poll = () => { if (!document.hidden && !timer) sync(); };
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) flush();
      else poll();
    });
    window.addEventListener('pagehide', flush);
    window.addEventListener('online', poll);
    setInterval(poll, POLL_MS);
    sync();
  };
}
