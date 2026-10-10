// Sync through one private GitHub Gist holding the encrypted plan. Used by the GitHub Pages
// build. The Gist only ever contains ciphertext plus the (non-secret) salt and a timestamp.
import { seal, unseal } from './vault.js';

const API = 'https://api.github.com';
export const GIST_FILE = 'roomfit-plan.json';
const GIST_DESC = 'RoomFit plan (encrypted)';
const PUSH_DELAY = 1500;
const POLL_MS = 30000;

export class GitHubError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function gh(token, path, { method = 'GET', body } = {}) {
  const res = await fetch(API + path, {
    method,
    cache: 'no-store',
    headers: {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const msg = res.status === 401 ? 'GitHub rejected the token. It may have expired or been revoked.'
      : res.status === 403 || res.status === 404 ? 'The token can’t access Gists. Create one with the “gist” scope.'
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

export async function writeGist(token, id, payload) {
  await gh(token, `/gists/${id}`, { method: 'PATCH', body: { files: { [GIST_FILE]: { content: JSON.stringify(payload) } } } });
}

export async function sealPlan(key, salt, iterations, plan) {
  return { app: 'roomfit', v: 1, salt, iterations, updatedAt: plan.updatedAt || 0, ...(await seal(key, JSON.stringify(plan))) };
}

export async function openPlan(key, payload) {
  return JSON.parse(await unseal(key, payload));
}

// Same contract as sync.js startSync, for the Pages build.
export function gistSync({ token, gistId, key, salt, iterations }) {
  return function startGistSync(store, setStatus, { onRemote } = {}) {
    let timer = null, writing = false, again = false, reading = false, stopped = false;

    async function push() {
      if (stopped) return;
      if (writing) { again = true; return; }
      writing = true;
      setStatus('syncing', 'Syncing…');
      try {
        await writeGist(token, gistId, await sealPlan(key, salt, iterations, store.plan));
        setStatus('synced', 'Synced to your private GitHub Gist (encrypted)');
      } catch (e) {
        fail(e);
      } finally {
        writing = false;
        if (again) { again = false; push(); }
      }
    }

    async function pull() {
      if (stopped || reading || writing || document.hidden) return;
      reading = true;
      try {
        const remote = await readGist(token, gistId);
        const local = store.plan.updatedAt || 0;
        if (remote.updatedAt > local) {
          if (store.dragging) return;
          store.applyRemote(await openPlan(key, remote));
          setStatus('synced', 'Updated from another device');
          onRemote?.();
        } else if (remote.updatedAt < local) {
          push();
        } else {
          setStatus('synced', 'Synced to your private GitHub Gist (encrypted)');
        }
      } catch (e) {
        fail(e);
      } finally {
        reading = false;
      }
    }

    function fail(e) {
      if (e instanceof GitHubError && e.status === 401) {
        stopped = true;
        setStatus('error', `${e.message} Use Plan ▾ → Forget this device, then set up again with a new token.`);
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
      clearTimeout(timer);
      setStatus('pending', 'Unsynced changes');
      timer = setTimeout(push, PUSH_DELAY);
    };
    document.addEventListener('visibilitychange', () => { if (!document.hidden) pull(); });
    window.addEventListener('online', () => pull());
    setInterval(pull, POLL_MS);
    pull();
  };
}
