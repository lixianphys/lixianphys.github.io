// Cross-device sync through the claude.ai artifact store. The plan lives in the viewer's
// private subtree (data/users/<id>/plan), so only its owner sees it. Newest edit wins.
// Without the store (running locally, signed out) the app keeps working from localStorage.

const PUSH_DELAY = 800;

export async function startSync(store, setStatus, { onRemote } = {}) {
  if (!window.claude?.use) return setStatus('local', 'Saved in this browser only');
  setStatus('connecting', 'Connecting…');
  const [db, user] = await Promise.all([window.claude.use('db'), window.claude.use('user')]);
  const id = user ? await user.id() : null;
  if (!db || !id) return setStatus('local', 'Sign in to claude.ai to sync between devices');

  const ref = db.doc(`data/users/${id}/plan`);
  let timer = null, writing = false, again = false, lastWritten = 0, readOnly = false;

  async function push() {
    if (readOnly) return;
    if (writing) { again = true; return; }
    writing = true;
    setStatus('syncing', 'Syncing…');
    const plan = store.plan;
    try {
      await ref.set({ json: JSON.stringify(plan), updatedAt: plan.updatedAt || Date.now() });
      lastWritten = plan.updatedAt;
      setStatus('synced', 'Synced to your account');
    } catch (e) {
      if (e?.code === 'invalid_argument' || e?.code === 'not_granted') {
        readOnly = true;
        setStatus('local', 'This copy can’t sync, so changes stay in this browser');
      } else if (e?.code === 'quota_exceeded') {
        setStatus('error', 'Sync storage is full. Export a backup and remove unused items.');
      } else {
        setStatus('error', 'Sync failed, will retry on the next change');
      }
    } finally {
      writing = false;
      if (again) { again = false; push(); }
    }
  }

  store.onSave = () => {
    clearTimeout(timer);
    setStatus('pending', 'Unsynced changes');
    timer = setTimeout(push, PUSH_DELAY);
  };

  ref.onSnapshot((snap) => {
    if (snap.metadata.hasPendingWrites) return;
    const local = store.plan.updatedAt || 0;
    if (!snap.exists) {
      // First time on this account: upload whatever this browser has.
      push();
      return;
    }
    const data = snap.data();
    if (data.updatedAt === lastWritten) { setStatus('synced', 'Synced to your account'); return; }
    if (data.updatedAt > local) {
      if (store.dragging) return; // the drag's own save will win
      try {
        store.applyRemote(JSON.parse(data.json));
        lastWritten = data.updatedAt;
        setStatus('synced', 'Updated from another device');
        onRemote?.();
      } catch {
        setStatus('error', 'The synced plan could not be read');
      }
    } else if (data.updatedAt < local) {
      push();
    } else {
      setStatus('synced', 'Synced to your account');
    }
  }, () => setStatus('error', 'Lost connection to sync. Reload to reconnect.'));
}

export async function saveFile(filename, text) {
  const downloads = window.claude?.use ? await window.claude.use('downloads') : null;
  if (downloads) {
    await downloads.save({ filename, data: text });
    return;
  }
  // running outside claude.ai: a plain download link works
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
