// Entry point for the GitHub Pages build: password lock screen, then the app.
// First visit on a device asks for a password and a GitHub token (gist scope); later visits
// only the password. Everything stored or synced is encrypted with a key derived from it.
import {
  ITERATIONS, newSalt, deriveKey, seal, unseal, readVault, writeVault, forgetVault, encryptedStorage,
} from './vault.js?v=aab60b9078';
import { findGist, readGist, createGist, writeGist, sealPlan, openPlan, gistSync } from './gist.js?v=aab60b9078';
import { defaultPlan } from './model.js?v=aab60b9078';

window.__roomfitStarted = true; // the lock screen counts as started for the boot check

const AUTO_LOCK_MS = 15 * 60 * 1000;
const TOKEN_URL = 'https://github.com/settings/tokens/new?scopes=gist&description=RoomFit';
const CHECK = 'roomfit-ok';

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const app = document.getElementById('app');
app.hidden = true;
const lock = document.createElement('div');
lock.className = 'lock';
document.body.appendChild(lock);

function screen(html) {
  lock.innerHTML = `<form class="lock-card" id="lock-form" novalidate>
    <p class="lock-brand"><b>Room<span>Fit</span></b></p>${html}
    <p class="lock-error" id="lock-error" role="alert" hidden></p>
  </form>`;
  lock.querySelector('input')?.focus();
  return lock.querySelector('#lock-form');
}
function showError(msg) {
  const el = lock.querySelector('#lock-error');
  el.textContent = msg;
  el.hidden = !msg;
}
function busy(form, on, label) {
  const btn = form.querySelector('button[type=submit]');
  btn.disabled = on;
  if (label) btn.textContent = label;
}

const vault = readVault();
if (vault?.salt && vault.token && vault.check) unlockScreen(vault);
else setupScreen();

function unlockScreen(v) {
  const form = screen(`
    <h1>Unlock your plan</h1>
    <label class="field"><span>Password</span><input id="pw" type="password" autocomplete="current-password" required></label>
    <button class="btn primary" type="submit">Unlock</button>
    <button class="linklike small" type="button" id="forget">Forgot it? Reset this device</button>`);
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    showError('');
    busy(form, true, 'Unlocking…');
    try {
      const key = await deriveKey(form.querySelector('#pw').value, v.salt, v.iterations || ITERATIONS);
      if (await unseal(key, v.check) !== CHECK) throw new Error('bad');
      const token = await unseal(key, v.token);
      const plan = v.plan ? JSON.parse(await unseal(key, v.plan)) : defaultPlan();
      start({ vault: v, key, token, plan });
    } catch {
      busy(form, false, 'Unlock');
      showError('That password is wrong. Try again.');
      form.querySelector('#pw').select();
    }
  });
  const forget = form.querySelector('#forget');
  forget.addEventListener('click', () => {
    if (forget.dataset.armed) { forgetVault(); setupScreen(); return; }
    forget.dataset.armed = '1';
    forget.textContent = 'Tap again to remove the saved token and plan from this device. Your synced plan stays in the Gist.';
  });
}

function setupScreen() {
  const form = screen(`
    <h1>Set up this device</h1>
    <p class="small">Your plan is encrypted with your password and synced through a private Gist on your GitHub account. Use the same password on every device.</p>
    <label class="field"><span>Password</span><input id="pw" type="password" autocomplete="new-password" minlength="8" required></label>
    <label class="field"><span>Repeat password</span><input id="pw2" type="password" autocomplete="new-password" required></label>
    <label class="field"><span>GitHub token</span><input id="token" type="password" autocomplete="off" spellcheck="false" placeholder="ghp_…" required></label>
    <p class="small muted">On your first device, <a href="${TOKEN_URL}" target="_blank" rel="noopener">create a token with only the “gist” scope</a> and copy it. On your other devices, paste the same token (or make another one). It’s stored here encrypted with your password.</p>
    <label class="check small" id="replace-row" hidden><input type="checkbox" id="replace"> I forgot the old password. Replace the synced plan with a new, empty one.</label>
    <button class="btn primary" type="submit">Unlock RoomFit</button>`);
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    showError('');
    const pw = form.querySelector('#pw').value;
    const token = form.querySelector('#token').value.trim();
    if (pw.length < 8) return showError('Use a password of at least 8 characters.');
    if (pw !== form.querySelector('#pw2').value) return showError('The two passwords don’t match.');
    if (!token) return showError('Paste your GitHub token.');
    busy(form, true, 'Connecting to GitHub…');
    try {
      let gistId = await findGist(token);
      let salt, iterations, key, plan;
      const replace = form.querySelector('#replace').checked;
      if (gistId && !replace) {
        // another device set this up already: the password must open the existing plan
        const remote = await readGist(token, gistId);
        salt = remote.salt; iterations = remote.iterations || ITERATIONS;
        busy(form, true, 'Checking password…');
        key = await deriveKey(pw, salt, iterations);
        try { plan = await openPlan(key, remote); } catch {
          form.querySelector('#replace-row').hidden = false;
          throw new Error('This password doesn’t match the one you used on your other device.');
        }
      } else {
        salt = newSalt(); iterations = ITERATIONS;
        busy(form, true, 'Encrypting…');
        key = await deriveKey(pw, salt, iterations);
        plan = defaultPlan();
        const payload = await sealPlan(key, salt, iterations, plan);
        if (gistId) await writeGist(token, gistId, payload);
        else gistId = await createGist(token, payload);
      }
      const v = {
        v: 1, salt, iterations, gistId,
        check: await seal(key, CHECK),
        token: await seal(key, token),
        plan: await seal(key, JSON.stringify(plan)),
      };
      writeVault(v);
      start({ vault: v, key, token, plan });
    } catch (err) {
      busy(form, false, 'Unlock RoomFit');
      showError(err.message && !/fetch|network/i.test(err.message) ? err.message : 'Couldn’t reach GitHub. Check your connection and try again.');
    }
  });
}

async function start({ vault: v, key, token, plan }) {
  window.__roomfitBoot = {
    storage: encryptedStorage(v, key, plan),
    startSync: gistSync({ token, gistId: v.gistId, key, salt: v.salt, iterations: v.iterations || ITERATIONS }),
    lock: () => location.reload(),
  };
  lock.remove();
  app.hidden = false;
  // lock again after a while in the background
  let hiddenAt = 0;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) hiddenAt = Date.now();
    else if (hiddenAt && Date.now() - hiddenAt > AUTO_LOCK_MS) location.reload();
  });
  try {
    await import('./main.js?v=aab60b9078');
  } catch (e) {
    document.body.insertAdjacentHTML('beforeend', `<div class="boot-error"><h2>RoomFit couldn't start</h2><pre>${esc(e?.message || e)}</pre></div>`);
  }
}
