// Password-based encryption (PBKDF2-SHA256 → AES-256-GCM) for the GitHub Pages build.
// The plan and the GitHub token never touch storage or the network unencrypted.

const enc = new TextEncoder();
const dec = new TextDecoder();
export const ITERATIONS = 600000;
const VAULT_KEY = 'roomfit:vault';

function b64(buf) {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

export const newSalt = () => b64(crypto.getRandomValues(new Uint8Array(16)));

export async function deriveKey(password, salt, iterations = ITERATIONS) {
  const base = await crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: unb64(salt), iterations, hash: 'SHA-256' },
    base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt'],
  );
}

export async function seal(key, text) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, enc.encode(text));
  return { iv: b64(iv), ct: b64(ct) };
}

// Throws when the key is wrong (AES-GCM authentication fails).
export async function unseal(key, box) {
  const pt = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: unb64(box.iv) }, key, unb64(box.ct));
  return dec.decode(pt);
}

export function readVault() {
  try { return JSON.parse(localStorage.getItem(VAULT_KEY)); } catch { return null; }
}
export function writeVault(v) {
  localStorage.setItem(VAULT_KEY, JSON.stringify(v));
}
export function forgetVault() {
  try { localStorage.removeItem(VAULT_KEY); } catch { /* ignore */ }
}

// Store adapter: keeps the plan encrypted in localStorage. Saves are queued so only the
// latest state is written when edits come quickly.
export function encryptedStorage(vault, key, initialPlan) {
  let pending = null, busy = false;
  async function flush() {
    busy = true;
    while (pending) {
      const text = pending;
      pending = null;
      vault.plan = await seal(key, text);
      try { writeVault(vault); } catch { /* storage full or blocked; the Gist still has it */ }
    }
    busy = false;
  }
  return {
    load: () => initialPlan,
    save(plan) {
      pending = JSON.stringify(plan);
      if (!busy) flush();
    },
  };
}
