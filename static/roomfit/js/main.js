import {
  Store, defaultPlan, blankPlan, rectRoom, newItem, wallOf, checkPlan, worstLevel, setWallLength,
  removeVertex, insertVertex, roomArea, roomAt, uid, FLOORS,
} from './model.js';
import { CATALOG } from './catalog.js';
import { Plan2D } from './plan2d.js';
import { bbox, snap, centroid, pointInPolygon } from './geometry.js';
import { startSync, saveFile } from './sync.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const money = (n) => n.toLocaleString(undefined, { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 });
const priceOf = (it) => parseFloat(String(it.price || '').replace(',', '.')) || 0;

const prefs = (() => {
  try { return JSON.parse(localStorage.getItem('roomfit:prefs')) || {}; } catch { return {}; }
})();
const savePrefs = () => { try { localStorage.setItem('roomfit:prefs', JSON.stringify(prefs)); } catch { /* ignore */ } };

// The GitHub Pages build (pages.js) hands over encrypted storage, Gist sync and a lock action.
const boot = window.__roomfitBoot || {};
const store = new Store(boot.storage);
let view3d = null;
let issues = new Map();

const plan2d = new Plan2D($('#plan-pane'), store, {
  onEditWall(roomId, i) {
    store.select({ kind: 'room', id: roomId });
    const input = document.getElementById(`wall-${i}`);
    input?.focus();
    input?.select();
  },
  onRoomDrawn(points) {
    const room = { id: uid(), name: `Room ${store.plan.rooms.length + 1}`, height: 250, floor: 'plain', points };
    store.edit((p) => { p.rooms.push(room); p.sample = false; store.selection = { kind: 'room', id: room.id }; });
    setTab('rooms');
    toast(`${room.name} added: ${roomArea(room).toFixed(1)} m². Rename it and set the ceiling height on the right.`);
    document.getElementById('f-name')?.focus();
  },
  onToolChange(tool) {
    $('#tool-measure').setAttribute('aria-pressed', String(tool === 'measure'));
    $('#tool-draw').setAttribute('aria-pressed', String(tool === 'draw'));
    $('#draw-help').hidden = tool !== 'draw';
    if (tool === 'draw') $('#draw-len').value = '';
  },
});

// ---------- toast ----------
let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3200);
}

// ---------- views ----------
async function setView(mode) {
  prefs.view = mode; savePrefs();
  const ws = $('#workspace');
  ws.className = `workspace mode-${mode}`;
  document.querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === mode)));
  if (mode !== 'plan' && !view3d) {
    try {
      const { View3D } = await import('./view3d.js');
      view3d = new View3D($('#view3d-pane'), store);
      view3d.showCandidates = plan2d.showCandidates;
    } catch (e) {
      console.error(e);
      toast('The 3D view could not load. Check your connection and reload.');
      return;
    }
  }
  plan2d.render();
  view3d?.resize();
  view3d?.schedule();
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

function setTab(name) {
  prefs.tab = name; savePrefs();
  document.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  for (const t of ['rooms', 'furniture', 'shopping']) $(`#tab-${t}`).hidden = t !== name;
}
document.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));

// ---------- top bar ----------
$('#plan-name').addEventListener('change', (e) => {
  const v = e.target.value.trim() || 'My apartment';
  store.edit((p) => { p.name = v; });
});
$('#undo').addEventListener('click', () => store.undo());
$('#redo').addEventListener('click', () => store.redo());
$('#fit').addEventListener('click', () => { plan2d.fit(); view3d?.frame(); });

function setMeasure(on) {
  plan2d.setTool(on ? 'measure' : 'select');
  if (on) toast('Click two points to measure. Esc to stop.');
}
function setDraw(on) {
  if (on && prefs.view === '3d') setView('both');
  plan2d.setTool(on ? 'draw' : 'select');
}
$('#tool-draw').addEventListener('click', () => setDraw(plan2d.tool !== 'draw'));
$('#draw-done').addEventListener('click', () => { if (!plan2d.finishDraft()) toast('A room needs at least 3 corners.'); });
$('#draw-cancel').addEventListener('click', () => setDraw(false));
document.querySelectorAll('[data-dir]').forEach((b) => b.addEventListener('click', () => {
  const L = parseFloat($('#draw-len').value);
  if (!(L > 0)) { toast('Type the wall length in cm first.'); $('#draw-len').focus(); return; }
  if (plan2d.placeDraftStep(b.dataset.dir, L) === 'no-start') { toast('Tap the plan to place the first corner.'); return; }
  $('#draw-len').value = '';
}));
$('#tool-measure').addEventListener('click', () => setMeasure(plan2d.tool !== 'measure'));
$('#toggle-snap').addEventListener('click', (e) => {
  plan2d.snapOn = !plan2d.snapOn;
  e.currentTarget.setAttribute('aria-pressed', String(plan2d.snapOn));
});
$('#toggle-candidates').addEventListener('click', (e) => {
  plan2d.showCandidates = !plan2d.showCandidates;
  if (view3d) { view3d.showCandidates = plan2d.showCandidates; view3d.dirty = true; view3d.schedule(); }
  e.currentTarget.setAttribute('aria-pressed', String(plan2d.showCandidates));
  plan2d.render();
});

// plan menu
const menu = $('#menu');
$('#menu-btn').addEventListener('click', (e) => {
  e.stopPropagation();
  menu.hidden = !menu.hidden;
  $('#menu-btn').setAttribute('aria-expanded', String(!menu.hidden));
});
document.addEventListener('click', (e) => {
  if (!menu.hidden && !e.target.closest('.menu-wrap')) { menu.hidden = true; $('#menu-btn').setAttribute('aria-expanded', 'false'); }
});
const closeMenu = () => { menu.hidden = true; $('#menu-btn').setAttribute('aria-expanded', 'false'); };

const fileName = () => `${(store.plan.name || 'plan').replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '-').toLowerCase() || 'plan'}-roomfit.json`;
$('#m-export').addEventListener('click', async () => {
  closeMenu();
  try {
    await saveFile(fileName(), JSON.stringify(store.plan, null, 2));
  } catch (e) {
    if (e?.code !== 'declined') toast('Export didn’t work here. Use “Copy plan as text” instead.');
  }
});
$('#m-import').addEventListener('click', () => { closeMenu(); $('#import-file').click(); });
$('#import-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    store.replace(JSON.parse(await file.text()));
    plan2d.fit(); view3d?.frame();
    toast(`Imported “${store.plan.name}”. Undo brings back the previous plan.`);
  } catch {
    toast('That file isn’t a RoomFit backup. Pick a .json file exported from RoomFit.');
  }
});
$('#m-copy').addEventListener('click', async () => {
  closeMenu();
  try {
    await navigator.clipboard.writeText(JSON.stringify(store.plan));
    toast('Plan copied. Paste it into a text file to keep a backup.');
  } catch {
    toast('Copying isn’t allowed here. Use “Export backup” instead.');
  }
});
function startBlank() {
  store.replace(blankPlan());
  plan2d.fit(); view3d?.frame();
  setTab('rooms');
  store.select({ kind: 'room', id: store.plan.rooms[0].id });
  toast('Blank plan started. Undo brings back the previous one.');
}
$('#m-blank').addEventListener('click', () => { closeMenu(); startBlank(); });
$('#m-sample').addEventListener('click', () => {
  closeMenu();
  store.replace(defaultPlan());
  plan2d.fit(); view3d?.frame();
  toast('Sample flat loaded. Undo brings back your plan.');
});
$('#banner-blank').addEventListener('click', startBlank);
$('#banner-keep').addEventListener('click', () => store.edit((p) => { p.sample = false; }));

// sync status
function setSync(state, text) {
  const s = $('#sync');
  s.dataset.state = state;
  s.textContent = { synced: 'Synced', pending: 'Unsynced', syncing: 'Syncing…', connecting: 'Connecting…', error: 'Sync problem', local: 'This browser only' }[state] || text;
  s.title = text;
}

// ---------- inspector ----------
const NUM = (k, label, unit = 'cm', extra = {}) => ({ k, label, type: 'number', unit, ...extra });

function fieldHtml(f, obj) {
  const id = `f-${f.k}`;
  const v = obj[f.k];
  const cls = `field ${f.cls || ''}`;
  if (f.type === 'select') {
    const opts = f.options.map(([val, txt]) => `<option value="${esc(val)}"${String(v) === String(val) ? ' selected' : ''}>${esc(txt)}</option>`).join('');
    return `<label class="${cls}"><span>${f.label}</span><select id="${id}" data-k="${f.k}">${opts}</select></label>`;
  }
  if (f.type === 'checkbox') {
    return `<label class="check ${f.cls || ''}"><input type="checkbox" id="${id}" data-k="${f.k}"${v ? ' checked' : ''}> ${f.label}</label>`;
  }
  if (f.type === 'textarea') {
    return `<label class="${cls}"><span>${f.label}</span><textarea id="${id}" data-k="${f.k}" rows="2">${esc(v)}</textarea></label>`;
  }
  const input = `<input id="${id}" data-k="${f.k}" type="${f.type}" value="${esc(v)}"${f.type === 'number' ? ` step="${f.step || 1}" min="${f.min ?? 0}" inputmode="decimal"` : ''}${f.placeholder ? ` placeholder="${esc(f.placeholder)}"` : ''}>`;
  return `<label class="${cls}"><span>${f.label}</span>${f.unit ? `<span class="unit" data-unit="${f.unit}">${input}</span>` : input}</label>`;
}

let inspectorKey = '';
function renderInspector() {
  const sel = store.selection;
  const target = store.selected;
  const key = !target ? 'none'
    : `${sel.kind}:${sel.id}:${sel.index ?? ''}:${target.shape || target.type || ''}:${target.points?.length || ''}:${target.swing || ''}`;
  const box = $('#inspector');
  if (key !== inspectorKey) {
    inspectorKey = key;
    if (!target) box.innerHTML = overviewHtml();
    else if (sel.kind === 'item') box.innerHTML = itemHtml(target);
    else if (sel.kind === 'opening') box.innerHTML = openingHtml(target);
    else box.innerHTML = roomHtml(target, sel);
  } else {
    // same thing selected: refresh values in place so typing isn't interrupted
    box.querySelectorAll('[data-k]').forEach((input) => {
      if (input === document.activeElement || !target) return;
      const v = target[input.dataset.k];
      if (input.type === 'checkbox') input.checked = !!v;
      else if (input.value !== String(v ?? '')) input.value = v ?? '';
    });
    box.querySelectorAll('[data-wall]').forEach((input) => {
      if (input === document.activeElement) return;
      input.value = Math.round(wallOf(target, +input.dataset.wall).length * 10) / 10;
    });
    if (sel?.kind === 'vertex') {
      const p = target.points[sel.index];
      for (const [axis, input] of [['x', box.querySelector('#vx')], ['y', box.querySelector('#vy')]]) {
        if (input && input !== document.activeElement) input.value = Math.round(p[axis] * 10) / 10;
      }
    }
    if (!target) box.innerHTML = overviewHtml();
  }
  const dyn = box.querySelector('#insp-dynamic');
  if (dyn) dyn.innerHTML = dynamicHtml(target, sel);
}

function issuesHtml(list) {
  if (!list?.length) return '<ul class="issues"><li class="ok">Fits. Nothing in the way.</li></ul>';
  return `<ul class="issues">${list.map((i) => `<li class="${i.level}">${esc(i.msg)}</li>`).join('')}</ul>`;
}

function dynamicHtml(target, sel) {
  if (!target) return '';
  if (sel.kind === 'item') return issuesHtml(issues.get(target.id));
  if (sel.kind === 'room' || sel.kind === 'vertex') {
    const corner = sel.kind === 'vertex' ? ` · corner ${sel.index + 1} is ${Math.round(cornerAngle(target, sel.index))}°` : '';
    return `<p class="muted small">${roomArea(target).toFixed(2)} m² floor · ${(roomArea(target) * target.height / 100).toFixed(1)} m³ · ${target.points.length} corners${corner}</p>`;
  }
  return '';
}

function overviewHtml() {
  const { plan } = store;
  const area = plan.rooms.reduce((s, r) => s + roomArea(r), 0);
  const bad = plan.items.filter((it) => !it.hidden && issues.get(it.id)?.some((i) => i.level !== 'info'));
  const problems = bad.map((it) => {
    const lvl = worstLevel(issues.get(it.id));
    return `<li class="${lvl}"><button class="linklike" data-select-item="${it.id}">${esc(it.name)}</button>: ${esc(issues.get(it.id)[0].msg)}</li>`;
  }).join('');
  return `
    <p class="eyebrow">Overview</p>
    <h2>${esc(plan.name)}</h2>
    <div class="stats">
      <div class="stat"><b>${area.toFixed(1)}</b><span>m² floor</span></div>
      <div class="stat"><b>${plan.items.length}</b><span>items</span></div>
      <div class="stat"><b>${bad.length}</b><span>${bad.length === 1 ? 'problem' : 'problems'}</span></div>
    </div>
    ${bad.length ? `<ul class="issues">${problems}</ul>` : '<ul class="issues"><li class="ok">Everything fits.</li></ul>'}
    <p class="muted small">Click a room to enter its wall lengths. Pick furniture below, type in the sizes from the shop, and drag it into place.</p>`;
}

function itemHtml(it) {
  const f = (x) => fieldHtml(x, it);
  const L = it.shape === 'L';
  return `
    <p class="eyebrow">Furniture · ${it.status === 'have' ? 'I have this' : 'Considering'}</p>
    <h2>${esc(it.name)}</h2>
    <div id="insp-dynamic"></div>
    <div class="seg" role="group" aria-label="Status">
      <button data-action="status-have" aria-pressed="${it.status === 'have'}">I have this</button>
      <button data-action="status-candidate" aria-pressed="${it.status === 'candidate'}">Considering</button>
    </div>
    <div class="fields">
      ${f({ k: 'name', label: 'Name', type: 'text', cls: 'wide' })}
      ${f(NUM('w', 'Width', 'cm', { min: 1 }))}
      ${f(NUM('d', 'Depth', 'cm', { min: 1 }))}
      ${f(NUM('h', 'Height', 'cm', { min: 0 }))}
      ${f({ k: 'shape', label: 'Shape', type: 'select', options: [['rect', 'Rectangle'], ['ellipse', 'Round / oval'], ['L', 'Corner (L)']] })}
      ${f(NUM('elevation', 'Off floor', 'cm'))}
      ${f(NUM('rot', 'Rotation', '°', { step: 5, min: -360 }))}
      ${L ? f(NUM('lw', 'Chaise width', 'cm', { min: 1 })) + f(NUM('ld', 'Seat depth', 'cm', { min: 1 })) + f({ k: 'mirror', label: 'Mirror', type: 'checkbox' }) : ''}
    </div>
    <div class="row">
      <button class="btn" data-action="rot-left" title="Rotate left (Shift+R)">⟲ 90°</button>
      <button class="btn" data-action="rot-right" title="Rotate right (R)">⟳ 90°</button>
      <button class="btn" data-action="duplicate">Duplicate</button>
      <button class="btn danger" data-action="delete">Remove</button>
    </div>
    <div class="fields">
      ${f({ k: 'color', label: 'Colour', type: 'color' })}
      ${f({ k: 'price', label: 'Price (€)', type: 'text', placeholder: '0' })}
      <div class="field"><span>&nbsp;</span>${f({ k: 'hidden', label: 'Hide', type: 'checkbox' })}</div>
      ${f({ k: 'url', label: 'Product link', type: 'url', cls: 'wide', placeholder: 'https://' })}
      ${it.url ? `<a class="small wide" href="${esc(it.url)}" target="_blank" rel="noopener">Open product page ↗</a>` : ''}
      ${f({ k: 'notes', label: 'Notes', type: 'textarea', cls: 'wide' })}
    </div>
    <p class="muted small">Blue numbers on the plan are the gaps in cm to the nearest wall or furniture. The dark edge marks the front. Allow about 70 cm for walkways and 90 cm in front of wardrobes.</p>`;
}

function roomHtml(room, sel) {
  const f = (x) => fieldHtml(x, room);
  const walls = room.points.map((_, i) => {
    const w = wallOf(room, i);
    return `<div class="wall-row">
      <label for="wall-${i}">Wall ${i + 1}</label>
      <span class="unit" data-unit="cm"><input id="wall-${i}" data-wall="${i}" type="number" step="0.5" min="1" inputmode="decimal" value="${Math.round(w.length * 10) / 10}"></span>
      <button class="btn icon" data-action="split-wall" data-index="${i}" title="Add a corner in the middle of wall ${i + 1}" aria-label="Add a corner to wall ${i + 1}">+</button>
    </div>`;
  }).join('');
  const v = sel.kind === 'vertex' ? room.points[sel.index] : null;
  const cornerBox = v ? `
    <div class="corner-box">
      <p class="eyebrow">Corner ${sel.index + 1} position</p>
      <div class="fields two">
        <label class="field"><span>X</span><span class="unit" data-unit="cm"><input id="vx" data-vertex-axis="x" type="number" step="0.5" inputmode="decimal" value="${Math.round(v.x * 10) / 10}"></span></label>
        <label class="field"><span>Y</span><span class="unit" data-unit="cm"><input id="vy" data-vertex-axis="y" type="number" step="0.5" inputmode="decimal" value="${Math.round(v.y * 10) / 10}"></span></label>
      </div>
      <div class="row"><button class="btn danger" data-action="delete-vertex">Remove corner</button></div>
    </div>` : '';
  const ops = store.plan.openings.filter((o) => o.roomId === room.id);
  return `
    <p class="eyebrow">Room</p>
    <h2>${esc(room.name)}</h2>
    <div id="insp-dynamic"></div>
    <div class="fields">
      ${f({ k: 'name', label: 'Name', type: 'text', cls: 'wide' })}
      ${f(NUM('height', 'Ceiling', 'cm', { min: 50 }))}
      ${f({ k: 'floor', label: 'Floor', type: 'select', options: Object.entries(FLOORS), cls: 'wide' })}
    </div>
    ${cornerBox}
    <h3 class="eyebrow">Walls · inside lengths</h3>
    <div class="walls">${walls}</div>
    <p class="muted small">Tap a <b>+</b> on a wall (on the plan or here) to add a corner, then drag it into place. Select a corner to type its exact position or remove it. Corners snap to neighbouring rooms.</p>
    <div class="row">
      <button class="btn" data-action="add-door">Add door</button>
      <button class="btn" data-action="add-window">Add window</button>
      <button class="btn danger" data-action="delete-room">Remove room</button>
    </div>
    ${ops.length ? `<ul class="list">${ops.map((o) => `<li><button data-select-opening="${o.id}"><span class="swatch"></span><span class="name">${o.type === 'door' ? (o.swing === 'none' ? 'Opening' : 'Door') : 'Window'} on wall ${o.wall + 1}</span><span class="meta">${o.width} cm</span></button></li>`).join('')}</ul>` : ''}`;
}

function openingHtml(op) {
  const room = store.plan.rooms.find((r) => r.id === op.roomId);
  const f = (x) => fieldHtml(x, op);
  const wallOpts = room.points.map((_, i) => [i, `Wall ${i + 1} (${Math.round(wallOf(room, i).length)} cm)`]);
  const isDoor = op.type === 'door';
  return `
    <p class="eyebrow">${isDoor ? 'Door' : 'Window'} · ${esc(room.name)}</p>
    <h2>${isDoor ? (op.swing === 'none' ? 'Open passage' : 'Door') : 'Window'}</h2>
    <div class="fields">
      ${f({ k: 'type', label: 'Type', type: 'select', options: [['door', 'Door'], ['window', 'Window']] })}
      ${f({ k: 'wall', label: 'Wall', type: 'select', options: wallOpts, cls: 'wide' })}
      ${f(NUM('offset', 'From corner'))}
      ${f(NUM('width', 'Width', 'cm', { min: 10 }))}
      ${f(NUM('height', 'Height', 'cm', { min: 10 }))}
      ${isDoor ? f({ k: 'swing', label: 'Opens', type: 'select', options: [['in', 'Into this room'], ['out', 'Out of this room'], ['none', 'No door (opening)']], cls: 'wide' })
        + (op.swing !== 'none' ? f({ k: 'hinge', label: 'Hinge', type: 'select', options: [['start', 'Left end'], ['end', 'Right end']] }) : '')
        : f(NUM('sill', 'Sill height'))}
    </div>
    <p class="muted small">Drag it along the wall on the plan. ${isDoor ? 'Furniture in the dashed swing area gets flagged.' : 'Anything taller than the sill in front of it gets a note.'}</p>
    <div class="row">
      <button class="btn" data-action="select-room" data-room="${room.id}">Back to room</button>
      <button class="btn danger" data-action="delete-opening">Remove</button>
    </div>`;
}

// field edits
$('#inspector').addEventListener('change', (e) => {
  const input = e.target;
  const target = store.selected;
  if (!target) return;
  if (input.dataset.vertexAxis) {
    const v = parseFloat(input.value);
    const idx = store.selection.index;
    if (Number.isNaN(v) || idx === undefined) return renderInspector();
    store.edit(() => { target.points[idx] = { ...target.points[idx], [input.dataset.vertexAxis]: v }; });
    return;
  }
  if (input.dataset.wall !== undefined) {
    const v = parseFloat(input.value);
    if (!(v > 0)) return renderInspector();
    store.edit(() => setWallLength(target, +input.dataset.wall, v));
    return;
  }
  const k = input.dataset.k;
  if (!k) return;
  let v;
  if (input.type === 'checkbox') v = input.checked;
  else if (input.type === 'number' || k === 'wall') {
    v = parseFloat(input.value);
    if (Number.isNaN(v)) return renderInspector();
    if (input.min !== '' && v < +input.min) v = +input.min;
  } else v = input.value;
  if (k === 'rot') v = ((v % 360) + 360) % 360;
  store.edit(() => {
    target[k] = v;
    if (k === 'type' && v === 'window') { if (!target.sill) target.sill = 90; target.height = Math.min(target.height, 150); }
    if (k === 'type' && v === 'door') { if (!target.swing) target.swing = 'in'; if (!target.hinge) target.hinge = 'start'; target.height = Math.max(target.height, 200); }
  });
});

$('#inspector').addEventListener('click', (e) => {
  const sel = e.target.closest('[data-select-item],[data-select-opening]');
  if (sel) {
    if (sel.dataset.selectItem) store.select({ kind: 'item', id: sel.dataset.selectItem });
    else store.select({ kind: 'opening', id: sel.dataset.selectOpening });
    return;
  }
  const btn = e.target.closest('[data-action]');
  if (btn) runAction(btn.dataset.action, btn);
});

function runAction(action, btn) {
  const target = store.selected;
  const sel = store.selection;
  switch (action) {
    case 'status-have': case 'status-candidate':
      store.edit(() => { target.status = action === 'status-have' ? 'have' : 'candidate'; });
      inspectorKey = '';
      renderInspector();
      break;
    case 'rot-left': rotateSelected(-90); break;
    case 'rot-right': rotateSelected(90); break;
    case 'duplicate': duplicateSelected(); break;
    case 'delete': case 'delete-opening': deleteSelected(); break;
    case 'delete-room':
      store.edit((p) => {
        p.rooms = p.rooms.filter((r) => r.id !== target.id);
        p.openings = p.openings.filter((o) => o.roomId !== target.id);
        store.selection = null;
      });
      toast(`Removed ${target.name}. Undo brings it back.`);
      break;
    case 'delete-vertex': deleteSelected(); break;
    case 'add-door': case 'add-window': {
      const wallIdx = sel.kind === 'vertex' ? sel.index : longestWall(target);
      const w = wallOf(target, wallIdx);
      const isDoor = action === 'add-door';
      const width = Math.min(isDoor ? 90 : 120, Math.floor(w.length));
      const op = {
        id: uid(), type: isDoor ? 'door' : 'window', roomId: target.id, wall: wallIdx,
        offset: snap((w.length - width) / 2, 5), width, height: isDoor ? 205 : 140, sill: isDoor ? 0 : 90,
        hinge: 'start', swing: 'in',
      };
      store.edit((p) => { p.openings.push(op); store.selection = { kind: 'opening', id: op.id }; });
      break;
    }
    case 'select-room': store.select({ kind: 'room', id: btn.dataset.room }); break;
    case 'split-wall': {
      const i = +btn.dataset.index;
      const w = wallOf(target, i);
      const mid = { x: snap((w.a.x + w.b.x) / 2, 0.5), y: snap((w.a.y + w.b.y) / 2, 0.5) };
      store.edit((p) => { insertVertex(p, target, i, mid); store.selection = { kind: 'vertex', id: target.id, index: i + 1 }; });
      toast(`Corner added to wall ${i + 1}. Drag it on the plan or type its position.`);
      break;
    }
  }
}

// interior angle at a corner, in degrees (over 180 for an inward-pointing corner)
function cornerAngle(room, i) {
  const n = room.points.length;
  const p = room.points[i], a = room.points[(i - 1 + n) % n], b = room.points[(i + 1) % n];
  const u = { x: a.x - p.x, y: a.y - p.y }, v = { x: b.x - p.x, y: b.y - p.y };
  const lu = Math.hypot(u.x, u.y), lv = Math.hypot(v.x, v.y);
  if (!lu || !lv) return 180;
  const ang = Math.acos(Math.max(-1, Math.min(1, (u.x * v.x + u.y * v.y) / (lu * lv)))) * 180 / Math.PI;
  const bis = { x: u.x / lu + v.x / lv, y: u.y / lu + v.y / lv };
  const bl = Math.hypot(bis.x, bis.y);
  if (bl < 1e-6) return 180;
  const probe = { x: p.x + bis.x / bl * 0.5, y: p.y + bis.y / bl * 0.5 };
  return pointInPolygon(probe, room.points) ? ang : 360 - ang;
}

function longestWall(room) {
  let best = 0, bestLen = 0;
  room.points.forEach((_, i) => { const l = wallOf(room, i).length; if (l > bestLen) { bestLen = l; best = i; } });
  return best;
}

function rotateSelected(deg) {
  const it = store.selection?.kind === 'item' ? store.selected : null;
  if (!it) return;
  store.edit(() => { it.rot = (((Math.round(it.rot / 90) * 90 + deg) % 360) + 360) % 360; });
}
function duplicateSelected() {
  const it = store.selection?.kind === 'item' ? store.selected : null;
  if (!it) return;
  const copy = { ...JSON.parse(JSON.stringify(it)), id: uid(), x: it.x + 20, y: it.y + 20 };
  store.edit((p) => { p.items.push(copy); store.selection = { kind: 'item', id: copy.id }; });
}
function deleteSelected() {
  const sel = store.selection;
  if (!sel) return;
  if (sel.kind === 'item') store.edit((p) => { p.items = p.items.filter((x) => x.id !== sel.id); store.selection = null; });
  else if (sel.kind === 'opening') store.edit((p) => { p.openings = p.openings.filter((x) => x.id !== sel.id); store.selection = null; });
  else if (sel.kind === 'vertex') {
    const room = store.selected;
    if (room.points.length <= 3) { toast('A room needs at least 3 corners.'); return; }
    store.edit((p) => { removeVertex(p, room, sel.index); store.selection = { kind: 'room', id: room.id }; });
  }
}

// ---------- tabs ----------
function roomsTabStatic() {
  $('#tab-rooms').innerHTML = `
    <h3>Rooms</h3>
    <ul class="list" id="rooms-list"></ul>
    <form id="add-room" class="fields">
      <label class="field wide"><span>New room name</span><input id="nr-name" value="Room" maxlength="40"></label>
      <label class="field"><span>Width</span><span class="unit" data-unit="cm"><input id="nr-w" type="number" value="400" min="20" inputmode="decimal"></span></label>
      <label class="field"><span>Depth</span><span class="unit" data-unit="cm"><input id="nr-d" type="number" value="300" min="20" inputmode="decimal"></span></label>
      <label class="field"><span>Ceiling</span><span class="unit" data-unit="cm"><input id="nr-h" type="number" value="250" min="50" inputmode="decimal"></span></label>
      <button class="btn primary wide" type="submit" style="grid-column: 1 / -1">Add room</button>
    </form>
    <h3>Irregular room</h3>
    <button class="btn primary" type="button" id="draw-room-btn">Draw a room corner by corner</button>
    <p class="muted small">Click each corner in turn. To place the next corner at an exact distance, point in its direction, type the length in cm and press Enter. Click the first corner (or press Enter) to close the shape.</p>
    <p class="muted small">Measure each room inside, wall to wall, at floor level. Drag rooms so their walls touch; corners snap together. To reshape any room, select it and tap the <b>+</b> in the middle of a wall.</p>`;
  $('#draw-room-btn').addEventListener('click', () => setDraw(true));
  $('#add-room').addEventListener('submit', (e) => {
    e.preventDefault();
    const w = parseFloat($('#nr-w').value), d = parseFloat($('#nr-d').value), h = parseFloat($('#nr-h').value);
    if (!(w > 0 && d > 0 && h > 0)) { toast('Enter a width, depth and ceiling height in cm.'); return; }
    const pts = store.plan.rooms.flatMap((r) => r.points);
    const origin = pts.length ? (() => { const b = bbox(pts); return { x: b.maxX, y: b.minY }; })() : { x: 0, y: 0 };
    const room = rectRoom($('#nr-name').value.trim() || 'Room', w, d, h, origin);
    store.edit((p) => { p.rooms.push(room); store.selection = { kind: 'room', id: room.id }; });
    plan2d.fit();
    $('#nr-name').value = 'Room';
  });
}

function furnitureTabStatic() {
  const groups = [...new Set(CATALOG.map((c) => c.group))];
  $('#tab-furniture').innerHTML = `
    <h3>In the plan</h3>
    <ul class="list" id="items-list"></ul>
    <h3>Add furniture</h3>
    <p class="muted small">Pick the closest type, then type in the exact size from the shop or product page.</p>
    <div class="catalog">${groups.map((g, gi) => `
      <details${gi === 0 ? ' open' : ''}><summary>${esc(g)}</summary><ul class="list">
        ${CATALOG.map((c, i) => c.group !== g ? '' : `<li><button data-preset="${i}"><span class="swatch" style="background:${c.color}"></span><span class="name">${esc(c.name)}</span><span class="meta">${c.w}×${c.d}×${c.h}</span></button></li>`).join('')}
      </ul></details>`).join('')}
    </div>`;
  $('#tab-furniture').addEventListener('click', (e) => {
    const b = e.target.closest('[data-preset]');
    if (b) return addPreset(CATALOG[+b.dataset.preset]);
    const s = e.target.closest('[data-select-item]');
    if (s) store.select({ kind: 'item', id: s.dataset.selectItem });
  });
}

function addPreset(preset) {
  let at = plan2d.center;
  if (!roomAt(store.plan, at) && store.plan.rooms.length) at = centroid(store.plan.rooms[0].points);
  const it = newItem(preset, { x: snap(at.x, 5), y: snap(at.y, 5) });
  store.edit((p) => { p.items.push(it); store.selection = { kind: 'item', id: it.id }; });
  document.getElementById('f-w')?.focus();
}

function renderLists() {
  const { plan, selection: sel } = store;
  $('#rooms-list').innerHTML = plan.rooms.map((r) => `
    <li><button data-select-room="${r.id}" class="${sel && sel.id === r.id ? 'active' : ''}">
      <span class="swatch" style="background: var(--floor-${r.floor || 'plain'})"></span>
      <span class="name">${esc(r.name)}</span>
      <span class="meta">${roomArea(r).toFixed(1)} m² · ${r.height}</span>
    </button></li>`).join('') || '<li class="muted small">No rooms yet. Add your first one below.</li>';

  const items = [...plan.items].sort((a, b) => (a.status === b.status ? a.name.localeCompare(b.name) : a.status === 'candidate' ? -1 : 1));
  $('#items-list').innerHTML = items.map((it) => {
    const lvl = it.hidden ? 'ok' : worstLevel(issues.get(it.id));
    return `<li><button data-select-item="${it.id}" class="${sel?.id === it.id ? 'active' : ''}">
      <span class="swatch ${it.status}" style="background:${esc(it.color)}"></span>
      <span class="name">${esc(it.name)}${it.hidden ? ' <span class="muted">(hidden)</span>' : ''}</span>
      <span class="meta">${it.w}×${it.d}×${it.h} <span class="dot ${lvl}" style="display:inline-block"></span></span>
    </button></li>`;
  }).join('') || '<li class="muted small">Nothing placed yet. Pick something below.</li>';

  renderShopping();
}

function renderShopping() {
  const cands = store.plan.items.filter((it) => it.status === 'candidate');
  if (!cands.length) {
    $('#tab-shopping').innerHTML = `<h3>Shopping list</h3><p class="muted small">Furniture you mark as “Considering” shows up here with whether it fits, its price and link. Add something from the Furniture tab to start.</p>`;
    return;
  }
  let total = 0, fitting = 0;
  const cards = cands.map((it) => {
    const list = issues.get(it.id) || [];
    const lvl = it.hidden ? 'info' : worstLevel(list);
    const p = priceOf(it);
    total += p;
    if (lvl !== 'error') fitting += p;
    const verdict = it.hidden ? 'Hidden from the plan'
      : lvl === 'ok' ? 'Fits' : lvl === 'error' ? `Doesn’t fit: ${list.find((i) => i.level === 'error').msg}`
      : lvl === 'warn' ? `Fits, but: ${list.find((i) => i.level === 'warn').msg}` : `Fits. ${list[0].msg}`;
    return `<div class="card ${lvl}">
      <div class="top"><button data-select-item="${it.id}">${esc(it.name)}</button><span class="price">${p ? money(p) : '–'}</span></div>
      <span class="meta muted small">${it.w} × ${it.d} × ${it.h} cm</span>
      <span class="verdict ${lvl === 'info' && it.hidden ? 'info' : lvl}">${esc(verdict)}</span>
      ${it.url ? `<a class="small" href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.url.replace(/^https?:\/\/(www\.)?/, '').slice(0, 60))}</a>` : ''}
      ${it.notes ? `<span class="small">${esc(it.notes)}</span>` : ''}
      <div class="row"><button class="btn" data-bought="${it.id}">Mark as bought</button></div>
    </div>`;
  }).join('');
  $('#tab-shopping').innerHTML = `<h3>Shopping list</h3>${cards}
    <div class="total"><span>Fits</span><span>${money(fitting)}</span></div>
    <div class="total"><span>Everything considered</span><span>${money(total)}</span></div>`;
}
$('#tab-shopping').addEventListener('click', (e) => {
  const s = e.target.closest('[data-select-item]');
  if (s) return store.select({ kind: 'item', id: s.dataset.selectItem });
  const b = e.target.closest('[data-bought]');
  if (b) {
    const it = store.plan.items.find((x) => x.id === b.dataset.bought);
    store.edit(() => { it.status = 'have'; });
    toast(`${it.name} moved to “I have this”.`);
  }
});
$('#tab-rooms').addEventListener('click', (e) => {
  const s = e.target.closest('[data-select-room]');
  if (s) store.select({ kind: 'room', id: s.dataset.selectRoom });
});

// ---------- keyboard ----------
document.addEventListener('keydown', (e) => {
  const typing = e.target.closest('input, textarea, select');
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === 'z' && !typing) { e.preventDefault(); e.shiftKey ? store.redo() : store.undo(); return; }
  if (mod && e.key.toLowerCase() === 'y' && !typing) { e.preventDefault(); store.redo(); return; }
  if (typing) return;
  if (plan2d.tool === 'draw' && !mod && plan2d.draftKey(e)) { e.preventDefault(); return; }
  if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSelected(); return; }
  if (mod) return;
  const it = store.selection?.kind === 'item' ? store.selected : null;
  switch (e.key) {
    case 'Escape':
      if (plan2d.tool === 'measure') setMeasure(false);
      else store.select(null);
      break;
    case 'Delete': case 'Backspace': e.preventDefault(); deleteSelected(); break;
    case 'r': case 'R': rotateSelected(e.shiftKey ? -90 : 90); break;
    case 'm': case 'M': setMeasure(plan2d.tool !== 'measure'); break;
    case 'n': case 'N': setDraw(plan2d.tool !== 'draw'); break;
    case 'ArrowUp': case 'ArrowDown': case 'ArrowLeft': case 'ArrowRight': {
      if (!it) return;
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
      const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0;
      store.edit(() => { it.x += dx; it.y += dy; });
      break;
    }
  }
});

// ---------- wiring ----------
store.subscribe(() => {
  issues = checkPlan(store.plan);
  const nameInput = $('#plan-name');
  if (document.activeElement !== nameInput) nameInput.value = store.plan.name;
  $('#undo').disabled = !store.undoStack.length;
  $('#redo').disabled = !store.redoStack.length;
  $('#banner').hidden = !store.plan.sample;
  renderInspector();
  renderLists();
});

roomsTabStatic();
furnitureTabStatic();
setTab(prefs.tab || 'furniture');
store.emit();
setView(prefs.view || 'plan');
requestAnimationFrame(() => plan2d.fit());

window.__roomfitStarted = true;

let fittedRemote = false;
if (boot.lock) {
  const lockBtn = $('#m-lock');
  lockBtn.hidden = false;
  lockBtn.addEventListener('click', boot.lock);
}
(boot.startSync || startSync)(store, setSync, {
  onRemote() {
    if (fittedRemote) return;
    fittedRemote = true;
    plan2d.fit();
    view3d?.frame();
  },
});
