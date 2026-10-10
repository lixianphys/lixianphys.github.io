// Plan data model, derived geometry, fit checks, persistence and undo.
import {
  add, sub, mul, norm, perp, rotate, len, centroid, pointInPolygon, polygonInside,
  convexOverlap, signedArea,
} from './geometry.js';

const STORAGE_KEY = 'roomfit:v1';
// Items this thin (rugs, mats) never collide with anything standing on them.
export const FLAT_HEIGHT = 3;

export const uid = () => Math.random().toString(36).slice(2, 10);

// A small sample flat so the first view shows what the tool does. `sample: true` shows a
// banner offering a blank start; updatedAt 0 means any synced plan wins over it.
export function defaultPlan() {
  const living = { ...rectRoom('Living room', 450, 380, 250, { x: 0, y: 0 }), floor: 'wood' };
  const bed = { ...rectRoom('Bedroom', 330, 380, 250, { x: 450, y: 0 }), floor: 'carpet' };
  const hall = { ...rectRoom('Hallway', 220, 140, 250, { x: 230, y: 380 }), floor: 'tile' };
  const item = (preset, x, y, rot, status) => ({ ...newItem(preset, { x, y }), rot, status });
  return {
    version: 1,
    name: 'Sample flat',
    sample: true,
    updatedAt: 0,
    rooms: [living, bed, hall],
    openings: [
      { id: uid(), type: 'door', roomId: living.id, wall: 1, offset: 250, width: 85, height: 205, sill: 0, hinge: 'end', swing: 'in' },
      { id: uid(), type: 'door', roomId: hall.id, wall: 2, offset: 70, width: 90, height: 205, sill: 0, hinge: 'start', swing: 'in' },
      { id: uid(), type: 'door', roomId: living.id, wall: 2, offset: 0, width: 220, height: 205, sill: 0, hinge: 'start', swing: 'none' },
      { id: uid(), type: 'window', roomId: living.id, wall: 0, offset: 150, width: 160, height: 140, sill: 85 },
      { id: uid(), type: 'window', roomId: bed.id, wall: 0, offset: 100, width: 120, height: 140, sill: 85 },
    ],
    items: [
      item({ name: 'Sofa 3-seat', w: 210, d: 90, h: 85, color: '#7c8ea3' }, 145, 45, 0, 'have'),
      item({ name: 'Coffee table', w: 110, d: 60, h: 42, color: '#b08b62' }, 145, 150, 0, 'have'),
      item({ name: 'Rug', w: 200, d: 140, h: 1, color: '#d9c7a7' }, 145, 140, 0, 'have'),
      item({ name: 'TV bench', w: 180, d: 40, h: 50, color: '#5d5d66' }, 145, 360, 180, 'have'),
      item({ name: 'Bed 160×200', w: 170, d: 212, h: 45, color: '#a7b6a0' }, 615, 106, 0, 'have'),
      item({ name: 'Wardrobe 3-door', w: 150, d: 60, h: 236, color: '#e2ddd3' }, 525, 350, 180, 'candidate'),
      item({ name: 'Bookshelf', w: 80, d: 30, h: 200, color: '#c9a77c' }, 435, 285, 90, 'candidate'),
    ],
  };
}

export function blankPlan() {
  const room = rectRoom('Room 1', 400, 300, 250);
  return { version: 1, name: 'My apartment', sample: false, updatedAt: Date.now(), rooms: [room], openings: [], items: [] };
}

export function rectRoom(name, w, d, height, origin = { x: 0, y: 0 }) {
  const { x, y } = origin;
  return {
    id: uid(), name, height, floor: 'plain',
    points: [{ x, y }, { x: x + w, y }, { x: x + w, y: y + d }, { x, y: y + d }],
  };
}

export function newItem(preset, at) {
  return {
    id: uid(),
    name: preset.name,
    shape: preset.shape || 'rect',
    w: preset.w, d: preset.d, h: preset.h,
    lw: preset.lw ?? Math.round(preset.w / 3),
    ld: preset.ld ?? Math.round(preset.d / 2),
    mirror: false,
    elevation: preset.elevation || 0,
    x: at.x, y: at.y, rot: 0,
    color: preset.color || '#b9a0c9',
    status: 'candidate',
    hidden: false,
    url: '', price: '', notes: '',
  };
}

// ---------- derived geometry ----------

function localOutline(it) {
  const hw = it.w / 2, hd = it.d / 2;
  if (it.shape === 'ellipse') {
    const n = 32;
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2;
      return { x: Math.cos(a) * hw, y: Math.sin(a) * hd };
    });
  }
  if (it.shape === 'L') {
    const lw = Math.min(it.lw, it.w), ld = Math.min(it.ld, it.d);
    const pts = [
      { x: -hw, y: -hd }, { x: hw, y: -hd }, { x: hw, y: hd },
      { x: hw - lw, y: hd }, { x: hw - lw, y: -hd + ld }, { x: -hw, y: -hd + ld },
    ];
    return it.mirror ? pts.map((p) => ({ x: -p.x, y: p.y })).reverse() : pts;
  }
  return [{ x: -hw, y: -hd }, { x: hw, y: -hd }, { x: hw, y: hd }, { x: -hw, y: hd }];
}

function localParts(it) {
  if (it.shape !== 'L') return [localOutline(it)];
  const hw = it.w / 2, hd = it.d / 2;
  const lw = Math.min(it.lw, it.w), ld = Math.min(it.ld, it.d);
  const rect = (x0, y0, x1, y1) => [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
  const parts = [rect(-hw, -hd, hw, -hd + ld), rect(hw - lw, -hd, hw, hd)];
  return it.mirror ? parts.map((p) => p.map((q) => ({ x: -q.x, y: q.y })).reverse()) : parts;
}

const toWorld = (it) => (p) => add(rotate(p, it.rot), { x: it.x, y: it.y });

export const itemOutline = (it) => localOutline(it).map(toWorld(it));
export const itemParts = (it) => localParts(it).map((part) => part.map(toWorld(it)));

export function wallOf(room, i) {
  const a = room.points[i], b = room.points[(i + 1) % room.points.length];
  const dir = norm(sub(b, a));
  let n = perp(dir);
  const mid = mul(add(a, b), 0.5);
  if (!pointInPolygon(add(mid, mul(n, 1)), room.points)) n = mul(n, -1);
  return { a, b, dir, length: len(sub(b, a)), inward: n };
}

export function openingGeom(op, room) {
  const wall = wallOf(room, op.wall % room.points.length);
  const offset = Math.max(0, Math.min(op.offset, wall.length - op.width));
  const start = add(wall.a, mul(wall.dir, offset));
  const end = add(start, mul(wall.dir, Math.min(op.width, wall.length)));
  const g = { wall, offset, start, end };
  if (op.type === 'door' && op.swing !== 'none') {
    const hinge = op.hinge === 'end' ? end : start;
    const free = op.hinge === 'end' ? start : end;
    const side = op.swing === 'out' ? mul(wall.inward, -1) : wall.inward;
    const open = add(hinge, mul(side, op.width));
    g.hinge = hinge; g.free = free; g.open = open;
    // conservative clearance zone: the square the leaf sweeps through
    g.zone = [hinge, free, add(free, mul(side, op.width)), open];
  } else {
    // keep-clear area in front of a window (only matters for things taller than the sill)
    const depth = 30;
    g.zone = [start, end, add(end, mul(wall.inward, depth)), add(start, mul(wall.inward, depth))];
  }
  return g;
}

export function roomArea(room) {
  return Math.abs(signedArea(room.points)) / 10000; // m²
}

export function roomAt(plan, pt) {
  return plan.rooms.find((r) => pointInPolygon(pt, r.points));
}

// ---------- fit checks ----------
// Returns Map<itemId, Array<{level: 'error'|'warn'|'info', msg}>>

export function checkPlan(plan) {
  const issues = new Map();
  const items = plan.items.filter((it) => !it.hidden);
  const push = (id, level, msg) => {
    if (!issues.has(id)) issues.set(id, []);
    issues.get(id).push({ level, msg });
  };

  const parts = new Map(items.map((it) => [it.id, itemParts(it)]));
  const outlines = new Map(items.map((it) => [it.id, itemOutline(it)]));

  for (const it of items) {
    const outline = outlines.get(it.id);
    const room = roomAt(plan, { x: it.x, y: it.y });
    if (!room) {
      push(it.id, 'error', 'Not inside any room');
    } else {
      if (!polygonInside(outline, room.points)) push(it.id, 'error', `Goes through a wall of ${room.name}`);
      const top = it.elevation + it.h;
      if (top > room.height) push(it.id, 'error', `Too tall: ${top} cm vs ceiling ${room.height} cm`);
    }
  }

  const vOverlap = (a, b) => a.elevation < b.elevation + b.h && b.elevation < a.elevation + a.h;
  const collides = (pa, pb) => pa.some((x) => pb.some((y) => convexOverlap(x, y)));

  for (let i = 0; i < items.length; i++) {
    const a = items[i];
    if (a.h <= FLAT_HEIGHT) continue;
    for (let j = i + 1; j < items.length; j++) {
      const b = items[j];
      if (b.h <= FLAT_HEIGHT || !vOverlap(a, b)) continue;
      if (collides(parts.get(a.id), parts.get(b.id))) {
        push(a.id, 'error', `Overlaps ${b.name}`);
        push(b.id, 'error', `Overlaps ${a.name}`);
      }
    }
  }

  for (const op of plan.openings) {
    const room = plan.rooms.find((r) => r.id === op.roomId);
    if (!room) continue;
    const g = openingGeom(op, room);
    for (const it of items) {
      if (it.h <= FLAT_HEIGHT) continue;
      if (!parts.get(it.id).some((p) => convexOverlap(p, g.zone))) continue;
      if (op.type === 'door') {
        if (op.swing !== 'none' && it.elevation < op.height) push(it.id, 'warn', `Blocks door swing (${room.name})`);
      } else if (it.elevation + it.h > op.sill && it.elevation < op.sill + op.height) {
        push(it.id, 'info', `In front of a window (sill ${op.sill} cm)`);
      }
    }
  }
  return issues;
}

export function worstLevel(list) {
  if (!list || !list.length) return 'ok';
  if (list.some((i) => i.level === 'error')) return 'error';
  if (list.some((i) => i.level === 'warn')) return 'warn';
  return 'info';
}

// ---------- room editing helpers ----------

// Change wall i's length by moving every vertex "beyond" the wall's end along its direction.
// Keeps rectangles rectangular and works well for L/U shaped rooms with square corners.
export function setWallLength(room, i, newLen) {
  const w = wallOf(room, i);
  const delta = newLen - w.length;
  if (Math.abs(delta) < 1e-9 || w.length < 1e-9) return;
  const endProj = (w.b.x - w.a.x) * w.dir.x + (w.b.y - w.a.y) * w.dir.y;
  room.points = room.points.map((p) => {
    const proj = (p.x - w.a.x) * w.dir.x + (p.y - w.a.y) * w.dir.y;
    return proj >= endProj - 0.5 ? add(p, mul(w.dir, delta)) : p;
  });
}

export function insertVertex(plan, room, wallIndex, pt) {
  const w = wallOf(room, wallIndex);
  const cut = Math.hypot(pt.x - w.a.x, pt.y - w.a.y);
  room.points.splice(wallIndex + 1, 0, pt);
  for (const op of plan.openings) {
    if (op.roomId !== room.id) continue;
    if (op.wall > wallIndex) op.wall++;
    else if (op.wall === wallIndex && op.offset >= cut) { op.wall++; op.offset -= cut; }
  }
}

export function removeVertex(plan, room, idx) {
  if (room.points.length <= 3) return false;
  room.points.splice(idx, 1);
  const n = room.points.length;
  // walls idx-1 and idx merge into wall idx-1
  const merged = (idx - 1 + n + 1) % (n + 1);
  plan.openings = plan.openings.filter((op) => op.roomId !== room.id || (op.wall !== merged && op.wall !== idx));
  for (const op of plan.openings) if (op.roomId === room.id && op.wall > idx) op.wall--;
  return true;
}

export const FLOORS = { plain: 'Plain', wood: 'Wood', tile: 'Tiles', carpet: 'Carpet' };

export function roomCenter(room) {
  return centroid(room.points);
}

// ---------- store ----------

const localStore = {
  load: () => JSON.parse(localStorage.getItem(STORAGE_KEY)),
  save: (plan) => localStorage.setItem(STORAGE_KEY, JSON.stringify(plan)),
};

export class Store {
  // `storage` is {load(), save(plan)}; the GitHub Pages build passes an encrypted one.
  constructor(storage = localStore) {
    this.storage = storage;
    this.plan = this.load() || defaultPlan();
    this.selection = null; // {kind: 'item'|'room'|'opening'|'vertex', id, index?}
    this.undoStack = [];
    this.redoStack = [];
    this.listeners = new Set();
    this.dragging = false;   // set by the plan editor while a drag is in progress
    this.onSave = null;      // sync hook: called after every local change is persisted
  }

  load() {
    try {
      const plan = this.storage.load();
      return plan ? migrate(plan) : null;
    } catch {
      return null;
    }
  }

  save() {
    this.plan.updatedAt = Date.now();
    this.saveLocal();
    this.onSave?.();
  }

  saveLocal() {
    try { this.storage.save(this.plan); } catch { /* storage unavailable */ }
  }

  subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(this); }

  checkpoint() {
    this.undoStack.push(JSON.stringify(this.plan));
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
  }
  dropCheckpoint() { this.undoStack.pop(); }

  // Undoable edit: snapshot, mutate, persist, notify.
  edit(fn) {
    this.checkpoint();
    fn(this.plan);
    this.save();
    this.emit();
  }
  // Live change during a drag (caller already took a checkpoint).
  change() { this.emit(); }
  commit() { this.save(); this.emit(); }

  undo() {
    if (!this.undoStack.length) return;
    this.redoStack.push(JSON.stringify(this.plan));
    this.plan = JSON.parse(this.undoStack.pop());
    this.validateSelection();
    this.commit();
  }
  redo() {
    if (!this.redoStack.length) return;
    this.undoStack.push(JSON.stringify(this.plan));
    this.plan = JSON.parse(this.redoStack.pop());
    this.validateSelection();
    this.commit();
  }

  replace(plan) {
    this.checkpoint();
    this.plan = migrate(plan);
    this.selection = null;
    this.commit();
  }

  // A newer plan arrived from another device. Undoable, and not pushed back.
  applyRemote(plan) {
    this.checkpoint();
    this.plan = migrate(plan);
    this.validateSelection();
    this.saveLocal();
    this.emit();
  }

  select(sel) { this.selection = sel; this.emit(); }

  validateSelection() {
    const s = this.selection;
    if (!s) return;
    const list = { item: this.plan.items, room: this.plan.rooms, vertex: this.plan.rooms, opening: this.plan.openings }[s.kind];
    if (!list.some((x) => x.id === s.id)) this.selection = null;
  }

  get selected() {
    const s = this.selection;
    if (!s) return null;
    if (s.kind === 'item') return this.plan.items.find((x) => x.id === s.id);
    if (s.kind === 'opening') return this.plan.openings.find((x) => x.id === s.id);
    return this.plan.rooms.find((x) => x.id === s.id);
  }
}

function migrate(plan) {
  if (!plan || !Array.isArray(plan.rooms) || !Array.isArray(plan.items)) throw new Error('Not a RoomFit plan');
  if (!plan.openings) plan.openings = [];
  if (!plan.name) plan.name = 'My apartment';
  for (const r of plan.rooms) if (!r.floor) r.floor = 'plain';
  for (const it of plan.items) {
    if (!it.shape) it.shape = 'rect';
    if (it.elevation == null) it.elevation = 0;
    if (it.lw == null) it.lw = Math.round(it.w / 3);
    if (it.ld == null) it.ld = Math.round(it.d / 2);
    if (!it.status) it.status = 'candidate';
  }
  return plan;
}
