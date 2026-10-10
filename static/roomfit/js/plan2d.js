// Top-down SVG plan editor: pan/zoom, drag items/rooms/vertices/openings, measure tool,
// live clearance distances for the selected item.
import {
  add, sub, mul, dist, norm, rotate, snap, closestPointOnSegment, raySegment, edges, bbox, centroid, len,
} from './geometry.js?v=d711b78469';
import {
  itemOutline, wallOf, openingGeom, checkPlan, worstLevel, insertVertex, roomArea, FLAT_HEIGHT,
  partitionOutline, partitionLength, partitionCeiling,
} from './model.js?v=d711b78469';

const NS = 'http://www.w3.org/2000/svg';
const SNAP_WALL = 8; // cm: items closer than this to a wall snap flush against it

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const ptsAttr = (pts) => pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

export class Plan2D {
  constructor(container, store, { onEditWall, onRoomDrawn, onWallDrawn, onToolChange } = {}) {
    this.store = store;
    this.onWallDrawn = onWallDrawn;
    this.onEditWall = onEditWall;
    this.onRoomDrawn = onRoomDrawn;
    this.onToolChange = onToolChange;
    this.svg = el('svg', { class: 'plan' }, container);
    this.view = { x: -60, y: -60, scale: 1.5 }; // scale = px per cm
    this.grid = 5;
    this.snapOn = true;
    this.tool = 'select'; // select | measure | draw (a room) | wall (an inner wall)
    this.measure = null;  // {a, b, done}
    this.draft = null;    // room being drawn: {pts, cursor, typed}
    this.drag = null;
    this.pointers = new Map();
    this.showCandidates = true;
    this.locked = false;  // layout locked: only furniture can be moved

    this.svg.addEventListener('pointerdown', (e) => this.onDown(e));
    this.svg.addEventListener('pointermove', (e) => this.onMove(e));
    this.svg.addEventListener('pointerup', (e) => this.onUp(e));
    this.svg.addEventListener('pointercancel', (e) => this.onUp(e));
    this.svg.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    this.svg.addEventListener('dblclick', (e) => this.onDblClick(e));
    this.userMoved = false; // until the user pans/zooms, keep the whole plan fitted to the pane
    new ResizeObserver(() => (this.userMoved ? this.render() : this.fit())).observe(this.svg);
    store.subscribe(() => this.render());
  }

  // ---------- coordinates ----------
  toPlan(e) {
    const r = this.svg.getBoundingClientRect();
    return { x: this.view.x + (e.clientX - r.left) / this.view.scale, y: this.view.y + (e.clientY - r.top) / this.view.scale };
  }
  get size() {
    const r = this.svg.getBoundingClientRect();
    return { w: r.width, h: r.height };
  }
  get center() {
    const { w, h } = this.size;
    return { x: this.view.x + w / 2 / this.view.scale, y: this.view.y + h / 2 / this.view.scale };
  }
  px(n) { return n / this.view.scale; } // screen pixels → cm (for constant-size strokes/labels)

  zoomAt(screenPt, factor) {
    this.userMoved = true;
    const r = this.svg.getBoundingClientRect();
    const sx = screenPt.x - r.left, sy = screenPt.y - r.top;
    const before = { x: this.view.x + sx / this.view.scale, y: this.view.y + sy / this.view.scale };
    this.view.scale = Math.max(0.15, Math.min(12, this.view.scale * factor));
    this.view.x = before.x - sx / this.view.scale;
    this.view.y = before.y - sy / this.view.scale;
    this.render();
  }

  fit() {
    const pts = this.store.plan.rooms.flatMap((r) => r.points);
    if (!pts.length) return;
    const b = bbox(pts);
    const { w, h } = this.size;
    if (!w || !h) return;
    const pad = 60;
    const scale = Math.min(w / (b.maxX - b.minX + pad * 2), h / (b.maxY - b.minY + pad * 2));
    this.view.scale = Math.max(0.15, Math.min(12, scale));
    this.view.x = (b.minX + b.maxX) / 2 - w / 2 / this.view.scale;
    this.view.y = (b.minY + b.maxY) / 2 - h / 2 / this.view.scale;
    this.render();
  }

  setTool(tool) {
    this.tool = tool;
    this.measure = null;
    this.draft = tool === 'draw' || tool === 'wall' ? { pts: [], cursor: null, typed: '' } : null;
    this.svg.classList.toggle('measuring', tool !== 'select');
    this.onToolChange?.(tool);
    this.render();
  }

  // ---------- rendering ----------
  render() {
    const { w, h } = this.size;
    if (!w || !h) return;
    const { plan, selection: sel } = this.store;
    const v = this.view;
    this.svg.setAttribute('viewBox', `${v.x} ${v.y} ${w / v.scale} ${h / v.scale}`);
    this.svg.classList.toggle('locked', this.locked);
    this.svg.textContent = '';
    this.issues = checkPlan(plan);

    this.renderGrid(w, h);

    const gRooms = el('g', {}, this.svg);
    for (const room of plan.rooms) {
      const selected = sel && (sel.kind === 'room' || sel.kind === 'vertex') && sel.id === room.id;
      el('polygon', {
        points: ptsAttr(room.points), class: 'room' + (selected ? ' selected' : ''),
        style: `fill: var(--floor-${room.floor || 'plain'})`, 'stroke-width': this.px(selected ? 4 : 3), 'data-room': room.id,
      }, gRooms);
      const c = centroid(room.points);
      const t = el('text', { x: c.x, y: c.y, class: 'room-label', 'font-size': this.px(13) }, gRooms);
      t.textContent = room.name;
      const t2 = el('text', { x: c.x, y: c.y + this.px(15), class: 'room-sub', 'font-size': this.px(11) }, gRooms);
      t2.textContent = `${roomArea(room).toFixed(1)} m² · ${room.height} cm high`;
    }

    // openings
    const gOpen = el('g', {}, this.svg);
    for (const op of plan.openings) {
      const room = plan.rooms.find((r) => r.id === op.roomId);
      if (!room) continue;
      const g = openingGeom(op, room);
      const selected = sel?.kind === 'opening' && sel.id === op.id;
      const grp = el('g', { class: 'opening ' + op.type + (selected ? ' selected' : ''), 'data-opening': op.id }, gOpen);
      el('line', { x1: g.start.x, y1: g.start.y, x2: g.end.x, y2: g.end.y, class: 'gap', 'stroke-width': this.px(7) }, grp);
      if (op.type === 'door' && op.swing !== 'none') {
        const r = op.width;
        const sweep = cross2(sub(g.free, g.hinge), sub(g.open, g.hinge)) > 0 ? 1 : 0;
        el('path', {
          d: `M ${g.open.x} ${g.open.y} A ${r} ${r} 0 0 ${1 - sweep} ${g.free.x} ${g.free.y}`,
          class: 'swing', 'stroke-width': this.px(1.2),
        }, grp);
        el('line', { x1: g.hinge.x, y1: g.hinge.y, x2: g.open.x, y2: g.open.y, class: 'leaf', 'stroke-width': this.px(2.5) }, grp);
      } else if (op.type === 'window') {
        el('line', { x1: g.start.x, y1: g.start.y, x2: g.end.x, y2: g.end.y, class: 'glass', 'stroke-width': this.px(2) }, grp);
      }
      // fat invisible hit target
      el('line', { x1: g.start.x, y1: g.start.y, x2: g.end.x, y2: g.end.y, class: 'hit', 'stroke-width': this.px(16) }, grp);
    }

    // inner walls: solid when full height, hatched when lower than the ceiling
    const gPart = el('g', {}, this.svg);
    for (const pw of plan.partitions) {
      const selected = sel?.kind === 'partition' && sel.id === pw.id;
      const half = pw.height < partitionCeiling(plan, pw);
      el('polygon', {
        points: ptsAttr(partitionOutline(pw)), class: 'partition' + (half ? ' half' : '') + (selected ? ' selected' : ''),
        'stroke-width': this.px(selected ? 2.5 : 1.2), 'data-partition': pw.id,
      }, gPart);
      if (half && !selected) {
        const m = mul(add(pw.a, pw.b), 0.5);
        const fs = this.px(10);
        if (partitionLength(pw) * this.view.scale > 70) {
          const t = el('text', { x: m.x, y: m.y, class: 'partition-label', 'font-size': fs, 'data-partition': pw.id }, gPart);
          t.textContent = `h ${pw.height}`;
        }
      }
    }

    // wall length labels for the selected room
    const selRoom = sel && (sel.kind === 'room' || sel.kind === 'vertex') ? plan.rooms.find((r) => r.id === sel.id) : null;
    if (selRoom && !this.locked) this.renderWallLabels(selRoom, sel);

    // items (flat ones first so they sit underneath)
    const gItems = el('g', {}, this.svg);
    const ordered = [...plan.items].sort((a, b) => (a.h <= FLAT_HEIGHT ? 0 : 1) - (b.h <= FLAT_HEIGHT ? 0 : 1));
    for (const it of ordered) {
      if (it.hidden || (!this.showCandidates && it.status === 'candidate')) continue;
      const selected = sel?.kind === 'item' && sel.id === it.id;
      const level = worstLevel(this.issues.get(it.id));
      const outline = itemOutline(it);
      const grp = el('g', { class: `item ${it.status} lvl-${level}` + (selected ? ' selected' : ''), 'data-item': it.id }, gItems);
      el('polygon', { points: ptsAttr(outline), fill: it.color, 'stroke-width': this.px(selected ? 2.5 : 1.5) }, grp);
      // front marker (the +y side in local coordinates is the "front")
      const f1 = add(rotate({ x: -it.w / 2 * 0.6, y: it.d / 2 }, it.rot), it);
      const f2 = add(rotate({ x: it.w / 2 * 0.6, y: it.d / 2 }, it.rot), it);
      if (it.shape === 'rect') el('line', { x1: f1.x, y1: f1.y, x2: f2.x, y2: f2.y, class: 'front', 'stroke-width': this.px(3) }, grp);
      const fs = Math.min(this.px(12), Math.min(it.w, it.d) / 3.2);
      const label = el('text', { x: it.x, y: it.y - fs * 0.15, class: 'item-label', 'font-size': fs }, grp);
      label.textContent = it.name;
      const sub2 = el('text', { x: it.x, y: it.y + fs * 1.05, class: 'item-sub', 'font-size': fs * 0.8 }, grp);
      sub2.textContent = `${it.w}×${it.d}×${it.h}`;
    }

    if (sel?.kind === 'item') {
      const it = plan.items.find((x) => x.id === sel.id);
      if (it && !it.hidden) this.renderClearances(it);
    }
    if (sel?.kind === 'partition') {
      const pw = plan.partitions.find((x) => x.id === sel.id);
      if (pw && !this.locked) this.renderPartitionHandles(pw);
    }
    this.renderMeasure();
    this.renderDraft();
  }

  renderGrid(w, h) {
    const v = this.view;
    const step = v.scale > 2.5 ? 10 : v.scale > 0.6 ? 50 : 100;
    const major = step * 10 >= 1000 ? 500 : 100;
    const x0 = Math.floor(v.x / step) * step, y0 = Math.floor(v.y / step) * step;
    const x1 = v.x + w / v.scale, y1 = v.y + h / v.scale;
    let minor = '', maj = '';
    for (let x = x0; x <= x1; x += step) (x % major === 0 ? (maj += `M${x} ${v.y}V${y1}`) : (minor += `M${x} ${v.y}V${y1}`));
    for (let y = y0; y <= y1; y += step) (y % major === 0 ? (maj += `M${v.x} ${y}H${x1}`) : (minor += `M${v.x} ${y}H${x1}`));
    el('path', { d: minor, class: 'grid-minor', 'stroke-width': this.px(1) }, this.svg);
    el('path', { d: maj, class: 'grid-major', 'stroke-width': this.px(1) }, this.svg);
  }

  renderWallLabels(room, sel) {
    const g = el('g', {}, this.svg);
    room.points.forEach((_, i) => {
      const w = wallOf(room, i);
      const mid = mul(add(w.a, w.b), 0.5);
      const txt = `${Math.round(w.length * 10) / 10} cm`;
      const bw = this.px(8 + txt.length * 6.4), bh = this.px(17);
      // push the label outside the wall far enough to clear the "+" handle whatever the wall's angle
      const off = this.px(12) + Math.abs(w.inward.x) * bw / 2 + Math.abs(w.inward.y) * bh / 2;
      const pos = add(mid, mul(w.inward, -off));
      const grp = el('g', { class: 'wall-label', 'data-wall': i, 'data-room': room.id }, g);
      el('rect', { x: pos.x - bw / 2, y: pos.y - bh / 2, width: bw, height: bh, rx: this.px(4) }, grp);
      const t = el('text', { x: pos.x, y: pos.y, 'font-size': this.px(11) }, grp);
      t.textContent = txt;
    });
    // "+" handles mid-wall: tap to add a corner there, drag to pull it into shape
    room.points.forEach((_, i) => {
      const w = wallOf(room, i);
      if (w.length * this.view.scale < 44) return;
      const m = mul(add(w.a, w.b), 0.5);
      const grp = el('g', { class: 'midpoint', 'data-mid': i, 'data-room': room.id }, g);
      el('circle', { cx: m.x, cy: m.y, r: this.px(7), 'stroke-width': this.px(1.5) }, grp);
      const k = this.px(3.5);
      el('path', { d: `M${m.x - k} ${m.y}H${m.x + k}M${m.x} ${m.y - k}V${m.y + k}`, 'stroke-width': this.px(1.6) }, grp);
      el('title', {}, grp).textContent = 'Add a corner here';
    });
    room.points.forEach((p, i) => {
      const active = sel.kind === 'vertex' && sel.index === i;
      el('circle', { cx: p.x, cy: p.y, r: this.px(active ? 8 : 6), class: 'vertex' + (active ? ' selected' : ''), 'data-vertex': i, 'data-room': room.id, 'stroke-width': this.px(2) }, g);
    });
  }

  renderPartitionHandles(pw) {
    const g = el('g', {}, this.svg);
    const d = sub(pw.b, pw.a), L = len(d);
    if (L > 1) {
      const n = { x: -d.y / L, y: d.x / L };
      const txt = `${Math.round(L * 10) / 10} cm`;
      const bw = this.px(8 + txt.length * 6.4), bh = this.px(17);
      const off = pw.thickness / 2 + this.px(10) + Math.abs(n.x) * bw / 2 + Math.abs(n.y) * bh / 2;
      const pos = add(mul(add(pw.a, pw.b), 0.5), mul(n, -off));
      const grp = el('g', { class: 'wall-label static' }, g);
      el('rect', { x: pos.x - bw / 2, y: pos.y - bh / 2, width: bw, height: bh, rx: this.px(4) }, grp);
      el('text', { x: pos.x, y: pos.y, 'font-size': this.px(11) }, grp).textContent = txt;
    }
    [pw.a, pw.b].forEach((p, i) => {
      el('circle', { cx: p.x, cy: p.y, r: this.px(7), class: 'vertex', 'data-pend': i, 'data-partition': pw.id, 'stroke-width': this.px(2) }, g);
    });
  }

  // Distance from each side of the item to the nearest wall/item, drawn as dimension lines.
  clearances(it) {
    const { plan } = this.store;
    const segs = [];
    for (const r of plan.rooms) segs.push(...edges(r.points));
    for (const pw of plan.partitions) {
      if (it.h > FLAT_HEIGHT && it.elevation < pw.height && partitionLength(pw) >= 1) segs.push(...edges(partitionOutline(pw)));
    }
    for (const o of plan.items) {
      if (o.id === it.id || o.hidden || o.h <= FLAT_HEIGHT || it.h <= FLAT_HEIGHT) continue;
      if (!(o.elevation < it.elevation + it.h && it.elevation < o.elevation + o.h)) continue;
      segs.push(...edges(itemOutline(o)));
    }
    const out = [];
    const sides = [
      { o: { x: 0, y: -it.d / 2 }, d: { x: 0, y: -1 } },
      { o: { x: it.w / 2, y: 0 }, d: { x: 1, y: 0 } },
      { o: { x: 0, y: it.d / 2 }, d: { x: 0, y: 1 } },
      { o: { x: -it.w / 2, y: 0 }, d: { x: -1, y: 0 } },
    ];
    for (const s of sides) {
      const o = add(rotate(s.o, it.rot), it);
      const d = rotate(s.d, it.rot);
      let best = Infinity;
      for (const [a, b] of segs) best = Math.min(best, raySegment(add(o, mul(d, 0.01)), d, a, b));
      if (best < 2000) out.push({ from: o, dir: d, dist: best });
    }
    return out;
  }

  renderClearances(it) {
    const g = el('g', { class: 'clearance' }, this.svg);
    for (const c of this.clearances(it)) {
      if (c.dist < 0.5) continue;
      const to = add(c.from, mul(c.dir, c.dist));
      el('line', { x1: c.from.x, y1: c.from.y, x2: to.x, y2: to.y, 'stroke-width': this.px(1.2) }, g);
      const mid = mul(add(c.from, to), 0.5);
      const txt = `${Math.round(c.dist)}`;
      const bw = this.px(8 + txt.length * 7), bh = this.px(16);
      el('rect', { x: mid.x - bw / 2, y: mid.y - bh / 2, width: bw, height: bh, rx: this.px(3) }, g);
      const t = el('text', { x: mid.x, y: mid.y, 'font-size': this.px(11) }, g);
      t.textContent = txt;
    }
  }

  renderMeasure() {
    const m = this.measure;
    if (!m || !m.b) return;
    const g = el('g', { class: 'measure' }, this.svg);
    el('line', { x1: m.a.x, y1: m.a.y, x2: m.b.x, y2: m.b.y, 'stroke-width': this.px(2) }, g);
    for (const p of [m.a, m.b]) el('circle', { cx: p.x, cy: p.y, r: this.px(4) }, g);
    const mid = mul(add(m.a, m.b), 0.5);
    const txt = `${dist(m.a, m.b).toFixed(1)} cm`;
    const bw = this.px(10 + txt.length * 7), bh = this.px(18);
    el('rect', { x: mid.x - bw / 2, y: mid.y - bh - this.px(4), width: bw, height: bh, rx: this.px(4) }, g);
    const t = el('text', { x: mid.x, y: mid.y - bh / 2 - this.px(4), 'font-size': this.px(12) }, g);
    t.textContent = txt;
  }

  renderDraft() {
    const d = this.draft;
    if (!d) return;
    const g = el('g', { class: 'draft' }, this.svg);
    const next = this.draftNext();
    const pts = next ? [...d.pts, next] : d.pts;
    if (pts.length >= 2) {
      el('polyline', { points: ptsAttr(pts), 'stroke-width': this.px(2.5) }, g);
      pts.slice(1).forEach((q, i) => {
        const a = pts[i];
        const L = dist(a, q);
        if (L < 1) return;
        const mid = mul(add(a, q), 0.5);
        const live = i === pts.length - 2 && next;
        const txt = live && d.typed ? `${d.typed}▏cm` : `${Math.round(L * 10) / 10} cm`;
        const bw = this.px(10 + txt.length * 6.6), bh = this.px(17);
        el('rect', { x: mid.x - bw / 2, y: mid.y - bh / 2, width: bw, height: bh, rx: this.px(4), class: live ? 'live' : '' }, g);
        el('text', { x: mid.x, y: mid.y, 'font-size': this.px(11) }, g).textContent = txt;
      });
    }
    d.pts.forEach((q, i) => {
      const closable = i === 0 && d.pts.length >= 3 && this.tool === 'draw';
      el('circle', { cx: q.x, cy: q.y, r: this.px(closable ? 8 : 5), class: closable ? 'first' : '', 'stroke-width': this.px(2) }, g);
    });
  }

  // where the next corner would go: typed length along the cursor direction, else the cursor
  draftNext() {
    const d = this.draft;
    if (!d?.cursor) return null;
    const last = d.pts[d.pts.length - 1];
    const L = parseFloat(d.typed);
    if (last && L > 0) {
      const dir = norm(sub(d.cursor, last));
      if (len(dir) > 0) return add(last, mul(dir, L));
    }
    return d.cursor;
  }

  draftPoint(p) {
    const d = this.draft;
    const r = this.px(12);
    if (d.pts.length >= 3 && dist(p, d.pts[0]) < r) return { ...d.pts[0] };
    const { plan } = this.store;
    for (const room of plan.rooms) for (const q of room.points) if (dist(p, q) < r) return { ...q };
    for (const pw of plan.partitions) for (const q of [pw.a, pw.b]) if (dist(p, q) < r) return { ...q };
    let out = this.snapPt(p);
    const last = d.pts[d.pts.length - 1];
    if (last) { // keep walls straight when close to horizontal/vertical
      if (Math.abs(out.x - last.x) < this.px(10)) out.x = last.x;
      if (Math.abs(out.y - last.y) < this.px(10)) out.y = last.y;
    }
    // an inner wall starts or ends flush against a wall
    if (this.tool === 'wall') out = this.onWallLine(out, r);
    return out;
  }

  // the closest point on any room wall or inner wall within r, else p
  onWallLine(p, r, skipId) {
    const { plan } = this.store;
    let best = null;
    const segs = plan.rooms.flatMap((room) => edges(room.points));
    for (const pw of plan.partitions) if (pw.id !== skipId) segs.push([pw.a, pw.b]);
    for (const [a, b] of segs) {
      const c = closestPointOnSegment(p, a, b).point;
      const dd = dist(p, c);
      if (dd < r && (!best || dd < best.d)) best = { d: dd, pt: c };
    }
    return best ? { x: Math.round(best.pt.x * 10) / 10, y: Math.round(best.pt.y * 10) / 10 } : p;
  }

  addDraftPoint(pt) {
    const d = this.draft;
    if (d.pts.length >= 3 && dist(pt, d.pts[0]) < 0.5) return this.finishDraft();
    if (d.pts.length && dist(pt, d.pts[d.pts.length - 1]) < 1) return;
    d.pts.push(pt);
    d.typed = '';
    if (this.tool === 'wall' && d.pts.length === 2) return this.finishDraft();
    this.render();
  }

  // Place the next corner `length` cm from the last one in a straight direction.
  placeDraftStep(dir, length) {
    const d = this.draft;
    if (!d) return 'no-draft';
    if (!d.pts.length) return 'no-start';
    const last = d.pts[d.pts.length - 1];
    const v = { up: { x: 0, y: -1 }, down: { x: 0, y: 1 }, left: { x: -1, y: 0 }, right: { x: 1, y: 0 } }[dir];
    this.addDraftPoint(add(last, mul(v, length)));
    return 'ok';
  }

  finishDraft() {
    const pts = this.draft?.pts || [];
    const round = (q) => ({ x: Math.round(q.x * 10) / 10, y: Math.round(q.y * 10) / 10 });
    if (this.tool === 'wall') {
      if (pts.length < 2) return false;
      this.setTool('select');
      this.onWallDrawn?.(round(pts[0]), round(pts[1]));
      return true;
    }
    if (pts.length < 3) return false;
    this.setTool('select');
    this.onRoomDrawn?.(pts.map((q) => ({ x: Math.round(q.x * 10) / 10, y: Math.round(q.y * 10) / 10 })));
    return true;
  }

  // Keyboard while drawing: digits type the next wall's length, Enter places it (or closes
  // the shape), Backspace undoes, Esc cancels. Returns true when the key was used.
  draftKey(e) {
    const d = this.draft;
    if (!d) return false;
    if (/^[0-9]$/.test(e.key) || ((e.key === '.' || e.key === ',') && !d.typed.includes('.'))) {
      d.typed += e.key === ',' ? '.' : e.key;
    } else if (e.key === 'Backspace') {
      if (d.typed) d.typed = d.typed.slice(0, -1);
      else d.pts.pop();
    } else if (e.key === 'Enter') {
      const next = this.draftNext();
      if (d.typed && next && d.pts.length) this.addDraftPoint(next);
      else if (!this.finishDraft()) return true;
      return true;
    } else if (e.key === 'Escape') {
      this.setTool('select');
      return true;
    } else return false;
    this.render();
    return true;
  }

  // ---------- interaction ----------
  hit(e) {
    const t = e.target.closest('[data-pend],[data-vertex],[data-mid],[data-wall],[data-opening],[data-item],[data-partition],[data-room]');
    if (!t) return null;
    if (t.dataset.pend !== undefined) return { kind: 'pend', id: t.dataset.partition, index: +t.dataset.pend };
    if (t.dataset.partition) return { kind: 'partition', id: t.dataset.partition };
    if (t.dataset.mid !== undefined) return { kind: 'mid', id: t.dataset.room, index: +t.dataset.mid };
    if (t.dataset.vertex !== undefined) return { kind: 'vertex', id: t.dataset.room, index: +t.dataset.vertex };
    if (t.dataset.wall !== undefined) return { kind: 'wall', id: t.dataset.room, index: +t.dataset.wall };
    if (t.dataset.opening) return { kind: 'opening', id: t.dataset.opening };
    if (t.dataset.item) return { kind: 'item', id: t.dataset.item };
    return { kind: 'room', id: t.dataset.room };
  }

  snapPt(p) {
    return this.snapOn ? { x: snap(p.x, this.grid), y: snap(p.y, this.grid) } : p;
  }

  onDown(e) {
    this.svg.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 2) { // start pinch
      const [a, b] = [...this.pointers.values()];
      this.drag = { kind: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), scale0: this.view.scale };
      return;
    }
    const p = this.toPlan(e);
    const { store } = this;

    if (this.tool === 'measure') {
      const sp = this.snapPt(p);
      if (!this.measure || this.measure.done) this.measure = { a: sp, b: sp, done: false };
      else this.measure.done = true;
      this.render();
      return;
    }

    if (this.tool === 'draw' || this.tool === 'wall') {
      // a tap places a corner; a drag pans (decided on release)
      this.drag = { kind: 'pan', start: { x: e.clientX, y: e.clientY }, view0: { ...this.view }, tap: this.draftPoint(p) };
      return;
    }

    const h = this.hit(e);
    // with the layout locked, touching a room, wall, door or window just pans the view
    const fixed = this.locked && h && h.kind !== 'item';
    if (e.button === 1 || e.button === 2 || !h || fixed || e.shiftKey) {
      if ((!h || fixed) && e.button === 0) store.select(null);
      this.drag = { kind: 'pan', start: { x: e.clientX, y: e.clientY }, view0: { ...this.view } };
      return;
    }

    if (h.kind === 'wall') {
      this.onEditWall?.(h.id, h.index);
      return;
    }

    store.checkpoint();
    const plan = store.plan;
    if (h.kind === 'mid') {
      const room = plan.rooms.find((r) => r.id === h.id);
      const w = wallOf(room, h.index);
      insertVertex(plan, room, h.index, this.snapPt(mul(add(w.a, w.b), 0.5)));
      store.selection = { kind: 'vertex', id: room.id, index: h.index + 1 };
      this.drag = { kind: 'vertex', room, index: h.index + 1, moved: true };
    } else if (h.kind === 'item') {
      const it = plan.items.find((x) => x.id === h.id);
      store.selection = { kind: 'item', id: it.id };
      this.drag = { kind: 'item', target: it, grab: sub(p, it), moved: false };
    } else if (h.kind === 'vertex') {
      const room = plan.rooms.find((r) => r.id === h.id);
      store.selection = { kind: 'vertex', id: room.id, index: h.index };
      this.drag = { kind: 'vertex', room, index: h.index, moved: false };
    } else if (h.kind === 'opening') {
      const op = plan.openings.find((o) => o.id === h.id);
      const room = plan.rooms.find((r) => r.id === op.roomId);
      store.selection = { kind: 'opening', id: op.id };
      const g = openingGeom(op, room);
      this.drag = { kind: 'opening', op, room, grabOffset: closestPointOnSegment(p, g.wall.a, g.wall.b).t * g.wall.length - op.offset, moved: false };
    } else if (h.kind === 'partition') {
      const pw = plan.partitions.find((x) => x.id === h.id);
      store.selection = { kind: 'partition', id: pw.id };
      this.drag = { kind: 'partition', pw, start: p, a0: { ...pw.a }, b0: { ...pw.b }, moved: false };
    } else if (h.kind === 'pend') {
      const pw = plan.partitions.find((x) => x.id === h.id);
      store.selection = { kind: 'partition', id: pw.id };
      this.drag = { kind: 'pend', pw, index: h.index, moved: false };
    } else if (h.kind === 'room') {
      const room = plan.rooms.find((r) => r.id === h.id);
      store.selection = { kind: 'room', id: room.id };
      this.drag = { kind: 'room', room, start: p, pts0: room.points.map((q) => ({ ...q })), moved: false };
    }
    store.dragging = true;
    store.change();
  }

  onMove(e) {
    if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const p = this.toPlan(e);

    if (this.tool === 'measure' && this.measure && !this.measure.done) {
      this.measure.b = this.snapPt(p);
      this.render();
      return;
    }
    if ((this.tool === 'draw' || this.tool === 'wall') && !this.drag) {
      this.draft.cursor = this.draftPoint(p);
      this.render();
      return;
    }
    const d = this.drag;
    if (!d) return;
    const { store } = this;

    if (d.kind === 'pinch') {
      if (this.pointers.size < 2) return;
      const [a, b] = [...this.pointers.values()];
      const dd = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      this.zoomAt(mid, (d.scale0 * dd) / d.d0 / this.view.scale);
      return;
    }
    if (d.kind === 'pan') {
      this.userMoved = true;
      this.view.x = d.view0.x - (e.clientX - d.start.x) / this.view.scale;
      this.view.y = d.view0.y - (e.clientY - d.start.y) / this.view.scale;
      this.render();
      return;
    }
    d.moved = true;
    if (d.kind === 'item') {
      const it = d.target;
      const np = this.snapPt(sub(p, d.grab));
      it.x = np.x; it.y = np.y;
      if (this.snapOn) this.snapToWalls(it);
    } else if (d.kind === 'vertex') {
      d.room.points[d.index] = this.snapVertex(p, d.room, d.index);
    } else if (d.kind === 'partition') {
      const delta = this.snapPt(sub(p, d.start));
      d.pw.a = add(d.a0, delta);
      d.pw.b = add(d.b0, delta);
    } else if (d.kind === 'pend') {
      const other = d.index === 0 ? d.pw.b : d.pw.a;
      const r = this.px(12);
      let q = this.snapPt(p);
      if (Math.abs(q.x - other.x) < r) q.x = other.x;
      if (Math.abs(q.y - other.y) < r) q.y = other.y;
      if (this.snapOn) q = this.onWallLine(q, r, d.pw.id);
      d.pw[d.index === 0 ? 'a' : 'b'] = q;
    } else if (d.kind === 'room') {
      const delta = this.snapPt(sub(p, d.start));
      d.room.points = d.pts0.map((q) => add(q, delta));
    } else if (d.kind === 'opening') {
      const w = wallOf(d.room, d.op.wall);
      // allow moving the opening to whichever wall of its room is nearest
      let bestWall = d.op.wall, bestDist = Infinity;
      d.room.points.forEach((_, i) => {
        const wi = wallOf(d.room, i);
        const dd = dist(p, closestPointOnSegment(p, wi.a, wi.b).point);
        if (dd < bestDist && wi.length >= d.op.width) { bestDist = dd; bestWall = i; }
      });
      if (bestWall !== d.op.wall) { d.op.wall = bestWall; d.grabOffset = d.op.width / 2; }
      const ww = bestWall === d.op.wall ? wallOf(d.room, d.op.wall) : w;
      const along = closestPointOnSegment(p, ww.a, ww.b).t * ww.length - d.grabOffset;
      d.op.offset = Math.max(0, Math.min(ww.length - d.op.width, this.snapOn ? snap(along, this.grid) : along));
    }
    store.change();
  }

  onUp(e) {
    this.pointers.delete(e.pointerId);
    const d = this.drag;
    if (d?.kind === 'pinch') { if (this.pointers.size < 2) this.drag = null; return; }
    this.drag = null;
    if (d?.tap && this.draft && Math.hypot(e.clientX - d.start.x, e.clientY - d.start.y) < 6) {
      this.draft.cursor = d.tap;
      // with a typed length, a tap places the corner at that distance towards the tap
      this.addDraftPoint(this.draft.typed ? this.draftNext() : d.tap);
      return;
    }
    if (!d || d.kind === 'pan') return;
    this.store.dragging = false;
    if (!d.moved) this.store.dropCheckpoint();
    this.store.commit();
  }

  onWheel(e) {
    e.preventDefault();
    // mouse wheel and trackpad pinch (ctrlKey) zoom; sideways trackpad scrolling pans
    if (e.ctrlKey || e.deltaX === 0) {
      const k = e.deltaMode === 1 ? 0.05 : e.ctrlKey ? 0.01 : 0.0015;
      this.zoomAt({ x: e.clientX, y: e.clientY }, Math.exp(-e.deltaY * k));
    } else {
      // trackpad two-finger scroll pans
      this.userMoved = true;
      this.view.x += e.deltaX / this.view.scale;
      this.view.y += e.deltaY / this.view.scale;
      this.render();
    }
  }

  onDblClick(e) {
    if (this.locked) return;
    const p = this.toPlan(e);
    const { plan } = this.store;
    // double-click near a wall of the selected room inserts a corner there
    const sel = this.store.selection;
    const room = sel && (sel.kind === 'room' || sel.kind === 'vertex') ? plan.rooms.find((r) => r.id === sel.id) : null;
    if (!room) return;
    let best = null;
    room.points.forEach((_, i) => {
      const w = wallOf(room, i);
      const c = closestPointOnSegment(p, w.a, w.b);
      const dd = dist(p, c.point);
      if (dd < this.px(14) && c.t > 0.02 && c.t < 0.98 && (!best || dd < best.d)) best = { i, d: dd, pt: c.point };
    });
    if (!best) return;
    this.store.edit(() => {
      insertVertex(plan, room, best.i, this.snapPt(best.pt));
      this.store.selection = { kind: 'vertex', id: room.id, index: best.i + 1 };
    });
  }

  snapVertex(p, room, index) {
    // snap to other rooms' corners (to join rooms), then to the neighbour's axis, then grid
    const { plan } = this.store;
    const r = this.px(12);
    for (const other of plan.rooms) {
      if (other.id === room.id) continue;
      for (const q of other.points) if (dist(p, q) < r) return { ...q };
    }
    let out = this.snapPt(p);
    const n = room.points.length;
    for (const nb of [room.points[(index - 1 + n) % n], room.points[(index + 1) % n]]) {
      if (Math.abs(out.x - nb.x) < r) out.x = nb.x;
      if (Math.abs(out.y - nb.y) < r) out.y = nb.y;
    }
    return out;
  }

  snapToWalls(it) {
    for (const c of this.clearances(it)) {
      if (c.dist > 0.01 && c.dist < SNAP_WALL) {
        it.x += c.dir.x * c.dist;
        it.y += c.dir.y * c.dist;
      }
    }
  }
}

function cross2(a, b) { return a.x * b.y - a.y * b.x; }
