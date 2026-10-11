// 3D preview: walls at real ceiling height with door/window openings, furniture as solids.
// Plan (x, y) maps to world (x, z); height is world y. Units stay in cm.
// Full CDN URLs (jsDelivr's +esm build rewrites three's internal imports) so no import map is
// needed; import maps only arrived in Safari 16.4.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js/+esm';
import { sub, dot, len, bbox } from './geometry.js?v=6250f90731';
import { wallOf, openingGeom, checkPlan, worstLevel, partitionLength } from './model.js?v=6250f90731';

const WALL_T = 10;

export class View3D {
  constructor(container, store) {
    this.store = store;
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 5, 20000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
    this.controls.addEventListener('change', () => this.draw());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(400, 900, 600);
    this.scene.add(sun);

    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.framed = false;
    this.dirty = true;

    new ResizeObserver(() => this.resize()).observe(container);
    store.subscribe(() => { this.dirty = true; this.schedule(); });
    this.bindPicking();
  }

  get visible() { return this.container.offsetParent !== null && this.container.clientWidth > 0; }

  schedule() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = null;
      if (!this.visible) return;
      if (this.dirty) this.rebuild();
      this.draw();
    });
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (!this.framed) this.frame();
    this.schedule();
  }

  draw() {
    if (this.visible) this.renderer.render(this.scene, this.camera);
  }

  frame() {
    const pts = this.store.plan.rooms.flatMap((r) => r.points);
    if (!pts.length) return;
    const b = bbox(pts);
    const cx = (b.minX + b.maxX) / 2, cz = (b.minY + b.maxY) / 2;
    const span = Math.max(b.maxX - b.minX, b.maxY - b.minY);
    this.controls.target.set(cx, 60, cz);
    this.camera.position.set(cx + span * 0.35, span * 0.95, cz + span * 0.95);
    this.controls.update();
    this.framed = true;
    this.draw();
  }

  bindPicking() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    el.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      const r = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(ndc, this.camera);
      const hit = ray.intersectObjects(this.content.children, true).find((h) => findItemId(h.object) || h.object.userData.partitionId);
      const partitionId = hit?.object.userData.partitionId;
      this.store.select(!hit ? null : partitionId ? { kind: 'partition', id: partitionId } : { kind: 'item', id: findItemId(hit.object) });
    });
  }

  rebuild() {
    this.dirty = false;
    const { plan, selection } = this.store;
    const css = getComputedStyle(document.documentElement);
    const token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    this.scene.background = new THREE.Color(token('--paper', '#e9ecee'));

    for (const child of [...this.content.children]) disposeTree(child);
    this.content.clear();

    const wallMat = new THREE.MeshStandardMaterial({ color: token('--wall-3d', '#f4f4f1'), roughness: 0.95 });
    const issues = checkPlan(plan);

    // floors
    for (const room of plan.rooms) {
      const shape = new THREE.Shape(room.points.map((p) => new THREE.Vector2(p.x, -p.y)));
      const geo = new THREE.ShapeGeometry(shape);
      geo.rotateX(-Math.PI / 2);
      const floor = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: token(`--floor-${room.floor || 'plain'}`, '#ddd'), roughness: 1 }));
      this.content.add(floor);
    }

    // openings, as world segments, so a door cuts every wall it sits in (shared walls included)
    const holes = [];
    for (const op of plan.openings) {
      const room = plan.rooms.find((r) => r.id === op.roomId);
      if (!room) continue;
      const g = openingGeom(op, room);
      const bottom = op.type === 'window' ? op.sill : 0;
      holes.push({ a: g.start, b: g.end, bottom, top: bottom + op.height, type: op.type });
    }

    for (const room of plan.rooms) {
      room.points.forEach((_, i) => {
        const w = wallOf(room, i);
        if (w.length < 1) return;
        const cuts = [];
        for (const h of holes) {
          const da = distToLine(h.a, w), db = distToLine(h.b, w);
          if (da > 2 || db > 2) continue;
          let s = dot(sub(h.a, w.a), w.dir), e = dot(sub(h.b, w.a), w.dir);
          if (s > e) [s, e] = [e, s];
          s = Math.max(0, s); e = Math.min(w.length, e);
          if (e - s > 1) cuts.push({ s, e, bottom: h.bottom, top: Math.min(h.top, room.height) });
        }
        cuts.sort((p, q) => p.s - q.s);
        let cursor = 0;
        for (const c of cuts) {
          if (c.s > cursor) this.addWall(w, cursor, c.s, 0, room.height, wallMat);
          if (c.bottom > 0) this.addWall(w, c.s, c.e, 0, c.bottom, wallMat);
          if (c.top < room.height) this.addWall(w, c.s, c.e, c.top, room.height, wallMat);
          cursor = Math.max(cursor, c.e);
        }
        if (cursor < w.length) this.addWall(w, cursor, w.length, 0, room.height, wallMat);
      });
    }

    // inner walls (half walls stop short of the ceiling)
    for (const pw of plan.partitions) {
      const L = partitionLength(pw);
      if (L < 1 || pw.height < 0.5) continue;
      const selected = selection?.kind === 'partition' && selection.id === pw.id;
      const geo = new THREE.BoxGeometry(L, pw.height, pw.thickness);
      const m = new THREE.Mesh(geo, wallMat);
      m.position.set((pw.a.x + pw.b.x) / 2, pw.height / 2, (pw.a.y + pw.b.y) / 2);
      m.rotation.y = -Math.atan2(pw.b.y - pw.a.y, pw.b.x - pw.a.x);
      m.userData.partitionId = pw.id;
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: selected ? token('--tape', '#f2c200') : token('--muted', '#888') }));
      edges.position.copy(m.position);
      edges.rotation.copy(m.rotation);
      this.content.add(m, edges);
    }

    // window glass
    const glass = new THREE.MeshStandardMaterial({ color: 0x9cc7e6, transparent: true, opacity: 0.35 });
    for (const h of holes) {
      if (h.type !== 'window') continue;
      const d = sub(h.b, h.a), L = len(d);
      const m = new THREE.Mesh(new THREE.BoxGeometry(L, h.top - h.bottom, 1.5), glass);
      m.position.set((h.a.x + h.b.x) / 2, (h.top + h.bottom) / 2, (h.a.y + h.b.y) / 2);
      m.rotation.y = -Math.atan2(d.y, d.x);
      this.content.add(m);
    }

    // furniture
    const showCandidates = this.showCandidates !== false;
    for (const it of plan.items) {
      if (it.hidden || (!showCandidates && it.status === 'candidate')) continue;
      const level = worstLevel(issues.get(it.id));
      const selected = selection?.kind === 'item' && selection.id === it.id;
      const mat = new THREE.MeshStandardMaterial({
        color: it.color, roughness: 0.8,
        transparent: it.status === 'candidate', opacity: it.status === 'candidate' ? 0.82 : 1,
        emissive: level === 'error' ? 0x9a1d1d : level === 'warn' ? 0x7a4300 : 0x000000,
        emissiveIntensity: level === 'error' || level === 'warn' ? 0.45 : 0,
      });
      const edgeColor = selected ? token('--tape', '#f2c200') : '#2b2f36';
      const group = new THREE.Group();
      group.userData.itemId = it.id;
      const h = Math.max(it.h, 0.5);
      const addPart = (geo, x, z) => {
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, 0, z);
        mesh.userData.itemId = it.id;
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: edgeColor }));
        edges.position.copy(mesh.position);
        group.add(mesh, edges);
      };
      if (it.shape === 'ellipse') {
        const geo = new THREE.CylinderGeometry(it.w / 2, it.w / 2, h, 40);
        geo.scale(1, 1, it.d / it.w);
        addPart(geo, 0, 0);
      } else if (it.shape === 'L') {
        const lw = Math.min(it.lw, it.w), ld = Math.min(it.ld, it.d), sx = it.mirror ? -1 : 1;
        addPart(new THREE.BoxGeometry(it.w, h, ld), 0, -it.d / 2 + ld / 2);
        addPart(new THREE.BoxGeometry(lw, h, it.d - ld), sx * (it.w / 2 - lw / 2), ld / 2);
      } else {
        addPart(new THREE.BoxGeometry(it.w, h, it.d), 0, 0);
      }
      group.position.set(it.x, it.elevation + h / 2, it.y);
      group.rotation.y = -(it.rot * Math.PI) / 180;
      this.content.add(group);
    }
  }

  addWall(w, s, e, y0, y1, mat) {
    const L = e - s, H = y1 - y0;
    if (L < 0.5 || H < 0.5) return;
    const mid = (s + e) / 2;
    const cx = w.a.x + w.dir.x * mid - w.inward.x * WALL_T / 2;
    const cz = w.a.y + w.dir.y * mid - w.inward.y * WALL_T / 2;
    const m = new THREE.Mesh(new THREE.BoxGeometry(L, H, WALL_T), mat);
    m.position.set(cx, (y0 + y1) / 2, cz);
    m.rotation.y = -Math.atan2(w.dir.y, w.dir.x);
    this.content.add(m);
  }
}

function distToLine(p, w) {
  const d = sub(p, w.a);
  return Math.abs(d.x * w.dir.y - d.y * w.dir.x);
}

function findItemId(obj) {
  for (let o = obj; o; o = o.parent) if (o.userData?.itemId) return o.userData.itemId;
  return null;
}

function disposeTree(obj) {
  obj.traverse((o) => {
    o.geometry?.dispose();
    if (o.material && !o.material.userData.shared) o.material.dispose();
  });
}
