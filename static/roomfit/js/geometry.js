// 2D geometry helpers. All units are centimetres, plan coordinates have y pointing "down" (SVG).

export const EPS = 1e-6;

export const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
export const mul = (a, s) => ({ x: a.x * s, y: a.y * s });
export const dot = (a, b) => a.x * b.x + a.y * b.y;
export const cross = (a, b) => a.x * b.y - a.y * b.x;
export const len = (a) => Math.hypot(a.x, a.y);
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const norm = (a) => {
  const l = len(a);
  return l < EPS ? { x: 0, y: 0 } : { x: a.x / l, y: a.y / l };
};
export const perp = (a) => ({ x: -a.y, y: a.x });
export const rotate = (p, deg) => {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r), s = Math.sin(r);
  return { x: p.x * c - p.y * s, y: p.x * s + p.y * c };
};

export function signedArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

// Sutherland–Hodgman: the part of polygon `pts` (may be concave) inside the convex polygon
// `clip`. A concave `pts` can come back with zero-width slivers, but its area is still exact.
export function clipToConvex(pts, clip) {
  const s = signedArea(clip) < 0 ? -1 : 1;
  let out = pts;
  for (let i = 0; i < clip.length && out.length; i++) {
    const a = clip[i], e = sub(clip[(i + 1) % clip.length], a);
    const side = (p) => s * cross(e, sub(p, a));
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const p = input[j], q = input[(j + 1) % input.length];
      const dp = side(p), dq = side(q);
      if (dp >= 0) out.push(p);
      if ((dp >= 0) !== (dq >= 0)) out.push(add(p, mul(sub(q, p), dp / (dp - dq))));
    }
  }
  return out;
}

export function centroid(pts) {
  let cx = 0, cy = 0, a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    const f = p.x * q.y - q.x * p.y;
    cx += (p.x + q.x) * f;
    cy += (p.y + q.y) * f;
    a += f;
  }
  if (Math.abs(a) < EPS) {
    const n = pts.length;
    return { x: pts.reduce((s, p) => s + p.x, 0) / n, y: pts.reduce((s, p) => s + p.y, 0) / n };
  }
  return { x: cx / (3 * a), y: cy / (3 * a) };
}

export function pointInPolygon(pt, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a.y > pt.y) !== (b.y > pt.y) && pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

export function closestPointOnSegment(p, a, b) {
  const ab = sub(b, a);
  const l2 = dot(ab, ab);
  let t = l2 < EPS ? 0 : dot(sub(p, a), ab) / l2;
  t = Math.max(0, Math.min(1, t));
  return { point: add(a, mul(ab, t)), t };
}

export function segmentsIntersect(p1, p2, q1, q2) {
  const r = sub(p2, p1), s = sub(q2, q1);
  const den = cross(r, s);
  if (Math.abs(den) < EPS) return false;
  const t = cross(sub(q1, p1), s) / den;
  const u = cross(sub(q1, p1), r) / den;
  return t > EPS && t < 1 - EPS && u > EPS && u < 1 - EPS;
}

// Ray (origin o, unit direction d) vs segment ab; returns distance or Infinity.
export function raySegment(o, d, a, b) {
  const s = sub(b, a);
  const den = cross(d, s);
  if (Math.abs(den) < EPS) return Infinity;
  const ao = sub(a, o);
  const t = cross(ao, s) / den;
  const u = cross(ao, d) / den;
  if (t >= -EPS && u >= -EPS && u <= 1 + EPS) return Math.max(0, t);
  return Infinity;
}

export function edges(pts) {
  return pts.map((p, i) => [p, pts[(i + 1) % pts.length]]);
}

// Separating axis test for two convex polygons. Returns true if interiors overlap
// by more than `tol` along every axis (touching surfaces don't count).
export function convexOverlap(A, B, tol = 0.5) {
  for (const poly of [A, B]) {
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length];
      const axis = norm(perp(sub(q, p)));
      let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
      for (const v of A) { const d = dot(v, axis); minA = Math.min(minA, d); maxA = Math.max(maxA, d); }
      for (const v of B) { const d = dot(v, axis); minB = Math.min(minB, d); maxB = Math.max(maxB, d); }
      if (maxA - minB <= tol || maxB - minA <= tol) return false;
    }
  }
  return true;
}

// Is the (possibly concave) polygon `inner` fully inside polygon `outer`?
// `tol` shrinks the test so items flush against a wall still count as inside.
export function polygonInside(inner, outer, tol = 0.5) {
  const c = centroid(inner);
  const shrunk = inner.map((p) => {
    const d = sub(c, p), l = len(d);
    return l < tol ? p : add(p, mul(d, tol / l));
  });
  if (!shrunk.every((p) => pointInPolygon(p, outer))) return false;
  for (const [a, b] of edges(shrunk)) {
    for (const [c1, c2] of edges(outer)) {
      if (segmentsIntersect(a, b, c1, c2)) return false;
    }
  }
  return true;
}

export function bbox(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
  }
  return { minX, minY, maxX, maxY };
}

export const snap = (v, step) => (step > 0 ? Math.round(v / step) * step : v);
