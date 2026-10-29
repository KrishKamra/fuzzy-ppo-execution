/**
 * The eFNN's regime space in 3D (docs/UI_DESIGN.md §3.3). Its input is three-dimensional, so each rule is a real
 * ellipsoid (centre ± 1σ per input) and the order's market state is a point moving through it. Plain three.js
 * (no React wrapper library), loaded lazily with the Brain screen. It renders only when something changed and
 * stops while off-screen.
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

import type { Rules } from "@/api/client";
import type { Ink } from "@/lib/ink";
import type { Theme } from "@/lib/palette";

import { INPUT_NAMES, paceColor } from "./fuzzy";

const LIM = 3.2; // the box spans ±3.2 standard deviations
const clamp = (v: number) => Math.max(-LIM, Math.min(LIM, v));
/** Input j → scene axis: x = imbalance, y = volatility (up), z = spread variance. */
const toScene = (x: readonly number[]) => new THREE.Vector3(clamp(x[0] ?? 0), clamp(x[2] ?? 0), clamp(x[1] ?? 0));

export interface RegimeSpaceProps {
  rules: Rules;
  path: number[][]; // regime inputs per decision
  k: number; // decisions revealed
  active: number | null;
  hover: number | null;
  onHover: (rule: number | null) => void;
  theme: Theme;
  ink: Ink;
}

interface Ctx {
  renderer: THREE.WebGLRenderer;
  labels: CSS2DRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  ruleGroup: THREE.Group;
  guideGroup: THREE.Group;
  meshes: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>[];
  trail: THREE.Line;
  point: THREE.Group;
  dirty: boolean;
}

function label(text: string, className: string): CSS2DObject {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  return new CSS2DObject(el);
}

function glowTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  if (g) {
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    r.addColorStop(0, "rgba(255,255,255,0.95)");
    r.addColorStop(0.35, "rgba(167,139,250,0.55)");
    r.addColorStop(1, "rgba(34,211,238,0)");
    g.fillStyle = r;
    g.fillRect(0, 0, 64, 64);
  }
  return new THREE.CanvasTexture(c);
}

/** Dispose the dots riding on the path line (they share its geometry, so only their material). */
function disposeTrailDots(trail: THREE.Line) {
  trail.children.forEach((ch) => ((ch as THREE.Points).material as THREE.Material).dispose());
  trail.clear();
}

function disposeGroup(group: THREE.Group) {
  group.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat?.dispose();
    if (o instanceof CSS2DObject) o.element.remove();
  });
  group.clear();
}

export default function RegimeSpace({ rules, path, k, active, hover, onHover, theme, ink }: RegimeSpaceProps) {
  const host = useRef<HTMLDivElement>(null);
  const ctx = useRef<Ctx | null>(null);
  const hoverRef = useRef(onHover);
  hoverRef.current = onHover;

  // Scene, camera, renderers, controls and the render loop: once.
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.domElement.style.position = "absolute";
    labels.domElement.style.inset = "0";
    labels.domElement.style.pointerEvents = "none";
    el.appendChild(labels.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(8.4, 5.6, 9.6);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(5, 9, 6);
    scene.add(sun);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 6;
    controls.maxDistance = 20;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    controls.autoRotate = !reduced;
    controls.autoRotateSpeed = 0.6;
    controls.addEventListener("start", () => (controls.autoRotate = false)); // the viewer takes over

    const ruleGroup = new THREE.Group();
    const guideGroup = new THREE.Group();
    scene.add(guideGroup, ruleGroup);

    const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }));
    scene.add(trail);
    const point = new THREE.Group();
    point.add(new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })));
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
    glow.scale.setScalar(0.9);
    point.add(glow);
    scene.add(point);

    const c: Ctx = { renderer, labels, scene, camera, controls, ruleGroup, guideGroup, meshes: [], trail, point, dirty: true };
    ctx.current = c;
    controls.addEventListener("change", () => (c.dirty = true));

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight; // fmt: skip
      if (!w || !h) return;
      renderer.setSize(w, h);
      labels.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      c.dirty = true;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    // Hover a rule: raycast against the ellipsoids.
    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const onMove = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(mouse, camera);
      const hit = ray.intersectObjects(c.meshes, false)[0];
      const idx = hit ? c.meshes.indexOf(hit.object as Ctx["meshes"][number]) : -1;
      renderer.domElement.style.cursor = idx >= 0 ? "pointer" : "grab";
      hoverRef.current(idx >= 0 ? idx : null);
    };
    const onLeave = () => hoverRef.current(null);
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerleave", onLeave);

    // Render only when something changed (camera, data, highlight), and only while on screen.
    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = !!e?.isIntersecting));
    io.observe(el);
    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      if (!visible) return;
      if (controls.update()) c.dirty = true; // damping or auto-rotate moved the camera
      if (!c.dirty) return;
      c.dirty = false;
      renderer.render(scene, camera);
      labels.render(scene, camera);
    };
    loop();

    return () => {
      cancelAnimationFrame(frame);
      io.disconnect();
      ro.disconnect();
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerleave", onLeave);
      controls.dispose();
      disposeGroup(ruleGroup);
      disposeGroup(guideGroup);
      disposeGroup(point);
      disposeTrailDots(trail);
      trail.geometry.dispose();
      (trail.material as THREE.Material).dispose();
      renderer.dispose();
      el.replaceChildren();
      ctx.current = null;
    };
  }, []);

  // The box, the tercile planes and the axis titles: when the rules (terciles) or the theme change.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    disposeGroup(c.guideGroup);
    const line = new THREE.LineBasicMaterial({ color: new THREE.Color(ink.theme === "dark" ? 0xffffff : 0x0c0e18), transparent: true, opacity: 0.16 });
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * LIM, 2 * LIM, 2 * LIM)), line);
    c.guideGroup.add(box);
    const faint = new THREE.LineBasicMaterial({ color: line.color, transparent: true, opacity: 0.07 });
    const square = (axis: "x" | "y" | "z", v: number) => {
      const p = [[-LIM, -LIM], [LIM, -LIM], [LIM, LIM], [-LIM, LIM]].map(([a = 0, b = 0]) =>
        axis === "x" ? new THREE.Vector3(v, a, b) : axis === "y" ? new THREE.Vector3(a, v, b) : new THREE.Vector3(a, b, v),
      ); // fmt: skip
      return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(p), faint);
    };
    // Tercile cut planes: where LOW / MED / HIGH change, per input.
    const axisOf = ["x", "z", "y"] as const; // input 0 → x, 1 → z, 2 → y
    rules.terciles.forEach((cuts, j) => cuts.forEach((v) => c.guideGroup.add(square(axisOf[j] ?? "x", clamp(v)))));
    const cls = "pointer-events-none whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] text-fg-muted bg-surface-1/70";
    const titles: [string, THREE.Vector3][] = [
      // On the box's front-bottom and side edges, so they stay inside the frame from any default view.
      [`${INPUT_NAMES[0]} → with us`, new THREE.Vector3(0, -LIM, LIM)],
      [`${INPUT_NAMES[2]} ↑`, new THREE.Vector3(-LIM, 0, LIM)],
      [`${INPUT_NAMES[1]} →`, new THREE.Vector3(LIM, -LIM, 0)],
    ];
    for (const [text, at] of titles) {
      const l = label(text, cls);
      l.position.copy(at);
      c.guideGroup.add(l);
    }
    c.dirty = true;
  }, [rules, ink]);

  // One translucent ellipsoid per rule, coloured by the rule's pace.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    disposeGroup(c.ruleGroup);
    c.meshes = rules.rules.map((r) => {
      const color = new THREE.Color(paceColor(r.pace, theme));
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(1, 40, 24),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.08, transparent: true, opacity: 0.2, roughness: 0.45, metalness: 0, depthWrite: false }),
      );
      mesh.position.copy(toScene(r.centres));
      mesh.scale.set(
        Math.min(LIM, Math.max(0.12, r.widths[0] ?? 1)),
        Math.min(LIM, Math.max(0.12, r.widths[2] ?? 1)),
        Math.min(LIM, Math.max(0.12, r.widths[1] ?? 1)),
      );
      const tag = label(`R${r.rule}`, "pointer-events-none font-mono text-[11px] text-fg");
      tag.position.copy(mesh.position);
      c.ruleGroup.add(mesh, tag);
      return mesh;
    });
    c.dirty = true;
  }, [rules, theme]);

  // The order's path through the space, fading from graphite to the agent's amber over time.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const pts = path.map(toScene);
    const geo = new THREE.BufferGeometry().setFromPoints(pts.length ? pts : [new THREE.Vector3()]);
    const a = new THREE.Color("#5b6472");
    const b = new THREE.Color("#f2a93b");
    const colors = new Float32Array(Math.max(pts.length, 1) * 3);
    pts.forEach((_, i) => {
      const col = a.clone().lerp(b, pts.length > 1 ? i / (pts.length - 1) : 0);
      colors.set([col.r, col.g, col.b], i * 3);
    });
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    c.trail.geometry.dispose();
    c.trail.geometry = geo;
    const dots = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, opacity: 0.9 }));
    disposeTrailDots(c.trail);
    c.trail.add(dots); // shares the geometry, so the draw range below reveals both
    c.dirty = true;
  }, [path]);

  // The cursor: reveal the path up to k and move the point.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    const i = Math.max(0, Math.min(path.length - 1, k - 1));
    c.trail.geometry.setDrawRange(0, Math.max(1, Math.min(k, path.length)));
    if (path[i]) c.point.position.copy(toScene(path[i]));
    c.dirty = true;
  }, [k, path]);

  // Highlight: the active rule glows; a hovered rule (here, in the bars or in the table) lifts.
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    c.meshes.forEach((m, i) => {
      const on = i === active;
      const lifted = i === hover;
      m.material.opacity = on ? 0.5 : lifted ? 0.4 : hover !== null ? 0.08 : 0.2;
      m.material.emissiveIntensity = on ? 0.45 : lifted ? 0.3 : 0.08;
    });
    c.dirty = true;
  }, [active, hover, rules]);

  return <div ref={host} className="relative h-full w-full cursor-grab" role="img" aria-label={`3D regime space: ${rules.rules.length} fuzzy rules as ellipsoids over imbalance, spread variance and volatility, with the order's path`} />;
}
