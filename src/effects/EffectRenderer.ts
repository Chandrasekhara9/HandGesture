import type { Point, TrackedHand } from "../vision/gestures";
import { FILTROS } from "./PortalFilters";

export type Effect = "cursor" | "energy" | "draw" | "multi" | "portal" | "wireframe";

interface Stroke {
  id: string;
  points: Point[];
  color: string;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  hue: number;
}

interface RenderHand extends TrackedHand {
  indexTip: Point;
  palm: Point;
  landmarks: Point[];
}

const handColors: Record<string, string> = {
  Left: "#4cf6ff",
  Right: "#ff55cb",
  Unknown: "#b28cff",
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const PALM_INDICES = [0, 5, 9, 13, 17];
const CONFIG = {
  portalSmoothing: 0.5,
  portalFadeMs: 300,
  portalDebounceMs: 1000,
  drawClearHoldMs: 1000,
  portalOutline: 1,
  orbRings: 5,
  maxParticles: 300,
  rotationSpeed: 0.9,
  landmarkSmoothing: 18,
  glowStrength: 1,
  retainedStrokes: 8,
  pointsPerStroke: 600,
};
const OCTAHEDRON_VERTICES: [number, number, number][] = [
  [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
];
const OCTAHEDRON_EDGES: [number, number][] = [
  [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 4], [4, 3], [3, 5], [5, 2],
];

function convexQuad(points: Point[]): Point[] {
  if (points.length < 4) return [];
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: Point, a: Point, b: Point) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: Point[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Point[] = [];
  for (let index = sorted.length - 1; index >= 0; index--) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  if (hull.length !== 4) return [];
  let first = 0;
  for (let index = 1; index < hull.length; index++) {
    if (hull[index].x + hull[index].y < hull[first].x + hull[first].y) first = index;
  }
  return hull.slice(first).concat(hull.slice(0, first));
}

export class EffectRenderer {
  private ctx: CanvasRenderingContext2D;
  private drawStrokes: Stroke[] = [];
  private activeStroke: Stroke | null = null;
  private handTrails = new Map<string, Stroke>();
  private cursorTrail: Point[] = [];
  private smoothed = new Map<string, Point>();
  private lastTime = 0;
  private energy = 0;
  private animationTime = 0;
  private portalOpacity = 0;
  private portalPoints: Point[] = [];
  private portalSource = document.createElement("canvas");
  private portalOutput = document.createElement("canvas");
  private portalFilterIndex = 0;
  private portalLastRender = 0;
  private openPalmStarted = 0;
  private portalLastSwitch = 0;
  private portalOpenLatched = false;
  private drawClearStarted = 0;
  private drawClearLatched = false;
  private orbCenter: Point = { x: 0, y: 0 };
  private orbRadius = 0;
  private lightning: Point[][] = [];
  private lightningFrame = 0;
  private particles: Particle[] = [];
  private showBeams = true;
  private showParticles = true;
  private showGlow = true;
  private rotationSpeed = CONFIG.rotationSpeed;

  constructor(private canvas: HTMLCanvasElement) {
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) throw new Error("Canvas 2D rendering is unavailable.");
    this.ctx = context;
  }

  clearDrawing() {
    this.drawStrokes = [];
    this.activeStroke = null;
  }

  undoDrawing() {
    this.drawStrokes.pop();
    this.activeStroke = null;
  }

  nextPortalFilter() {
    this.portalFilterIndex = (this.portalFilterIndex + 1) % FILTROS.length;
  }

  toggleSetting(setting: "glow" | "rotation" | "beams" | "particles") {
    if (setting === "glow") this.showGlow = !this.showGlow;
    if (setting === "beams") this.showBeams = !this.showBeams;
    if (setting === "particles") this.showParticles = !this.showParticles;
    if (setting === "rotation") this.rotationSpeed = this.rotationSpeed >= 2.4 ? 0.3 : this.rotationSpeed + 0.3;
  }

  get currentPortalFilter() {
    return FILTROS[this.portalFilterIndex].name;
  }

  resize(width: number, height: number, pixelRatio: number) {
    const pixelWidth = Math.round(width * pixelRatio);
    const pixelHeight = Math.round(height * pixelRatio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    this.ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  render(
    effect: Effect,
    hands: TrackedHand[],
    width: number,
    height: number,
    time: number,
    videoSize: { width: number; height: number },
    video: HTMLVideoElement,
  ) {
    const ctx = this.ctx;
    const dt = Math.min((time - (this.lastTime || time)) / 1000, 0.05);
    this.lastTime = time;
    this.animationTime += dt;
    ctx.clearRect(0, 0, width, height);
    const points: RenderHand[] = hands.map((hand) => {
      const landmarks = hand.landmarks.map((point, index) =>
        this.smooth(`${hand.id}-landmark-${index}`, this.map(point, width, height, videoSize), dt),
      );
      const palm = { x: 0, y: 0 };
      for (const index of PALM_INDICES) {
        palm.x += landmarks[index].x / PALM_INDICES.length;
        palm.y += landmarks[index].y / PALM_INDICES.length;
      }
      return { ...hand, indexTip: landmarks[8], palm, landmarks };
    });
    const seen = new Set(points.map((hand) => hand.id));
    for (const key of this.smoothed.keys()) {
      let present = seen.has(key);
      for (const id of seen) if (!present && (key === `wire-${id}` || key.startsWith(`${id}-landmark-`))) present = true;
      if (!present) this.smoothed.delete(key);
    }

    if (effect === "draw") this.updateDrawing(points[0], time);
    else this.activeStroke = null;
    if (effect === "multi") this.updateHandLines(points);
    else this.handTrails.clear();

    if (effect === "cursor") this.drawCursor(points.find((hand) => hand.indexOnly), width, height, dt);
    if (effect === "energy") this.drawOrb(points, width, height, dt);
    if (effect === "draw") this.drawSavedStrokes(width, height);
    if (effect === "multi") this.drawMultiLines(width, height);
    if (effect === "portal") this.drawPortal(points, video, width, height, time, dt);
    else this.portalOpacity = Math.max(0, this.portalOpacity - dt / (CONFIG.portalFadeMs / 1000));
    if (effect === "wireframe") this.drawWireframes(points, width, dt);
    else if (effect !== "energy") this.particles.length = 0;
  }

  private map(point: Point, width: number, height: number, videoSize: { width: number; height: number }) {
    const videoAspect = videoSize.width / Math.max(videoSize.height, 1);
    const canvasAspect = width / Math.max(height, 1);
    const visibleWidth = videoAspect > canvasAspect ? canvasAspect / videoAspect : 1;
    const visibleHeight = videoAspect < canvasAspect ? videoAspect / canvasAspect : 1;
    const x = (point.x - (1 - visibleWidth) / 2) / visibleWidth;
    const y = (point.y - (1 - visibleHeight) / 2) / visibleHeight;
    return { x: x * width, y: y * height };
  }

  private smooth(id: string, target: Point, dt: number) {
    const previous = this.smoothed.get(id) ?? target;
    const smoothing = 1 - Math.exp(-dt * CONFIG.landmarkSmoothing);
    const next = { x: lerp(previous.x, target.x, smoothing), y: lerp(previous.y, target.y, smoothing) };
    this.smoothed.set(id, next);
    return next;
  }

  private updateDrawing(hand: RenderHand | undefined, time: number) {
    if (hand?.open) {
      if (!this.drawClearStarted) this.drawClearStarted = time;
      const progress = clamp((time - this.drawClearStarted) / CONFIG.drawClearHoldMs, 0, 1);
      this.drawClearLatched = progress >= 1;
      this.drawHoldRing(hand.palm, progress);
      if (this.drawClearLatched && this.drawStrokes.length) {
        this.clearDrawing();
      }
      return;
    }
    this.drawClearStarted = 0;
    this.drawClearLatched = false;
    if (!hand?.indexOnly) {
      this.activeStroke = null;
      return;
    }
    if (hand.pinching) {
      this.activeStroke = null;
      return;
    }
    const point = this.smooth(hand.id, hand.indexTip, 1 / 30);
    if (!this.activeStroke) {
      this.activeStroke = { id: "draw", points: [point], color: "#dc72ff" };
      this.drawStrokes.push(this.activeStroke);
      if (this.drawStrokes.length > CONFIG.retainedStrokes) this.drawStrokes.shift();
    } else if (Math.hypot(point.x - this.activeStroke.points[this.activeStroke.points.length - 1].x, point.y - this.activeStroke.points[this.activeStroke.points.length - 1].y) > 1.4) {
      this.activeStroke.points.push(point);
      if (this.activeStroke.points.length > CONFIG.pointsPerStroke) this.activeStroke.points.splice(0, 100);
    }
  }

  private updateHandLines(hands: TrackedHand[]) {
    const active = new Set<string>();
    for (const hand of hands) {
      if (!hand.indexOnly) continue;
      active.add(hand.id);
      const point = this.smooth(hand.id, hand.indexTip, 1 / 30);
      let stroke = this.handTrails.get(hand.id);
      if (!stroke) {
        stroke = { id: hand.id, points: [], color: handColors[hand.handedness] };
        this.handTrails.set(hand.id, stroke);
      }
      const previous = stroke.points[stroke.points.length - 1];
      if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) > 1.4) stroke.points.push(point);
      if (stroke.points.length > 420) stroke.points.splice(0, stroke.points.length - 420);
    }
    for (const [id, stroke] of this.handTrails) {
      if (!active.has(id)) stroke.points.length = 0;
    }
  }

  private drawCursor(hand: (TrackedHand & { indexTip: Point }) | undefined, width: number, height: number, dt: number) {
    if (!hand) {
      this.cursorTrail.length = 0;
      return;
    }
    // Faster chase when far (responsive), gentle when near (stable)
    const point = this.smooth(hand.id, hand.indexTip, dt);
    this.cursorTrail.push(point);
    if (this.cursorTrail.length > 46) this.cursorTrail.shift();
    this.drawCurve(this.cursorTrail, "#86f7ff", 7, 0.22);
    this.drawCurve(this.cursorTrail, "#86f7ff", 3, 0.88);
    this.drawSparks(this.cursorTrail, "#86f7ff");
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.shadowColor = "#49efff";
    ctx.shadowBlur = 26;
    ctx.fillStyle = "#dbffff";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(118, 245, 255, .75)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 13 + Math.sin(this.animationTime * 5) * 2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = "rgba(208, 253, 255, .8)";
    ctx.fillText("AI CURSOR", clamp(point.x + 18, 8, width - 76), clamp(point.y - 13, 16, height - 6));
    ctx.restore();
  }

  private drawSavedStrokes(width: number, height: number) {
    for (const stroke of this.drawStrokes) {
      this.drawCurve(stroke.points, stroke.color, 9, 0.3);
      this.drawCurve(stroke.points, stroke.color, 3.4, 0.95);
    }
    for (const stroke of this.drawStrokes) this.drawSparks(stroke.points, stroke.color);
    const active = this.activeStroke?.points[this.activeStroke.points.length - 1];
    if (active) {
      this.ctx.save();
      this.ctx.globalCompositeOperation = "lighter";
      this.ctx.fillStyle = "#ffdfff";
      this.ctx.shadowColor = "#e66cff";
      this.ctx.shadowBlur = 20;
      this.ctx.beginPath();
      this.ctx.arc(clamp(active.x, 0, width), clamp(active.y, 0, height), 4, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
    }
  }

  private drawMultiLines(width: number, height: number) {
    for (const stroke of this.handTrails.values()) {
      this.drawCurve(stroke.points, stroke.color, 7, 0.24);
      this.drawCurve(stroke.points, stroke.color, 2.8, 0.92);
      this.drawSparks(stroke.points, stroke.color);
      const tip = stroke.points[stroke.points.length - 1];
      if (!tip) continue;
      this.ctx.save();
      this.ctx.globalCompositeOperation = "lighter";
      this.ctx.fillStyle = "#fff";
      this.ctx.shadowColor = stroke.color;
      this.ctx.shadowBlur = 18;
      this.ctx.beginPath();
      this.ctx.arc(clamp(tip.x, 0, width), clamp(tip.y, 0, height), 4, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
    }
  }

  private drawOrb(hands: RenderHand[], width: number, height: number, dt: number) {
    const validHands = hands.length > 1 && hands[0].open && hands[1].open;
    const left = hands[0];
    const right = hands[1];
    const span = validHands ? Math.hypot(right.palm.x - left.palm.x, right.palm.y - left.palm.y) : 0;
    const active = validHands && span < width * 0.9;
    this.energy = lerp(this.energy, active ? 1 : 0, 1 - Math.exp(-dt * 5));
    if (active) {
      const center = { x: (left.palm.x + right.palm.x) / 2, y: (left.palm.y + right.palm.y) / 2 };
      const targetRadius = clamp(span * 0.25, 34, Math.min(width, height) * 0.22);
      const smooth = 1 - Math.exp(-dt * 8);
      this.orbCenter.x = lerp(this.orbCenter.x || center.x, center.x, smooth);
      this.orbCenter.y = lerp(this.orbCenter.y || center.y, center.y, smooth);
      this.orbRadius = lerp(this.orbRadius || targetRadius, targetRadius, smooth);
      this.updateLightning([left.palm, right.palm]);
    }
    this.updateParticles(active ? [left.palm, right.palm, this.orbCenter] : [], active ? 4 : 0, dt);
    if (this.energy < 0.01) return;

    const ctx = this.ctx;
    const center = this.orbCenter;
    const pulse = 1 + Math.sin(this.animationTime * 3.8) * 0.055;
    const radius = this.orbRadius * pulse;
    ctx.save();
    ctx.globalAlpha = this.energy;
    ctx.globalCompositeOperation = "lighter";
    const halo = ctx.createRadialGradient(center.x, center.y, radius * 0.12, center.x, center.y, radius * 1.55);
    halo.addColorStop(0, "rgba(255, 141, 230, .48)");
    halo.addColorStop(0.45, "rgba(238, 53, 191, .2)");
    halo.addColorStop(1, "rgba(116, 45, 255, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius * 1.55, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = this.showGlow ? radius * 0.42 * CONFIG.glowStrength : 0;
    ctx.shadowColor = "#ff5fda";
    for (let ring = 0; ring < CONFIG.orbRings; ring++) {
      const direction = ring % 2 === 0 ? 1 : -1;
      const phase = this.animationTime * (0.42 + ring * 0.12) * direction;
      ctx.strokeStyle = ring % 2 ? "rgba(233, 135, 255, .72)" : "rgba(255, 174, 237, .88)";
      ctx.lineWidth = ring === 0 ? 1.8 : 1;
      ctx.setLineDash(ring % 2 ? [5, 7] : [12, 5, 2, 6]);
      ctx.beginPath();
      ctx.ellipse(center.x, center.y, radius * (0.57 + ring * 0.14), radius * (0.24 + ring * 0.085), phase, ring * 0.23, Math.PI * 1.78 + ring * 0.23);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (this.lightning.length && this.energy > 0.15) {
      for (const arc of this.lightning) {
        ctx.strokeStyle = "rgba(255, 194, 244, .86)";
        ctx.lineWidth = 1.4;
        ctx.shadowBlur = this.showGlow ? 13 * CONFIG.glowStrength : 0;
        ctx.beginPath();
        ctx.moveTo(arc[0].x, arc[0].y);
        for (let index = 1; index < arc.length; index++) ctx.lineTo(arc[index].x, arc[index].y);
        ctx.stroke();
      }
    }
    const core = ctx.createRadialGradient(center.x - radius * 0.2, center.y - radius * 0.24, 1, center.x, center.y, radius * 0.74);
    core.addColorStop(0, "rgba(255,255,255,.98)");
    core.addColorStop(0.18, "rgba(255,216,249,.96)");
    core.addColorStop(0.48, "rgba(255,103,224,.68)");
    core.addColorStop(1, "rgba(192,45,255,0)");
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius * 0.74, 0, Math.PI * 2);
    ctx.fill();
    this.drawParticles();
    ctx.restore();
  }

  private updateLightning(targets: Point[]) {
    if (++this.lightningFrame % 3 !== 0 && this.lightning.length) return;
    this.lightning = targets.map((target) => {
      const dx = target.x - this.orbCenter.x;
      const dy = target.y - this.orbCenter.y;
      const segments = 10;
      const path: Point[] = [{ ...this.orbCenter }];
      for (let index = 1; index < segments; index++) {
        const ratio = index / segments;
        path.push({
          x: this.orbCenter.x + dx * ratio + (Math.random() - 0.5) * 18,
          y: this.orbCenter.y + dy * ratio + (Math.random() - 0.5) * 18,
        });
      }
      path.push(target);
      return path;
    });
  }

  private updateParticles(sources: Point[], amount: number, dt: number) {
    if (!this.showParticles) {
      this.particles.length = 0;
      return;
    }
    for (let index = this.particles.length - 1; index >= 0; index--) {
      const particle = this.particles[index];
      particle.x += particle.vx * dt;
      particle.y += particle.vy * dt;
      particle.vx *= 0.992;
      particle.vy *= 0.992;
      particle.life -= dt;
      if (particle.life <= 0) this.particles.splice(index, 1);
    }
    if (!sources.length) return;
    for (let index = 0; index < amount && this.particles.length < CONFIG.maxParticles; index++) {
      const source = sources[index % sources.length];
      this.particles.push({
        x: source.x,
        y: source.y,
        vx: (Math.random() - 0.5) * 100,
        vy: (Math.random() - 0.5) * 100,
        life: 0.45 + Math.random() * 0.75,
        maxLife: 1.2,
        hue: 285 + Math.random() * 70,
      });
    }
  }

  private drawParticles() {
    if (!this.showParticles) return;
    const ctx = this.ctx;
    const baseAlpha = ctx.globalAlpha;
    for (const particle of this.particles) {
      ctx.globalAlpha = baseAlpha * clamp(particle.life / particle.maxLife, 0, 1);
      ctx.fillStyle = `hsl(${particle.hue} 100% 82%)`;
      ctx.shadowBlur = this.showGlow ? 8 * CONFIG.glowStrength : 0;
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = baseAlpha;
  }

  private drawPortal(hands: RenderHand[], video: HTMLVideoElement, width: number, height: number, time: number, dt: number) {
    const visibleHands = [...hands].sort((a, b) => a.palm.x - b.palm.x);
    const corners = visibleHands.length >= 2
      ? convexQuad([
          visibleHands[0].landmarks[4],
          visibleHands[0].landmarks[8],
          visibleHands[1].landmarks[8],
          visibleHands[1].landmarks[4],
        ])
      : [];
    const valid = corners.length === 4;
    this.portalOpacity = lerp(this.portalOpacity, valid ? 1 : 0, 1 - Math.exp(-dt / (valid ? 0.12 : CONFIG.portalFadeMs / 1000)));
    if (valid) {
      if (this.portalPoints.length !== 4) this.portalPoints = corners;
      else this.portalPoints = corners.map((point, index) => ({
        x: lerp(this.portalPoints[index].x, point.x, CONFIG.portalSmoothing),
        y: lerp(this.portalPoints[index].y, point.y, CONFIG.portalSmoothing),
      }));
    }
    const open = hands.some((hand) => hand.open);
    if (open) {
      if (!this.openPalmStarted) this.openPalmStarted = time;
      if (time - this.openPalmStarted >= CONFIG.portalDebounceMs &&
          time - this.portalLastSwitch >= CONFIG.portalDebounceMs && !this.portalOpenLatched) {
        this.nextPortalFilter();
        this.portalLastSwitch = time;
        this.portalOpenLatched = true;
      }
    } else {
      this.openPalmStarted = 0;
      this.portalOpenLatched = false;
    }
    if (this.portalOpacity < 0.01 || this.portalPoints.length !== 4) return;

    const sourceWidth = Math.max(1, Math.round(width * 0.46));
    const sourceHeight = Math.max(1, Math.round(height * 0.46));
    if (this.portalSource.width !== sourceWidth || this.portalSource.height !== sourceHeight) {
      this.portalSource.width = sourceWidth;
      this.portalSource.height = sourceHeight;
      this.portalOutput.width = sourceWidth;
      this.portalOutput.height = sourceHeight;
      this.portalLastRender = 0;
    }
    if (time - this.portalLastRender > 55 || !this.portalLastRender) {
      this.portalLastRender = time;
      const sourceContext = this.portalSource.getContext("2d");
      if (sourceContext) {
        sourceContext.save();
        sourceContext.setTransform(-1, 0, 0, 1, sourceWidth, 0);
        const videoAspect = video.videoWidth / Math.max(video.videoHeight, 1);
        const canvasAspect = sourceWidth / sourceHeight;
        const drawWidth = videoAspect > canvasAspect ? sourceHeight * videoAspect : sourceWidth;
        const drawHeight = videoAspect > canvasAspect ? sourceHeight : sourceWidth / videoAspect;
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
          sourceContext.drawImage(video, (sourceWidth - drawWidth) / 2, (sourceHeight - drawHeight) / 2, drawWidth, drawHeight);
        }
        sourceContext.restore();
        FILTROS[this.portalFilterIndex].apply(this.portalOutput, this.portalSource);
      }
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = this.portalOpacity;
    ctx.beginPath();
    ctx.moveTo(this.portalPoints[0].x, this.portalPoints[0].y);
    for (let index = 1; index < this.portalPoints.length; index++) ctx.lineTo(this.portalPoints[index].x, this.portalPoints[index].y);
    ctx.closePath();
    ctx.clip();
    ctx.drawImage(this.portalOutput, 0, 0, width, height);
    if (this.portalFilterIndex === 1) this.drawPerspectiveGrid();
    ctx.restore();
    ctx.save();
    ctx.globalAlpha = this.portalOpacity;
    ctx.strokeStyle = "rgba(255, 255, 255, .88)";
    ctx.lineWidth = CONFIG.portalOutline;
    ctx.shadowColor = "#fff";
    ctx.shadowBlur = this.showGlow ? 7 * CONFIG.glowStrength : 0;
    ctx.beginPath();
    ctx.moveTo(this.portalPoints[0].x, this.portalPoints[0].y);
    for (let index = 1; index < this.portalPoints.length; index++) ctx.lineTo(this.portalPoints[index].x, this.portalPoints[index].y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
    this.drawHud(`${this.currentPortalFilter} · N / OPEN PALM`, 12, height - 64);
    if (open && this.openPalmStarted) {
      const progress = clamp((time - this.openPalmStarted) / CONFIG.portalDebounceMs, 0, 1);
      this.drawHoldRing(hands.find((hand) => hand.open)!.palm, progress);
    }
  }

  private drawPerspectiveGrid() {
    const [topLeft, topRight, bottomRight, bottomLeft] = this.portalPoints;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(247, 252, 255, .72)";
    ctx.lineWidth = 0.8;
    ctx.shadowColor = "#d9f5ff";
    ctx.shadowBlur = 4;
    for (let index = 1; index < 8; index++) {
      const ratio = index / 8;
      const interpolate = (start: Point, end: Point) => ({ x: lerp(start.x, end.x, ratio), y: lerp(start.y, end.y, ratio) });
      const top = interpolate(topLeft, topRight);
      const bottom = interpolate(bottomLeft, bottomRight);
      const left = interpolate(topLeft, bottomLeft);
      const right = interpolate(topRight, bottomRight);
      ctx.beginPath();
      ctx.moveTo(top.x, top.y);
      ctx.lineTo(bottom.x, bottom.y);
      ctx.moveTo(left.x, left.y);
      ctx.lineTo(right.x, right.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawWireframes(hands: RenderHand[], width: number, dt: number) {
    const ctx = this.ctx;
    const orderedHands = [...hands].sort((a, b) => a.handedness.localeCompare(b.handedness));
    const projected: Point[][] = [];
    for (let handIndex = 0; handIndex < orderedHands.length; handIndex++) {
      const hand = orderedHands[handIndex];
      const center = this.smooth(`wire-${hand.id}`, hand.palm, dt);
      const scale = Math.max(18, Math.hypot(hand.landmarks[9].x - hand.landmarks[0].x, hand.landmarks[9].y - hand.landmarks[0].y) * (0.55 + hand.openness * 1.2));
      const angle = this.animationTime * this.rotationSpeed * (handIndex ? -1 : 1);
      const cosY = Math.cos(angle);
      const sinY = Math.sin(angle);
      const cosX = Math.cos(angle * 0.61);
      const sinX = Math.sin(angle * 0.61);
      const points: Point[] = [];
      const depths: number[] = [];
      for (const [x, y, z] of OCTAHEDRON_VERTICES) {
        const rotatedX = x * cosY - z * sinY;
        const rotatedZ = x * sinY + z * cosY;
        const rotatedY = y * cosX - rotatedZ * sinX;
        depths.push(y * sinX + rotatedZ * cosX);
        const perspective = 3.4 / (y * sinX + rotatedZ * cosX + 3.4);
        points.push({ x: center.x + rotatedX * scale * perspective, y: center.y + rotatedY * scale * perspective });
      }
      projected.push(points);
      const minDepth = Math.min(...depths);
      const maxDepth = Math.max(...depths);
      const hueBase = hand.handedness === "Right" ? 330 : 178;
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      for (let index = 0; index < OCTAHEDRON_EDGES.length; index++) {
        const [from, to] = OCTAHEDRON_EDGES[index];
        // Depth shading: far edges fade so the crystal reads as solid 3D.
        const edgeDepth = (depths[from] + depths[to]) / 2;
        const depthNorm = maxDepth > minDepth ? (edgeDepth - minDepth) / (maxDepth - minDepth) : 0;
        ctx.globalAlpha = 1 - depthNorm * 0.55;
        ctx.strokeStyle = `hsl(${(hueBase + index * 29 + this.animationTime * 40) % 360} 100% 69%)`;
        ctx.lineWidth = 1.5 - depthNorm * 0.5;
        ctx.shadowColor = ctx.strokeStyle;
        ctx.shadowBlur = this.showGlow ? 13 * CONFIG.glowStrength : 0;
        ctx.beginPath();
        ctx.moveTo(points[from].x, points[from].y);
        ctx.lineTo(points[to].x, points[to].y);
        ctx.stroke();
      }
      ctx.restore();
    }
    if (projected.length >= 2 && this.showBeams) {
      const handA = orderedHands[0];
      const handB = orderedHands[1];
      const closeness = 1 - clamp(Math.hypot(handA.palm.x - handB.palm.x, handA.palm.y - handB.palm.y) / Math.max(width * 0.8, 1), 0, 1);
      const beamPoints = [4, 8, 12, 16, 20].map((landmarkIndex) => [handA.landmarks[landmarkIndex], handB.landmarks[landmarkIndex]]);
      beamPoints.push([handA.palm, handB.palm]);
      for (const [start, end] of beamPoints) this.drawRainbowBeam(start, end, 1 + closeness * 2);
      const beamMidpoints = beamPoints.map(([start, end]) => ({ x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 }));
      this.updateParticles([handA.palm, handB.palm, ...beamMidpoints], 4, dt);
    }
    const particleSources = orderedHands.length > 1 ? [orderedHands[0].palm, orderedHands[1].palm] : orderedHands.map((hand) => hand.palm);
    if (orderedHands.length < 2 || !this.showBeams) this.updateParticles(particleSources, 1, dt);
    this.drawParticles();
    this.drawHud("G GLOW · R ROTATE · B BEAMS · P SPARKS", 12, 72);
  }

  private drawRainbowBeam(start: Point, end: Point, thickness: number) {
    const ctx = this.ctx;
    const segments = 15;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let index = 0; index < segments; index++) {
      const a = index / segments;
      const b = (index + 1) / segments;
      const color = `hsl(${(this.animationTime * 80 + index * (360 / segments)) % 360} 100% 65%)`;
      ctx.strokeStyle = color;
      ctx.lineWidth = thickness;
      ctx.shadowColor = color;
      ctx.shadowBlur = this.showGlow ? 12 * CONFIG.glowStrength : 0;
      ctx.beginPath();
      ctx.moveTo(lerp(start.x, end.x, a), lerp(start.y, end.y, a));
      ctx.lineTo(lerp(start.x, end.x, b), lerp(start.y, end.y, b));
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,255,255,.88)";
    ctx.lineWidth = 0.8;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.restore();
  }

  private drawHoldRing(point: Point, progress: number) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,.88)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#e783ff";
    ctx.shadowBlur = this.showGlow ? 10 * CONFIG.glowStrength : 0;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 25, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress);
    ctx.stroke();
    ctx.restore();
  }

  private drawHud(text: string, x: number, y: number) {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = "10px ui-monospace, monospace";
    const width = ctx.measureText(text).width + 16;
    ctx.fillStyle = "rgba(8, 10, 16, .68)";
    ctx.fillRect(x, y - 15, width, 23);
    ctx.strokeStyle = "rgba(180, 212, 232, .18)";
    ctx.strokeRect(x, y - 15, width, 23);
    ctx.fillStyle = "rgba(234, 245, 255, .9)";
    ctx.fillText(text, x + 8, y);
    ctx.restore();
  }

  private drawCurve(points: Point[], color: string, lineWidth: number, alpha: number) {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 17;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length - 1; i++) {
      const midpointX = (points[i].x + points[i + 1].x) * 0.5;
      const midpointY = (points[i].y + points[i + 1].y) * 0.5;
      ctx.quadraticCurveTo(points[i].x, points[i].y, midpointX, midpointY);
    }
    const last = points[points.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.stroke();
    ctx.restore();
  }

  private drawSparks(points: Point[], color: string) {
    if (points.length < 4) return;
    const ctx = this.ctx;
    const phase = Math.floor(this.animationTime * 12) % 4;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    for (let i = 0; i < 7; i++) {
      const index = points.length - 1 - i * 4 - phase;
      if (index < 0) break;
      const point = points[index];
      ctx.globalAlpha = Math.max(0.15, 0.78 - i * 0.09);
      ctx.beginPath();
      ctx.arc(point.x, point.y, i === 0 ? 1.8 : 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}
