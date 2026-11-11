import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";

export interface Point {
  x: number;
  y: number;
}

export interface TrackedHand {
  id: string;
  handedness: "Left" | "Right" | "Unknown";
  landmarks: Point[];
  indexTip: Point;
  palm: Point;
  indexOnly: boolean;
  pinching: boolean;
  open: boolean;
  palmFacing: boolean;
  thumbsUp: boolean;
  fingersExtended: [boolean, boolean, boolean, boolean, boolean];
  handScale: number;
  openness: number;
  confidence: number;
}

const fingertipIndices = [8, 12, 16, 20];
const pipIndices = [6, 10, 14, 18];
const PALM_INDICES = [0, 5, 9, 13, 17];

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function classifyHands(result: HandLandmarkerResult): TrackedHand[] {
  return result.landmarks.map((rawLandmarks, index) => {
    const landmarks = rawLandmarks.map(({ x, y, z }) => ({ x: 1 - x, y, z: z ?? 0 }));
    const wrist = landmarks[0];

    // Non-thumb fingers, orientation-invariant. Two tests, either passes:
    //  - byReach: tip clearly farther from the wrist than the PIP joint
    //    (works when the finger projects lengthwise on screen)
    //  - byDepth: tip-pip separation along MediaPipe's z axis (works when
    //    fingers point toward/away from the camera and collapse in 2D -
    //    an extended finger juts in depth, a folded one curls inside the
    //    palm plane). Threshold scales with the palm's own depth span.
    const palmDepthSpan = Math.abs(landmarks[9].z - wrist.z);
    const depthThreshold = Math.max(0.5 * palmDepthSpan, 0.03);
    const nonThumbExtended = fingertipIndices.map((tipIndex, fingerIndex) => {
      const tip = landmarks[tipIndex];
      const pip = landmarks[pipIndices[fingerIndex]];
      const byReach = distance(tip, wrist) > distance(pip, wrist) * 1.12;
      const byDepth = Math.abs(tip.z - pip.z) > depthThreshold;
      return byReach || byDepth;
    });

    const palm = {
      x: PALM_INDICES.reduce((sum, i) => sum + landmarks[i].x, 0) / PALM_INDICES.length,
      y: PALM_INDICES.reduce((sum, i) => sum + landmarks[i].y, 0) / PALM_INDICES.length,
    };
    const indexTip = landmarks[8];

    const reportedHandedness = result.handedness[index]?.[0]?.categoryName;
    const handedness = reportedHandedness === "Left" ? "Right" : reportedHandedness === "Right" ? "Left" : "Unknown";
    const confidence = result.handedness[index]?.[0]?.score ?? 0;

    // Robust thumb test: the thumb is extended when its tip spreads away from
    // the palm - farther from the pinky knuckle than the IP joint is. This is
    // rotation-tolerant, unlike tip-vs-joint length comparisons.
    const thumbTip = landmarks[4];
    const thumbIp = landmarks[3];
    const pinkyMcp = landmarks[17];
    const thumbExtended = distance(thumbTip, pinkyMcp) > distance(thumbIp, pinkyMcp) * 1.08;

    const fingersExtended: [boolean, boolean, boolean, boolean, boolean] = [
      thumbExtended,
      nonThumbExtended[0],
      nonThumbExtended[1],
      nonThumbExtended[2],
      nonThumbExtended[3],
    ];

    // Palm facing the camera: palm width (index knuckle to pinky knuckle)
    // relative to palm length (wrist to middle knuckle). Edge-on hands
    // collapse in width; palm-facing hands keep a ratio near 0.7-0.85.
    const handScale = distance(wrist, landmarks[9]);
    const palmWidth = distance(landmarks[5], landmarks[17]);
    const palmFacing = palmWidth / Math.max(handScale, 1e-5) > 0.58;

    return {
      id: handedness === "Left" || handedness === "Right" ? handedness : `hand-${index}`,
      handedness,
      landmarks: landmarks.map(({ x, y }) => ({ x, y })),
      indexTip,
      palm,
      // Index-only ignores the thumb: drawing should not break because the
      // thumb drifts slightly away from the fist.
      indexOnly: nonThumbExtended[0] && nonThumbExtended.slice(1).every((extended) => !extended),
      // Pinch distance is relative to hand scale so it works at any distance.
      pinching: distance(thumbTip, indexTip) < handScale * 0.5,
      // Open hand = the four fingers extended; the thumb may stay tucked,
      // and any hand orientation is accepted (palm, back, or angled).
      open: nonThumbExtended.every(Boolean),
      palmFacing,
      thumbsUp: thumbExtended && nonThumbExtended.every((extended) => !extended) && thumbTip.y < thumbIp.y,
      fingersExtended,
      handScale,
      openness: distance(thumbTip, landmarks[20]),
      confidence,
    };
  });
}

interface DebouncedState {
  current: boolean;
  pending: boolean;
  pendingCount: number;
}

interface HandStability {
  seen: number;
  missing: number;
  states: Map<string, DebouncedState>;
}

/**
 * Temporal gesture stabilizer. MediaPipe landmarks jitter frame to frame,
 * which makes single-frame booleans flicker and resets hold-to-activate
 * timers (portal filter switch, draw-to-clear). The stabilizer requires a
 * boolean to be observed for `confirmFrames` consecutive frames before it
 * flips, and purges state for hands that leave the camera view.
 */
export class GestureStabilizer {
  private hands = new Map<string, HandStability>();

  constructor(
    private confirmFrames = 2,
    private purgeAfter = 45,
  ) {}

  private debounce(hand: HandStability, key: string, raw: boolean, confirm: number) {
    let state = hand.states.get(key);
    if (!state) {
      state = { current: raw, pending: raw, pendingCount: 0 };
      hand.states.set(key, state);
      return state.current;
    }
    if (raw === state.current) {
      state.pending = raw;
      state.pendingCount = 0;
      return state.current;
    }
    if (raw !== state.pending) {
      state.pending = raw;
      state.pendingCount = 1;
    } else {
      state.pendingCount++;
    }
    if (state.pendingCount >= confirm) {
      state.current = raw;
      state.pendingCount = 0;
    }
    return state.current;
  }

  /** Returns the same hands with flicker-free open/palmFacing/indexOnly/pinching/thumbsUp. */
  update(hands: TrackedHand[]): TrackedHand[] {
    for (const hand of hands) {
      let entry = this.hands.get(hand.id);
      if (!entry) {
        entry = { seen: 0, missing: 0, states: new Map() };
        this.hands.set(hand.id, entry);
      }
      entry.seen++;
      entry.missing = 0;
      hand.open = this.debounce(entry, "open", hand.open, this.confirmFrames);
      hand.palmFacing = this.debounce(entry, "palmFacing", hand.palmFacing, this.confirmFrames);
      hand.thumbsUp = this.debounce(entry, "thumbsUp", hand.thumbsUp, this.confirmFrames);
      // Fast-response gestures: immediate classification, no added latency.
      hand.indexOnly = this.debounce(entry, "indexOnly", hand.indexOnly, 1);
      hand.pinching = this.debounce(entry, "pinching", hand.pinching, 1);
    }
    for (const [id, entry] of this.hands) {
      const present = hands.some((hand) => hand.id === id);
      if (present) continue;
      entry.missing++;
      if (entry.missing > this.purgeAfter) this.hands.delete(id);
    }
    return hands;
  }
}

export function gestureLabel(effect: string, hands: TrackedHand[]) {
  if (!hands.length) return "Camera live · tracking";
  if (effect === "cursor") {
    return hands.some((hand) => hand.indexOnly) ? "Index cursor active" : "Raise index · curl other fingers";
  }
  if (effect === "draw") {
    if (hands[0].open) return "Hold open hand 1s to clear";
    return hands[0].indexOnly ? "Drawing" : "Raise index to draw";
  }
  if (effect === "energy") {
    return hands.length > 1 && hands[0].open && hands[1].open
      ? "Orb linked · move hands to resize"
      : "Show two open hands to form orb";
  }
  if (effect === "portal") return hands.length > 1 ? "Portal linked · hold open palm to switch filter" : "Place both hands to form portal";
  if (effect === "wireframe") return hands.length > 1 ? "Crystals linked · move hands to bend beams" : "Wireframe follows each visible palm";
  return hands.length > 1 ? "Both lines active" : "Show both hands for two lines";
}
