"""
HAND GESTURE STUDIO: Real-Time Computer Vision & Motion Graphics
================================================================
A modular, high-performance OpenCV + MediaPipe pipeline implementing:
  1. Hand Portal Filter (Perspective-distorted quad with 8 real-time vectorized filters)
  2. Energy Orb & Neon Air Drawing (Bloom lighting, lightning arcs, persistent drawing PIP)
  3. Neon 3D Hand Wireframe (depth-lit glowing bones, fingertip laser beams, sparkle particles)

Supports MediaPipe Tasks (v1.0+) and classic MediaPipe Solutions (v0.10.x).
Optimized for 30+ FPS at 720p using vector NumPy operations.
"""

from __future__ import annotations
import math
import os
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================
CONFIG = {
    # Video & Tracking
    "CAMERA_INDEX": 0,
    "FRAME_WIDTH": 1280,
    "FRAME_HEIGHT": 720,
    "MAX_HANDS": 2,
    "MIN_DETECTION_CONFIDENCE": 0.65,
    "MIN_TRACKING_CONFIDENCE": 0.65,
    "MODEL_PATH": "hand_landmarker.task",

    # Feature 1: Hand Portal Filter
    "PORTAL_SMOOTHING_ALPHA": 0.50,       # EMA weight for corners
    "PORTAL_FADE_DURATION": 0.30,         # Fade out duration in seconds
    "PORTAL_OUTLINE_THICKNESS": 2,        # White border thickness
    "GESTURE_DEBOUNCE_SEC": 1.0,          # Palm hold duration for filter switch
    "PORTAL_GRID_SPACING": 32,            # Perspective-grid cell size in pixels

    # Feature 2: Energy Orb & Air Drawing
    "ORB_MIN_RADIUS": 35,
    "ORB_MAX_RADIUS": 170,
    "ORB_EMA_ALPHA": 0.35,
    "ORB_RING_COUNT": 4,
    "ORB_BLOOM_SMALL_BLUR": (21, 21),
    "ORB_BLOOM_LARGE_BLUR": (51, 51),
    "DRAWING_THICK_BLUR": 9,
    "DRAWING_OUTER_THICKNESS": 7,
    "DRAWING_INNER_THICKNESS": 2,
    "CURSOR_TRAIL_LENGTH": 42,
    "MULTI_TRAIL_LENGTH": 300,
    "TRAIL_MIN_POINT_DISTANCE": 2.0,
    "PIP_WIDTH": 280,
    "PIP_HEIGHT": 158,
    "PIP_MARGIN": 20,

    # Feature 3: Neon 3D Hand Wireframe
    "WIREFRAME_BONE_GLOW": 6,             # width of the soft outer bone stroke (px)
    "WIREFRAME_DEPTH_STRENGTH": 0.35,     # how strongly MediaPipe z scales bones/joints
    "BEAMS_ENABLED": True,
    "PARTICLES_ENABLED": True,
    "MAX_PARTICLES": 300,
    "GLOW_STRENGTH": 1.3,
}

# ==============================================================================
# UNIVERSAL MEDIAPIPE HAND TRACKER (Dual-mode: Tasks v1.0+ & Classic Solutions)
# ==============================================================================
class HandTracker:
    """Detects up to 2 hands across MediaPipe 1.0+ (tasks) and legacy (solutions)."""

    def __init__(self, max_num_hands: int = 2, detection_conf: float = 0.65, tracking_conf: float = 0.65):
        self.max_hands = max_num_hands
        self.det_conf = detection_conf
        self.track_conf = tracking_conf
        self.mode = "none"
        self.detector = None

        # Check for legacy mp.solutions
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.mp_hands = mp.solutions.hands
                self.detector = self.mp_hands.Hands(
                    max_num_hands=max_num_hands,
                    min_detection_confidence=detection_conf,
                    min_tracking_confidence=tracking_conf,
                )
                self.mode = "solutions"
                return
        except Exception:
            pass

        # Check for new mediapipe.tasks.python.vision
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            model_path = CONFIG["MODEL_PATH"]
            if not os.path.exists(model_path):
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                print(f"[HandTracker] Downloading {model_path}...")
                urllib.request.urlretrieve(url, model_path)

            options = vision.HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=model_path),
                num_hands=max_num_hands,
                min_hand_detection_confidence=detection_conf,
                min_hand_presence_confidence=tracking_conf,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.mode = "tasks"
        except Exception as e:
            print(f"[HandTracker] Warning: HandLandmarker initialization error: {e}")
            self.mode = "none"

    def process(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a BGR frame and returns a list of hand dicts:
        [{'landmarks': np.ndarray of shape (21, 2) in pixel coords, 'label': 'Left'|'Right', 'norm': (21, 3)}]
        """
        h, w = frame_bgr.shape[:2]
        hands_data: List[Dict[str, Any]] = []

        if self.mode == "solutions":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    coords = np.array([[lm.x * w, lm.y * h] for lm in hand_lms.landmark], dtype=np.float32)
                    norm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                    label = "Hand"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        label = results.multi_handedness[idx].classification[0].label
                    hands_data.append({"landmarks": coords, "norm": norm, "label": label})

        elif self.mode == "tasks":
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(mp_img)
            if res.hand_landmarks:
                for idx, hand_lms in enumerate(res.hand_landmarks):
                    coords = np.array([[lm.x * w, lm.y * h] for lm in hand_lms], dtype=np.float32)
                    norm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                    label = "Hand"
                    if res.handedness and idx < len(res.handedness):
                        label = res.handedness[idx][0].category_name
                    hands_data.append({"landmarks": coords, "norm": norm, "label": label})

        return hands_data


# ==============================================================================
# SHARED GESTURE UTILITIES
# ==============================================================================
def is_open_palm(landmarks: np.ndarray) -> bool:
    """Checks if all 5 fingers are extended away from the wrist (landmark 0)."""
    wrist = landmarks[0]
    # Landmark indices for tips and PIP joints:
    # Thumb (4 vs 2), Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    tip_dists = np.linalg.norm(landmarks[tips] - wrist, axis=1)
    pip_dists = np.linalg.norm(landmarks[pips] - wrist, axis=1)
    return bool(np.all(tip_dists > pip_dists * 1.15))

def is_index_pointing_only(landmarks: np.ndarray) -> bool:
    """Returns True if index finger is extended and middle/ring/pinky are folded."""
    wrist = landmarks[0]
    index_tip_dist = np.linalg.norm(landmarks[8] - wrist)
    index_pip_dist = np.linalg.norm(landmarks[6] - wrist)
    if index_tip_dist <= index_pip_dist * 1.2:
        return False

    folded_tips = [12, 16, 20]
    folded_pips = [10, 14, 18]
    tip_dists = np.linalg.norm(landmarks[folded_tips] - wrist, axis=1)
    pip_dists = np.linalg.norm(landmarks[folded_pips] - wrist, axis=1)
    return bool(np.all(tip_dists < pip_dists * 1.15))

def is_thumbs_up(landmarks: np.ndarray) -> bool:
    """Checks for a thumbs-up gesture (thumb up/extended, other fingers folded)."""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if thumb_tip[1] > thumb_ip[1]:  # Thumb tip must point upwards in image coords
        return False
    folded_tips = [8, 12, 16, 20]
    folded_pips = [6, 10, 14, 18]
    tip_dists = np.linalg.norm(landmarks[folded_tips] - wrist, axis=1)
    pip_dists = np.linalg.norm(landmarks[folded_pips] - wrist, axis=1)
    return bool(np.all(tip_dists < pip_dists * 1.25))


# ==============================================================================
# VIDEO 1: HAND PORTAL FILTER
# ==============================================================================
def sort_quad_corners(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points consistently as [top-left, top-right, bottom-right, bottom-left]
    using centroid angular sorting to prevent quad twisting.
    """
    pts = np.asarray(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Rotate starting point so index 0 is top-left (minimum sum x+y)
    top_left_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -top_left_idx, axis=0)
    return ordered

# ----------------- 8 VECTORIZED PORTAL FILTERS -----------------
def filter_red_halftone(img: np.ndarray) -> np.ndarray:
    """1. Red Halftone: red/pink tinted live image with a halftone dot overlay."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Keep the live image visible under the pattern, tinted red (bright -> pink, dark -> deep red)
    tinted = np.zeros((h, w, 3), dtype=np.float32)
    tinted[..., 0] = gray * 0.45                            # B channel
    tinted[..., 1] = gray * 0.32                            # G channel
    tinted[..., 2] = np.clip(gray * 1.1 + 12.0, 0, 255)     # R channel boosted

    grid_size = 8
    small = cv2.resize(gray.astype(np.uint8), (max(1, w // grid_size), max(1, h // grid_size)), interpolation=cv2.INTER_AREA)
    small = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    dx = (x_idx % grid_size) - (grid_size / 2.0)
    dy = (y_idx % grid_size) - (grid_size / 2.0)
    dist_sq = dx * dx + dy * dy

    max_r = (grid_size / 2.0) * 0.95
    radius_sq = ((255.0 - small) / 255.0) * (max_r * max_r)
    mask = (dist_sq <= radius_sq).astype(np.float32)

    dot = np.full_like(tinted, (95, 40, 232))    # Deep red dots (BGR)
    out = tinted * (1.0 - mask[..., None] * 0.5) + dot * (mask[..., None] * 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)

def filter_perspective_grid(img: np.ndarray, quad: Optional[np.ndarray] = None) -> np.ndarray:
    """
    2. Perspective Grid: white glowing grid lines over the live image that follow
    the quad's perspective.

    When ``quad`` (the 4 ROI-local corners ordered TL -> TR -> BR -> BL) is supplied,
    a flat grid texture is mapped onto that quad with ``cv2.getPerspectiveTransform``
    and ``cv2.warpPerspective`` so the lines converge exactly with the hands as they
    tilt/skew. Without a quad it falls back to a flat axis-aligned grid.
    """
    h, w = img.shape[:2]
    out = img.copy()
    spacing = CONFIG["PORTAL_GRID_SPACING"]

    # Flat grid texture (one line every `spacing` pixels)
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[::spacing, :] = 255
    grid[:, ::spacing] = 255

    if quad is not None and len(quad) == 4:
        src_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst_quad = np.asarray(quad, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_rect, dst_quad)
        warped = cv2.warpPerspective(grid, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    else:
        warped = grid

    grid_mask = (warped > 0).astype(np.uint8)
    # Soft glow by dilating the grid lines, then crisp white core on top
    grid_glow = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8))
    glow_rows = grid_glow > 0
    out[glow_rows] = cv2.addWeighted(out[glow_rows], 0.3, np.full_like(out[glow_rows], 255), 0.7, 0)
    out[grid_mask > 0] = (255, 255, 255)
    return out

# Opt this filter into receiving the live quad (other filters keep the (img) -> img contract).
filter_perspective_grid.needs_quad = True  # type: ignore[attr-defined]

def filter_pop_art_posterize(img: np.ndarray) -> np.ndarray:
    """3. Pop-art Posterize: high-contrast palette with orange / hot-pink / black."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(img)
    # Band 1: Shadows -> Deep Black
    c0 = np.array([15, 10, 15], dtype=np.uint8)
    # Band 2: Midtones -> Hot Pink
    c1 = np.array([180, 20, 255], dtype=np.uint8)
    # Band 3: Highlights -> Bright Orange
    c2 = np.array([0, 145, 255], dtype=np.uint8)

    m0 = gray < 85
    m1 = (gray >= 85) & (gray < 170)
    m2 = gray >= 170

    out[m0] = c0
    out[m1] = c1
    out[m2] = c2
    return out

def filter_bw_ordered_dither(img: np.ndarray) -> np.ndarray:
    """4. Black & White Halftone / Ordered Dither using a 4x4 Bayer matrix."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bayer4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ], dtype=np.float32) * (255.0 / 16.0)

    h, w = gray.shape
    tiled_bayer = np.tile(bayer4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    dithered = np.where(gray > tiled_bayer, 255, 0).astype(np.uint8)
    return cv2.cvtColor(dithered, cv2.COLOR_GRAY2BGR)

def filter_chromatic_aberration(img: np.ndarray) -> np.ndarray:
    """5. Chromatic Aberration / RGB Split: shifts R and B with slight glitch offset."""
    h, w = img.shape[:2]
    shift = 9
    b, g, r = cv2.split(img)

    # Shift Blue left and Red right
    b_shifted = np.zeros_like(b)
    r_shifted = np.zeros_like(r)
    b_shifted[:, :-shift] = b[:, shift:]
    r_shifted[:, shift:] = r[:, :-shift]

    # Glitch band: horizontal displacement in a random mid strip
    band_y1, band_y2 = int(h * 0.42), int(h * 0.54)
    g_glitch = g.copy()
    g_glitch[band_y1:band_y2, shift:] = g[band_y1:band_y2, :-shift]

    return cv2.merge([b_shifted, g_glitch, r_shifted])

def filter_thermal_heatmap(img: np.ndarray) -> np.ndarray:
    """6. Thermal Heatmap: grayscale -> COLORMAP_JET."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def filter_sepia_vignette(img: np.ndarray) -> np.ndarray:
    """7. Sepia / Vintage warm tone with subtle vignette."""
    h, w = img.shape[:2]
    # Sepia color matrix transformation
    sepia_kernel = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393]
    ], dtype=np.float32)
    sepia = cv2.transform(img, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    # Radial vignette falloff
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    max_d = math.hypot(cx, cy)
    dist = np.hypot(x - cx, y - cy) / max_d
    vignette = np.clip(1.0 - 0.45 * (dist ** 2), 0.0, 1.0).astype(np.float32)

    return (sepia * vignette[..., None]).astype(np.uint8)

def filter_frosted_glass(img: np.ndarray) -> np.ndarray:
    """8. Frosted Glass: strong gaussian blur plus white frosted overlay."""
    blur = cv2.GaussianBlur(img, (35, 35), 0)
    white = np.full_like(img, 245)
    return cv2.addWeighted(blur, 0.60, white, 0.40, 0)

FILTROS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("Red Halftone", filter_red_halftone),
    ("Perspective Grid", filter_perspective_grid),
    ("Pop-Art Posterize", filter_pop_art_posterize),
    ("B&W Ordered Dither", filter_bw_ordered_dither),
    ("Chromatic Aberration", filter_chromatic_aberration),
    ("Thermal Heatmap", filter_thermal_heatmap),
    ("Sepia Vintage Vignette", filter_sepia_vignette),
    ("Frosted Glass", filter_frosted_glass),
]

def render_portal(
    frame: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    filter_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 1.0
) -> np.ndarray:
    """
    Renders the perspective portal inside the quad defined by (p1, p2, p3, p4).
    Fast bounding-rect composite with cv2.fillConvexPoly and white AA outline.
    """
    h, w = frame.shape[:2]
    pts = np.array([p1, p2, p3, p4], dtype=np.int32)

    # 1. Bounding box computation for speed
    xmin = max(0, int(np.min(pts[:, 0])))
    xmax = min(w, int(np.max(pts[:, 0])))
    ymin = max(0, int(np.min(pts[:, 1])))
    ymax = min(h, int(np.max(pts[:, 1])))

    if xmax - xmin < 4 or ymax - ymin < 4:
        return frame

    roi = frame[ymin:ymax, xmin:xmax]

    # 2. ROI-local corners: int32 for the mask, float32 for perspective-aware filters.
    local_pts = pts - np.array([xmin, ymin], dtype=np.int32)
    if getattr(filter_fn, "needs_quad", False):
        filtered_roi = filter_fn(roi, local_pts.astype(np.float32))
    else:
        filtered_roi = filter_fn(roi)

    local_mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
    cv2.fillConvexPoly(local_mask, local_pts, 255)

    # Composite through mask with fade alpha
    mask_3d = (local_mask > 0)[..., None]
    blended_roi = np.where(mask_3d, filtered_roi, roi)

    if alpha < 1.0:
        frame[ymin:ymax, xmin:xmax] = cv2.addWeighted(frame[ymin:ymax, xmin:xmax], 1.0 - alpha, blended_roi, alpha, 0)
    else:
        frame[ymin:ymax, xmin:xmax] = blended_roi

    # 3. Thin white anti-aliased outline
    thickness = CONFIG["PORTAL_OUTLINE_THICKNESS"]
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    return frame


# ==============================================================================
# VIDEO 2: ENERGY ORB & NEON AIR DRAWING
# ==============================================================================
def draw_lightning(layer: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], num_segments: int = 5, jitter: float = 16.0) -> None:
    """Draws jagged flickering electric arc polylines from p1 to p2."""
    p1_arr = np.array(p1, dtype=np.float32)
    p2_arr = np.array(p2, dtype=np.float32)
    diff = p2_arr - p1_arr
    length = np.linalg.norm(diff)
    if length < 10:
        return

    # Normal vector for perpendicular jitter
    norm = np.array([-diff[1], diff[0]]) / length
    points = [p1_arr]
    for i in range(1, num_segments):
        t = i / float(num_segments)
        disp = (np.random.rand() - 0.5) * 2.0 * jitter
        pt = p1_arr + diff * t + norm * disp
        points.append(pt)
    points.append(p2_arr)

    pts_int = np.array(points, dtype=np.int32)
    # Bright cyan/electric violet lightning core
    cv2.polylines(layer, [pts_int], isClosed=False, color=(255, 180, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(layer, [pts_int], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

def draw_orb(
    layer: np.ndarray,
    c1: Tuple[int, int],
    c2: Tuple[int, int],
    t: float,
    center_override: Optional[Tuple[float, float]] = None,
    radius_override: Optional[float] = None,
) -> Tuple[int, int, float]:
    """
    Renders neon energy orb between palm centers c1 and c2 onto the glow layer.
    Features: bright glowing core, oscillating pulse, 4 rotating concentric rings, and lightning arcs.
    """
    p1 = np.array(c1, dtype=np.float32)
    p2 = np.array(c2, dtype=np.float32)
    center = np.asarray(center_override, dtype=np.float32) if center_override is not None else (p1 + p2) * 0.5
    dist = float(np.linalg.norm(p1 - p2))

    # Scale radius with palm distance
    base_radius = np.clip(dist * 0.32, CONFIG["ORB_MIN_RADIUS"], CONFIG["ORB_MAX_RADIUS"])
    # Pulse radius and brightness with sine wave
    pulse = 1.0 + 0.08 * math.sin(7.0 * t)
    radius = (float(radius_override) if radius_override is not None else float(base_radius)) * pulse
    cx, cy = int(center[0]), int(center[1])

    # 1. Concentric rotating rings with dashes
    ring_count = CONFIG["ORB_RING_COUNT"]
    for i in range(ring_count):
        ring_r = int(radius * (0.55 + 0.35 * (i / float(ring_count))))
        rot_speed = (1.5 if i % 2 == 0 else -1.8) * (1.0 + 0.3 * i)
        angle = (t * rot_speed * 50.0) % 360.0

        # Draw segmented dashed elliptical arcs
        for arc_start in range(0, 360, 45):
            cv2.ellipse(
                layer,
                (cx, cy),
                (ring_r, int(ring_r * 0.85)),
                angle,
                arc_start,
                arc_start + 22,
                (255, 80, 210),
                1,
                lineType=cv2.LINE_AA,
            )

    # 2. Glowing core: radial falloff
    core_r = max(6, int(radius * 0.38))
    cv2.circle(layer, (cx, cy), core_r + 8, (240, 60, 180), -1, lineType=cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), core_r, (255, 160, 230), -1, lineType=cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), max(3, int(core_r * 0.5)), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # 3. Electric lightning arcs from core to palms
    draw_lightning(layer, (cx, cy), (int(c1[0]), int(c1[1])), num_segments=6, jitter=18.0)
    draw_lightning(layer, (cx, cy), (int(c2[0]), int(c2[1])), num_segments=6, jitter=18.0)

    return (cx, cy, radius)

def apply_glow(layer: np.ndarray, frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Bloom effect: 2-scale Gaussian blur on black layer + additive blend."""
    blur_sm = cv2.GaussianBlur(layer, CONFIG["ORB_BLOOM_SMALL_BLUR"], 0)
    blur_lg = cv2.GaussianBlur(layer, CONFIG["ORB_BLOOM_LARGE_BLUR"], 0)
    bloom = cv2.addWeighted(blur_sm, 0.6 * strength, blur_lg, 0.7 * strength, 0)
    full_effects = cv2.add(layer, bloom)
    return cv2.add(frame, full_effects)

def update_drawing(
    canvas: np.ndarray,
    finger_pos: Optional[Tuple[int, int]],
    is_drawing: bool,
    stroke_history: List[List[Tuple[int, int]]],
    current_stroke: List[Tuple[int, int]]
) -> None:
    """Updates stroke state and renders glowing violet polylines on the persistent canvas."""
    if is_drawing and finger_pos is not None:
        current_stroke.append(finger_pos)
        if len(current_stroke) >= 2:
            p_prev = current_stroke[-2]
            p_curr = current_stroke[-1]
            # Thick purple outer line + thin bright violet/cyan inner core
            cv2.line(canvas, p_prev, p_curr, (230, 60, 160), CONFIG["DRAWING_OUTER_THICKNESS"], cv2.LINE_AA)
            cv2.line(canvas, p_prev, p_curr, (255, 230, 255), CONFIG["DRAWING_INNER_THICKNESS"], cv2.LINE_AA)
    else:
        if len(current_stroke) > 1:
            stroke_history.append(list(current_stroke))
            current_stroke.clear()


def draw_neon_trail(
    layer: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    cursor: bool = False,
) -> None:
    """Draws a smoothed neon trail and optional cursor reticle on a glow layer."""
    if not points:
        return
    if len(points) > 1:
        line = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(layer, [line], False, color, 8 if cursor else 7, cv2.LINE_AA)
        cv2.polylines(layer, [line], False, (255, 240, 255), 2 if cursor else 2, cv2.LINE_AA)
        for idx, point in enumerate(points[-12:-1]):
            fade = (idx + 1) / 12.0
            tint = tuple(int(channel * fade) for channel in color)
            cv2.circle(layer, point, max(1, int(3 * fade)), tint, -1, cv2.LINE_AA)

    tip = points[-1]
    cv2.circle(layer, tip, 7 if cursor else 5, color, -1, cv2.LINE_AA)
    cv2.circle(layer, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)
    if cursor:
        cv2.circle(layer, tip, 16, color, 1, cv2.LINE_AA)


# ==============================================================================
# VIDEO 3: NEON WIREFRAME HANDS WITH LASER BEAMS & SPARKLES
# ==============================================================================
OCTAHEDRON_VERTICES: List[Tuple[float, float, float]] = [
    (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
]

OCTAHEDRON_EDGES: List[Tuple[int, int]] = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (4, 3), (3, 5), (5, 2),
]

def project_wireframe(
    vertices_3d: List[Tuple[float, float, float]],
    edges: List[Tuple[int, int]],
    center_xy: Tuple[float, float],
    scale: float,
    rot_y: float,
    rot_x: float,
    perspective: float = 3.4,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Projects a 3D wireframe (vertices + edges) into 2D screen segments.
    Rotates around Y then X, applies perspective projection, and
    positions / scales the result around ``center_xy``.

    Returns one segment per input edge: ``[((x1, y1), (x2, y2)), ...]``.
    """
    cos_y = math.cos(rot_y)
    sin_y = math.sin(rot_y)
    cos_x = math.cos(rot_x)
    sin_x = math.sin(rot_x)

    projected: List[Tuple[float, float]] = []
    for (x, y, z) in vertices_3d:
        rx = x * cos_y - z * sin_y
        rz = x * sin_y + z * cos_y
        ry = y * cos_x - rz * sin_x
        depth = y * sin_x + rz * cos_x
        p = perspective / (depth + perspective)
        px = center_xy[0] + rx * scale * p
        py = center_xy[1] + ry * scale * p
        projected.append((px, py))

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for (a, b) in edges:
        segments.append((projected[a], projected[b]))
    return segments


# MediaPipe 21-landmark hand skeleton: bones connecting the joints (no solid shape).
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

def hue_to_bgr(hue: float, sat: int = 255, val: int = 255) -> Tuple[int, int, int]:
    """Converts an OpenCV hue (0-179) to a BGR tuple for neon line colours."""
    hsv = np.uint8([[[int(hue) % 180, sat, val]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(b), int(g), int(r))

def landmark_depth_weights(norm: np.ndarray) -> np.ndarray:
    """
    Maps MediaPipe landmark z (relative depth; negative = nearer the camera) to a
    per-landmark weight. Nearer joints get a larger weight so they are drawn thicker
    and brighter than far ones, which is what sells the 3D wireframe look.
    """
    z = norm[:, 2].astype(np.float32)
    centered = z - float(np.median(z))
    spread = float(np.std(z)) + 1e-4
    strength = CONFIG["WIREFRAME_DEPTH_STRENGTH"]
    return np.clip(1.0 - (centered / spread) * strength, 0.55, 1.5)

def draw_hand_wireframe(
    layer: np.ndarray,
    landmarks_px: np.ndarray,
    norm: np.ndarray,
    hue_offset: float = 0.0,
) -> None:
    """
    Draws a neon 3D hand skeleton onto ``layer`` (a BGR glow buffer):
    depth-scaled glowing bones plus joint nodes. Bones are drawn far-first so nearer
    ones overlap them, reinforcing the sense of depth. Bloom is added later by apply_glow.
    """
    depth = landmark_depth_weights(norm)
    glow_w = CONFIG["WIREFRAME_BONE_GLOW"]

    # Painter's algorithm: far bones (small depth) before near bones.
    bones = sorted(HAND_CONNECTIONS, key=lambda edge: depth[edge[0]] + depth[edge[1]])
    for a, b in bones:
        pa = (int(landmarks_px[a, 0]), int(landmarks_px[a, 1]))
        pb = (int(landmarks_px[b, 0]), int(landmarks_px[b, 1]))
        d = 0.5 * float(depth[a] + depth[b])
        color = hue_to_bgr(hue_offset + a * 6.0 + b * 3.0)
        # Wide soft outer stroke (later bloomed) + bright white core.
        cv2.line(layer, pa, pb, color, max(2, int(glow_w * d)), cv2.LINE_AA)
        cv2.line(layer, pa, pb, (255, 255, 255), max(1, int(1.6 * d)), cv2.LINE_AA)

    # Joint nodes: glowing dots sized by depth, drawn on top of the bones.
    for i in range(len(landmarks_px)):
        cx, cy = int(landmarks_px[i, 0]), int(landmarks_px[i, 1])
        radius = max(2, int(4.5 * float(depth[i])))
        cv2.circle(layer, (cx, cy), radius + 2, hue_to_bgr(hue_offset + i * 8.0), -1, cv2.LINE_AA)
        cv2.circle(layer, (cx, cy), max(1, radius - 1), (255, 255, 255), -1, cv2.LINE_AA)

def draw_beams(
    layer: np.ndarray,
    hand_a: np.ndarray,
    hand_b: np.ndarray,
    t: float,
    particle_emitter: Callable[[float, float, Tuple[int, int, int]], None]
) -> None:
    """
    Renders multicolored laser beams connecting corresponding fingertips (4, 8, 12, 16, 20)
    and palm centers between both hands, with bright cores and sparkle emission.
    """
    # 5 fingertip indices + palm center (computed average of 0, 5, 9, 13, 17)
    fingertip_ids = [4, 8, 12, 16, 20]
    palmA_center = np.mean(hand_a[[0, 5, 9, 13, 17]], axis=0)
    palmB_center = np.mean(hand_b[[0, 5, 9, 13, 17]], axis=0)

    pairs = [(palmA_center, palmB_center)]
    for fid in fingertip_ids:
        pairs.append((hand_a[fid], hand_b[fid]))

    dist = np.linalg.norm(palmA_center - palmB_center)
    intensity = np.clip(1.4 - (dist / 700.0), 0.5, 1.8)

    seg_count = 14
    for idx, (ptA, ptB) in enumerate(pairs):
        a_arr = np.asarray(ptA, dtype=np.float32)
        b_arr = np.asarray(ptB, dtype=np.float32)

        beam_thickness = max(2, int(4 * intensity))
        # Rainbow gradient along the beam: small segments with cycling hues
        for s in range(seg_count):
            t0 = s / seg_count
            t1 = (s + 1) / seg_count
            seg_a = (int(a_arr[0] + (b_arr[0] - a_arr[0]) * t0), int(a_arr[1] + (b_arr[1] - a_arr[1]) * t0))
            seg_b = (int(a_arr[0] + (b_arr[0] - a_arr[0]) * t1), int(a_arr[1] + (b_arr[1] - a_arr[1]) * t1))
            hue = int((t * 60.0 + idx * 40.0 + s * (180.0 / seg_count)) % 180)
            hsv_color = np.uint8([[[hue, 240, 255]]])
            bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.line(layer, seg_a, seg_b, color, beam_thickness, cv2.LINE_AA)

        # Bright white beam core
        cv2.line(layer, (int(ptA[0]), int(ptA[1])), (int(ptB[0]), int(ptB[1])), (255, 255, 255), max(1, beam_thickness - 2), cv2.LINE_AA)

        # Emit sparkles along beam
        if np.random.rand() < 0.35:
            interp = np.random.rand()
            emit_x = ptA[0] * interp + ptB[0] * (1.0 - interp)
            emit_y = ptA[1] * interp + ptB[1] * (1.0 - interp)
            particle_emitter(emit_x, emit_y, color)

class ParticleSystem:
    """300-capacity particle system with floating velocity and glowing decay."""
    def __init__(self, max_particles: int = 300):
        self.max_particles = max_particles
        self.particles: List[Dict[str, Any]] = []

    def emit(self, x: float, y: float, color: Tuple[int, int, int]) -> None:
        if len(self.particles) >= self.max_particles:
            return
        angle = np.random.uniform(0, 2.0 * math.pi)
        speed = np.random.uniform(1.0, 4.5)
        self.particles.append({
            "x": float(x),
            "y": float(y),
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed - 0.8,
            "life": 1.0,
            "decay": np.random.uniform(0.02, 0.05),
            "color": color,
            "size": np.random.randint(2, 5),
        })

    def update_and_draw(self, layer: np.ndarray) -> None:
        surviving: List[Dict[str, Any]] = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= p["decay"]
            if p["life"] > 0:
                surviving.append(p)
                r = max(1, int(p["size"] * p["life"]))
                alpha = p["life"]
                c = (int(p["color"][0] * alpha), int(p["color"][1] * alpha), int(p["color"][2] * alpha))
                cv2.circle(layer, (int(p["x"]), int(p["y"])), r, c, -1, lineType=cv2.LINE_AA)
                cv2.circle(layer, (int(p["x"]), int(p["y"])), max(1, r // 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        self.particles = surviving


# ==============================================================================
# ON-SCREEN HUD / CONTROL PANELS
# ==============================================================================
def draw_hud(
    frame: np.ndarray,
    mode_name: str,
    filter_name: str,
    fps: float,
    status_text: str = "",
    extra_settings: Optional[Dict[str, Any]] = None
) -> None:
    """Renders sleek glassmorphic HUD in the corners."""
    h, w = frame.shape[:2]

    # Mode and Filter HUD (Top-Left)
    hud_w, hud_h = 320, 115 if extra_settings is None else 180
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (120, 80, 240), 1, cv2.LINE_AA)

    cv2.putText(frame, f"MODE: {mode_name}", (28, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active: {filter_name}", (28, 68), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:0.1f} | Key: 'm' Switch Mode", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 200), 1, cv2.LINE_AA)

    if status_text:
        cv2.putText(frame, status_text, (28, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 140), 1, cv2.LINE_AA)

    if extra_settings:
        y_pos = 135
        for k, v in extra_settings.items():
            cv2.putText(frame, f"{k}: {v}", (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 210, 140), 1, cv2.LINE_AA)
            y_pos += 20


# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================
def main():
    cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])

    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return

    tracker = HandTracker(
        max_num_hands=CONFIG["MAX_HANDS"],
        detection_conf=CONFIG["MIN_DETECTION_CONFIDENCE"],
        tracking_conf=CONFIG["MIN_TRACKING_CONFIDENCE"]
    )

    # State Variables
    app_modes = [
        "1: Cursor Effects",
        "2: Hand Portal Filter",
        "3: Energy Orb",
        "4: Neon Air Drawing",
        "5: Multi-Hand Lines",
        "6: Neon Wireframe Hands",
    ]
    current_mode_idx = 0
    filtro_idx = 0

    # Video 1: Portal State
    smoothed_portal_pts: Optional[np.ndarray] = None
    portal_alpha = 0.0
    last_palm_switch_time = 0.0

    # Video 2: Orb & Drawing State
    smoothed_orb_pos = np.array([CONFIG["FRAME_WIDTH"] // 2, CONFIG["FRAME_HEIGHT"] // 2], dtype=np.float32)
    smoothed_orb_radius = 60.0
    orb_alpha = 0.0
    stroke_canvas = np.zeros((CONFIG["FRAME_HEIGHT"], CONFIG["FRAME_WIDTH"], 3), dtype=np.uint8)
    stroke_history: List[List[Tuple[int, int]]] = []
    current_stroke: List[Tuple[int, int]] = []
    palm_clear_start_time: Optional[float] = None
    smoothed_tips: Dict[str, np.ndarray] = {}
    cursor_trail: List[Tuple[int, int]] = []
    hand_trails: Dict[str, List[Tuple[int, int]]] = {}
    thumbs_up_started: Optional[float] = None
    thumbs_up_latched = False
    orb_palms = np.array([
        [CONFIG["FRAME_WIDTH"] * 0.4, CONFIG["FRAME_HEIGHT"] * 0.5],
        [CONFIG["FRAME_WIDTH"] * 0.6, CONFIG["FRAME_HEIGHT"] * 0.5],
    ], dtype=np.float32)

    # Video 3: Neon 3D Hand Wireframe State
    particles = ParticleSystem(CONFIG["MAX_PARTICLES"])
    glow_mult = CONFIG["GLOW_STRENGTH"]
    beams_active = CONFIG["BEAMS_ENABLED"]
    particles_active = CONFIG["PARTICLES_ENABLED"]
    rotation_speed = 0.9

    prev_time = time.time()
    fps = 30.0

    print("==================================================================")
    print("Hand Gesture Studio Started!")
    print("Hotkeys:")
    print("  'm': Cycle App Modes (Cursor -> Portal -> Orb -> Draw -> Multi-Hand -> Wireframe)")
    print("  Hold thumbs-up for 0.85s: Switch to the next effect")
    print("  'n': Cycle Portal Filters (Video 1)")
    print("  'c': Clear Drawing Canvas (Video 2)")
    print("  'u': Undo Last Stroke (Video 2)")
    print("  'g'/'G': Adjust Glow (Video 3)")
    print("  'r': Change Rotation Speed (Video 3)")
    print("  'b': Toggle Laser Beams (Video 3)")
    print("  'p': Toggle Particles (Video 3)")
    print("  'q' or ESC: Quit")
    print("==================================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame horizontally for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        dt = max(0.001, now - prev_time)
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Detect hands
        hands = tracker.process(frame)

        if any(is_thumbs_up(hand["landmarks"]) for hand in hands):
            if thumbs_up_started is None:
                thumbs_up_started = now
            elif now - thumbs_up_started >= 0.85 and not thumbs_up_latched:
                current_mode_idx = (current_mode_idx + 1) % len(app_modes)
                thumbs_up_latched = True
        else:
            thumbs_up_started = None
            thumbs_up_latched = False

        mode_name = app_modes[current_mode_idx]
        status_msg = ""

        # ======================================================================
        # MODE 1: CURSOR EFFECTS
        # ======================================================================
        if current_mode_idx == 0:
            active_hand = next((hand for hand in hands if is_index_pointing_only(hand["landmarks"])), None)
            glow_layer = np.zeros_like(frame)
            if active_hand is not None:
                label = active_hand["label"]
                target = active_hand["landmarks"][8]
                previous = smoothed_tips.get(label, target)
                smooth_alpha = 1.0 - math.exp(-dt * 15.0)
                point = previous + smooth_alpha * (target - previous)
                smoothed_tips[label] = point
                cursor_trail.append((int(point[0]), int(point[1])))
                cursor_trail = cursor_trail[-CONFIG["CURSOR_TRAIL_LENGTH"]:]
                draw_neon_trail(glow_layer, cursor_trail, (255, 220, 105), cursor=True)
                cv2.putText(glow_layer, "AI CURSOR", (int(point[0]) + 20, int(point[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 250, 255), 1, cv2.LINE_AA)
            else:
                cursor_trail.clear()
            frame = apply_glow(glow_layer, frame, strength=1.0)
            draw_hud(frame, mode_name, "Index-finger cursor", fps, status_text="Point with index | 'm' changes effect")

        # ======================================================================
        # MODE 2: HAND PORTAL FILTER
        # ======================================================================
        elif current_mode_idx == 1:
            filter_name, filter_fn = FILTROS[filtro_idx]

            # Gesture switch: Open palm debounce
            for hand in hands:
                if is_open_palm(hand["landmarks"]):
                    if now - last_palm_switch_time > CONFIG["GESTURE_DEBOUNCE_SEC"]:
                        filtro_idx = (filtro_idx + 1) % len(FILTROS)
                        last_palm_switch_time = now
                        break

            # 2 hands required for portal
            if len(hands) >= 2:
                # Sort hands left-to-right by wrist x
                hands_sorted = sorted(hands, key=lambda h_dict: h_dict["landmarks"][0][0])
                left_hand = hands_sorted[0]["landmarks"]
                right_hand = hands_sorted[1]["landmarks"]

                # Corners: left thumb (4), left index (8), right index (8), right thumb (4)
                raw_corners = np.array([
                    left_hand[4],
                    left_hand[8],
                    right_hand[8],
                    right_hand[4]
                ], dtype=np.float32)

                ordered_corners = sort_quad_corners(raw_corners)

                if smoothed_portal_pts is None:
                    smoothed_portal_pts = ordered_corners.copy()
                else:
                    alpha_smooth = CONFIG["PORTAL_SMOOTHING_ALPHA"]
                    smoothed_portal_pts = alpha_smooth * ordered_corners + (1.0 - alpha_smooth) * smoothed_portal_pts

                portal_alpha = min(1.0, portal_alpha + (dt / 0.15))
            else:
                # Fade out smoothly over PORTAL_FADE_DURATION
                portal_alpha = max(0.0, portal_alpha - (dt / CONFIG["PORTAL_FADE_DURATION"]))

            if smoothed_portal_pts is not None and portal_alpha > 0.01:
                frame = render_portal(
                    frame,
                    smoothed_portal_pts[0],
                    smoothed_portal_pts[1],
                    smoothed_portal_pts[2],
                    smoothed_portal_pts[3],
                    filter_fn,
                    alpha=portal_alpha
                )

            draw_hud(frame, mode_name, filter_name, fps, status_text="Open palm or 'n' to cycle filter")

        # ======================================================================
        # MODE 3: ENERGY ORB
        # ======================================================================
        elif current_mode_idx == 2:
            glow_layer = np.zeros_like(frame)
            orb_active = len(hands) >= 2 and is_open_palm(hands[0]["landmarks"]) and is_open_palm(hands[1]["landmarks"])
            if orb_active:
                palm1 = np.mean(hands[0]["landmarks"][[0, 5, 9, 13, 17]], axis=0)
                palm2 = np.mean(hands[1]["landmarks"][[0, 5, 9, 13, 17]], axis=0)
                target_center = (palm1 + palm2) * 0.5
                target_radius = np.clip(np.linalg.norm(palm1 - palm2) * 0.32, CONFIG["ORB_MIN_RADIUS"], CONFIG["ORB_MAX_RADIUS"])
                alpha = CONFIG["ORB_EMA_ALPHA"]
                smoothed_orb_pos = alpha * target_center + (1.0 - alpha) * smoothed_orb_pos
                smoothed_orb_radius = float(alpha * target_radius + (1.0 - alpha) * smoothed_orb_radius)
                orb_palms = alpha * np.array([palm1, palm2], dtype=np.float32) + (1.0 - alpha) * orb_palms
                orb_alpha = min(1.0, orb_alpha + dt / 0.2)
            else:
                orb_alpha = max(0.0, orb_alpha - dt / 0.3)

            if orb_alpha > 0.01:
                c1 = (int(orb_palms[0, 0]), int(orb_palms[0, 1]))
                c2 = (int(orb_palms[1, 0]), int(orb_palms[1, 1]))
                draw_orb(glow_layer, c1, c2, now,
                         center_override=(float(smoothed_orb_pos[0]), float(smoothed_orb_pos[1])),
                         radius_override=smoothed_orb_radius)
                frame = apply_glow(glow_layer, frame, strength=1.2 * orb_alpha)
            draw_hud(frame, mode_name, "Glowing pulsed energy field", fps, status_text="Hold two open palms | Spread to resize")

        # ======================================================================
        # MODE 4: NEON AIR DRAWING
        # ======================================================================
        elif current_mode_idx == 3:
            dark_base = np.full_like(frame, (16, 12, 22))
            is_drawing = False
            draw_pos = None
            for hand in hands:
                lms = hand["landmarks"]
                if is_open_palm(lms):
                    if palm_clear_start_time is None:
                        palm_clear_start_time = now
                    elif now - palm_clear_start_time >= 1.0:
                        stroke_canvas.fill(0)
                        stroke_history.clear()
                        current_stroke.clear()
                        palm_clear_start_time = None
                else:
                    palm_clear_start_time = None
                if is_index_pointing_only(lms):
                    is_drawing = True
                    draw_pos = (int(lms[8][0]), int(lms[8][1]))
                    break
            if not is_drawing:
                palm_clear_start_time = None

            update_drawing(stroke_canvas, draw_pos, is_drawing, stroke_history, current_stroke)
            blurred_canvas = cv2.GaussianBlur(
                stroke_canvas, (CONFIG["DRAWING_THICK_BLUR"], CONFIG["DRAWING_THICK_BLUR"]), 0
            )
            art_view = cv2.add(dark_base, cv2.add(stroke_canvas, blurred_canvas))
            if palm_clear_start_time is not None:
                hold_progress = min(1.0, (now - palm_clear_start_time) / 1.0)
                cv2.ellipse(art_view, (w // 2, h // 2), (40, 40), 0, 0,
                            int(hold_progress * 360), (0, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(art_view, "CLEARING...", (w // 2 - 50, h // 2 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            pip_w = min(CONFIG["PIP_WIDTH"], max(1, w // 3))
            pip_h = min(CONFIG["PIP_HEIGHT"], max(1, h // 3))
            pip_x = max(0, w - pip_w - CONFIG["PIP_MARGIN"])
            pip_y = CONFIG["PIP_MARGIN"]
            pip_small = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
            art_view[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_small
            cv2.rectangle(art_view, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (200, 100, 255), 2, cv2.LINE_AA)
            cv2.putText(art_view, "WEBCAM PIP", (pip_x + 10, pip_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            frame = art_view
            draw_hud(frame, mode_name, "Violet polyline canvas", fps,
                     status_text="Index draws | Palm 1s or 'c' clears | 'u' undoes")

        # ======================================================================
        # MODE 5: MULTI-HAND COLORED LINES
        # ======================================================================
        elif current_mode_idx == 4:
            glow_layer = np.zeros_like(frame)
            active_labels = set()
            colors = {"Left": (255, 210, 80), "Right": (210, 70, 255), "Hand": (180, 240, 90)}
            for hand in hands:
                if not is_index_pointing_only(hand["landmarks"]):
                    continue
                label = hand["label"]
                active_labels.add(label)
                target = hand["landmarks"][8]
                previous = smoothed_tips.get(label, target)
                smooth_alpha = 1.0 - math.exp(-dt * 12.0)
                point = previous + smooth_alpha * (target - previous)
                smoothed_tips[label] = point
                trail = hand_trails.setdefault(label, [])
                pixel = (int(point[0]), int(point[1]))
                if not trail or math.dist(trail[-1], pixel) >= CONFIG["TRAIL_MIN_POINT_DISTANCE"]:
                    trail.append(pixel)
                del trail[:-CONFIG["MULTI_TRAIL_LENGTH"]]
                draw_neon_trail(glow_layer, trail, colors.get(label, colors["Hand"]))
            for label in list(hand_trails):
                if label not in active_labels:
                    hand_trails.pop(label, None)
            frame = apply_glow(glow_layer, frame, strength=1.1)
            draw_hud(frame, mode_name, "Independent left/right neon trails", fps,
                     status_text="Raise each index finger to draw with both hands")

        # ======================================================================
        # MODE 6: NEON WIREFRAME HANDS WITH LASER BEAMS & SPARKLES
        # ======================================================================
        elif current_mode_idx == 5:
            glow_layer = np.zeros_like(frame)

            # Neon 3D hand-skeleton wireframe (no solid shape): glowing bones + joint nodes.
            for idx, hand in enumerate(hands):
                lms = hand["landmarks"]
                norm = hand["norm"]
                base_hue = (now * 60.0 + idx * 90.0) % 180.0
                draw_hand_wireframe(glow_layer, lms, norm, hue_offset=base_hue)

                # Rotating 3D octahedron crystal anchored at the palm.
                palm_center = np.mean(lms[[0, 5, 9, 13, 17]], axis=0)
                hand_scale = float(np.linalg.norm(lms[9] - lms[0]))
                crystal_scale = max(18.0, hand_scale * (0.55 + float(np.linalg.norm(lms[4] - lms[20])) * 1.2 / max(hand_scale, 1e-4)))
                angle = now * rotation_speed * (-1.0 if idx == 1 else 1.0)
                segments = project_wireframe(
                    OCTAHEDRON_VERTICES,
                    OCTAHEDRON_EDGES,
                    (float(palm_center[0]), float(palm_center[1])),
                    crystal_scale,
                    angle,
                    angle * 0.61,
                )
                crystal_hue_base = 330.0 if hand.get("label") == "Right" else 178.0
                for si, (pA, pB) in enumerate(segments):
                    seg_hue = int((crystal_hue_base + si * 29.0 + now * 40.0) % 180.0)
                    color = hue_to_bgr(seg_hue)
                    cv2.line(
                        glow_layer,
                        (int(pA[0]), int(pA[1])),
                        (int(pB[0]), int(pB[1])),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        glow_layer,
                        (int(pA[0]), int(pA[1])),
                        (int(pB[0]), int(pB[1])),
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # Emit sparkles from fingertips, coloured to match the hand.
                if particles_active:
                    for fid in [4, 8, 12, 16, 20]:
                        if np.random.rand() < 0.28:
                            particles.emit(lms[fid][0], lms[fid][1], hue_to_bgr(base_hue + fid * 8.0))

            # Draw laser beams between hands if both hands are detected
            if len(hands) >= 2 and beams_active:
                draw_beams(
                    glow_layer,
                    hands[0]["landmarks"],
                    hands[1]["landmarks"],
                    now,
                    particle_emitter=particles.emit if particles_active else lambda x, y, c: None
                )

            # Update and draw sparkle particles
            if particles_active:
                particles.update_and_draw(glow_layer)

            # Apply bloom lighting (two-scale glow keeps the wireframe luminous and 3D).
            frame = apply_glow(glow_layer, frame, strength=glow_mult)

            extra_hud = {
                "Glow (g/G)": f"{glow_mult:0.1f}",
                "Rotation (r)": f"{rotation_speed:0.1f}",
                "Laser Beams (b)": "ON" if beams_active else "OFF",
                "Sparkles (p)": f"ON ({len(particles.particles)}/300)" if particles_active else "OFF",
            }
            draw_hud(frame, mode_name, "Octa crystals + hand skeleton + beams", fps, extra_settings=extra_hud)

        cv2.imshow("Hand Gesture Studio - OpenCV + MediaPipe", frame)

        # Keyboard interaction
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        elif key == ord("m"):
            current_mode_idx = (current_mode_idx + 1) % len(app_modes)
        elif key == ord("n"):
            filtro_idx = (filtro_idx + 1) % len(FILTROS)
        elif key == ord("c"):
            stroke_canvas.fill(0)
            stroke_history.clear()
            current_stroke.clear()
        elif key == ord("u"):
            if stroke_history:
                stroke_history.pop()
                stroke_canvas.fill(0)
                for s in stroke_history:
                    for i in range(1, len(s)):
                        cv2.line(stroke_canvas, s[i-1], s[i], (230, 60, 160), CONFIG["DRAWING_OUTER_THICKNESS"], cv2.LINE_AA)
                        cv2.line(stroke_canvas, s[i-1], s[i], (255, 230, 255), CONFIG["DRAWING_INNER_THICKNESS"], cv2.LINE_AA)
        elif key == ord("g"):
            glow_mult = max(0.4, glow_mult - 0.2)
        elif key == ord("G"):
            glow_mult = min(2.5, glow_mult + 0.2)
        elif key == ord("r"):
            rotation_speed = 0.3 if rotation_speed >= 2.4 else rotation_speed + 0.3
        elif key == ord("b"):
            beams_active = not beams_active
        elif key == ord("p"):
            particles_active = not particles_active

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
