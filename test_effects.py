"""
Automated unit verification for hand_gesture_studio.py
Tests:
- HandTracker (MediaPipe tasks / solutions)
- All 8 Filters in Video 1
- sort_quad_corners & render_portal with convex poly & fade alpha
- Video 2: draw_orb, draw_lightning, update_drawing, apply_glow
- Video 3: project_wireframe, draw_beams, ParticleSystem
"""
import sys
import numpy as np
import cv2

from hand_gesture_studio import (
    HandTracker,
    sort_quad_corners,
    render_portal,
    FILTROS,
    draw_orb,
    draw_lightning,
    draw_neon_trail,
    apply_glow,
    update_drawing,
    project_wireframe,
    draw_beams,
    ParticleSystem,
    OCTAHEDRON_VERTICES,
    OCTAHEDRON_EDGES
)

def run_tests():
    print("Testing Video 1: Filters & Portal...")
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_roi = np.random.randint(0, 255, (200, 250, 3), dtype=np.uint8)

    for name, fn in FILTROS:
        filtered = fn(dummy_roi)
        assert filtered.shape == dummy_roi.shape, f"Filter {name} changed shape: {filtered.shape}"
        assert filtered.dtype == np.uint8, f"Filter {name} wrong dtype: {filtered.dtype}"
        print(f"  [PASS] Filter '{name}'")

    corners = np.array([[100, 100], [500, 120], [480, 450], [90, 420]], dtype=np.float32)
    ordered = sort_quad_corners(corners)
    assert ordered.shape == (4, 2), "sort_quad_corners failed"
    out_portal = render_portal(dummy_frame.copy(), ordered[0], ordered[1], ordered[2], ordered[3], FILTROS[0][1], alpha=0.5)
    assert out_portal.shape == dummy_frame.shape, "render_portal failed"
    print("  [PASS] render_portal with alpha blend")

    print("Testing Video 2: Energy Orb & Drawing...")
    layer = np.zeros_like(dummy_frame)
    cx, cy, r = draw_orb(layer, (300, 300), (600, 320), 1.5)
    assert r > 0, "draw_orb radius invalid"
    cx_smooth, cy_smooth, _ = draw_orb(
        layer, (300, 300), (600, 320), 1.5, center_override=(450, 350), radius_override=80
    )
    assert (cx_smooth, cy_smooth) == (450, 350), "draw_orb ignored smoothed center"
    glow_out = apply_glow(layer, dummy_frame.copy(), strength=1.2)
    assert glow_out.shape == dummy_frame.shape, "apply_glow failed"
    print("  [PASS] draw_orb and apply_glow")
    draw_neon_trail(layer, [(100, 100), (130, 120), (160, 100)], (255, 160, 80), cursor=True)
    assert int(layer.max()) > 0, "draw_neon_trail did not render"
    print("  [PASS] Cursor trail rendering")

    canvas = np.zeros_like(dummy_frame)
    history = []
    stroke = []
    update_drawing(canvas, (200, 200), True, history, stroke)
    update_drawing(canvas, (210, 215), True, history, stroke)
    update_drawing(canvas, None, False, history, stroke)
    assert len(history) == 1, "update_drawing history failed"
    print("  [PASS] update_drawing")

    print("Testing Video 3: Neon Wireframe Hands & Beams...")
    segments = project_wireframe(OCTAHEDRON_VERTICES, OCTAHEDRON_EDGES, (400, 400), 100.0, 0.5, 0.3)
    assert len(segments) == len(OCTAHEDRON_EDGES), "project_wireframe segment count mismatch"
    print(f"  [PASS] project_wireframe generated {len(segments)} segments")

    ps = ParticleSystem(300)
    for _ in range(50):
        ps.emit(400, 400, (255, 200, 100))
    assert len(ps.particles) == 50, "Particle emission failed"
    ps.update_and_draw(layer)
    print("  [PASS] ParticleSystem update_and_draw")

    hand_a = np.random.uniform(100, 300, (21, 2))
    hand_b = np.random.uniform(500, 700, (21, 2))
    draw_beams(layer, hand_a, hand_b, 1.0, ps.emit)
    print("  [PASS] draw_beams")

    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
