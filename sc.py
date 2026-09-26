#!/usr/bin/env python3
"""
SMART CITY — Animated Computer Graphics Simulation
B.Tech Computer Graphics competition project (single-file).

Run:
    python smart_city.py

Requires: Python 3.10+ and pygame (pygame-ce is fine).
    pip install pygame

Controls:
    SPACE  Pause / Resume
    N      Day / Night
    A      Ambulance emergency
    P      Parking overlay
    T      Traffic overlay
    R      Reset
    + / -  Animation speed
    ESC    Exit

File layout (competition / viva):
    1. Constants            2. CG algorithms
    3. Transformations      4. Classes
    5. City initialization  6. Animation / update
    7. Collision / traffic  8. Rendering
    9. Event handling       10. Main loop
"""

from __future__ import annotations

import bisect
import math
import random
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import pygame


# =============================================================================
# 1. COORDINATE SYSTEM & CONSTANTS
# =============================================================================

WIDTH, HEIGHT = 1600, 900
HUD_X = 1280
CITY_W, CITY_H = HUD_X, HEIGHT
FPS = 60

# Road geometry (screen space, origin top-left, +x right, +y down)
H_ROAD_Y = 430
H_ROAD_H = 88
V_ROAD_X = 600
V_ROAD_W = 88

LANE_EB_Y = H_ROAD_Y + 66  # eastbound (left -> right), south half
LANE_WB_Y = H_ROAD_Y + 22  # westbound (right -> left), north half
LANE_SB_X = V_ROAD_X + 66  # southbound (top -> bottom), east half
LANE_NB_X = V_ROAD_X + 22  # northbound (bottom -> top), west half

INTER_X = V_ROAD_X
INTER_Y = H_ROAD_Y
INTER_W = V_ROAD_W
INTER_H = H_ROAD_H

STOP_MARGIN = 18


Color = Tuple[int, int, int]
Vec2 = Tuple[float, float]
Mat3 = List[List[float]]


class LightState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class TimeMode(Enum):
    DAY = "DAY"
    NIGHT = "NIGHT"


# Local-space geometry is defined ONCE (not rebuilt every frame).
CAR_BODY: Sequence[Vec2] = ((-18, -8), (16, -8), (20, -4), (20, 4), (16, 8), (-18, 8), (-20, 4), (-20, -4))
BUS_BODY: Sequence[Vec2] = ((-28, -10), (26, -10), (30, -5), (30, 5), (26, 10), (-28, 10), (-30, 5), (-30, -5))
PARK_BODY: Sequence[Vec2] = ((-14, -8), (14, -8), (16, -4), (16, 4), (14, 8), (-14, 8))
CAR_WHEELS: Sequence[Vec2] = ((-12, -9), (10, -9), (-12, 9), (10, 9))
BUS_WHEELS: Sequence[Vec2] = ((-18, -9), (16, -9), (-18, 9), (16, 9))
PLANE_BODY: Sequence[Vec2] = ((18, 0), (8, -5), (-16, -4), (-20, 0), (-16, 4), (8, 5))
PLANE_WING: Sequence[Vec2] = ((-2, 0), (4, -16), (8, -16), (2, 0), (8, 16), (4, 16))
PLANE_TAIL: Sequence[Vec2] = ((-16, 0), (-20, -10), (-14, -10), (-12, 0))
HEADLIGHT: Sequence[Vec2] = ((18, -2), (55, -10), (55, 10), (18, 2))
PED_BODY: Sequence[Vec2] = ((0, 0), (6, 0))

BIN_SITES = ((120, 400), (520, 400), (760, 400), (1200, 400), (520, 620), (760, 620))
LAMP_GLOW = ((90, H_ROAD_Y - 32), (270, H_ROAD_Y - 32), (470, H_ROAD_Y - 32),
             (770, H_ROAD_Y - 32), (970, H_ROAD_Y - 32), (1190, H_ROAD_Y - 32))

ROUTE_AFTER = {"EB_NB": "NB", "SB_EB": "EB", "WB_SB": "SB", "NB_WB": "WB"}
EW_PATHS = {"EB", "WB", "EB_NB", "PARK_IN", "AMB", "AMB_RET"}
NS_PATHS = {"SB", "NB", "SB_EB", "WB_SB", "NB_WB"}
NS_CONFLICT = {"SB", "NB", "SB_EB", "NB_WB", "WB_SB"}

# Shared UI (created once after pygame.init).
FONTS: Dict[str, pygame.font.Font] = {}
SPRITES: Dict[str, pygame.Surface] = {}


# =============================================================================
# 2. CLASSICAL CG ALGORITHMS
#    Pixel-level DDA / Bresenham / Midpoint — not pygame.draw.line/circle
#    for the primitive itself. pygame is only the framebuffer + window.
# =============================================================================

INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8


def _code(x: float, y: float, xmin: float, ymin: float, xmax: float, ymax: float) -> int:
    c = INSIDE
    if x < xmin:
        c |= LEFT
    elif x > xmax:
        c |= RIGHT
    if y < ymin:
        c |= TOP
    elif y > ymax:
        c |= BOTTOM
    return c


def cohen_sutherland_clip(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float = 0, ymin: float = 0, xmax: float = CITY_W - 1, ymax: float = CITY_H - 1,
) -> Optional[Tuple[int, int, int, int]]:
    """Clip a line to the city viewport (Cohen–Sutherland)."""
    c1 = _code(x1, y1, xmin, ymin, xmax, ymax)
    c2 = _code(x2, y2, xmin, ymin, xmax, ymax)
    while True:
        if not (c1 | c2):
            return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if c1 & c2:
            return None
        c = c1 if c1 else c2
        if c & TOP:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1 + 1e-9)
            y = ymin
        elif c & BOTTOM:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1 + 1e-9)
            y = ymax
        elif c & RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1 + 1e-9)
            x = xmax
        else:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1 + 1e-9)
            x = xmin
        if c == c1:
            x1, y1 = x, y
            c1 = _code(x1, y1, xmin, ymin, xmax, ymax)
        else:
            x2, y2 = x, y
            c2 = _code(x2, y2, xmin, ymin, xmax, ymax)


def plot(surface: pygame.Surface, x: int, y: int, color: Color) -> None:
    if 0 <= x < surface.get_width() and 0 <= y < surface.get_height():
        surface.set_at((x, y), color)


def dda_line(surface: pygame.Surface, x1: float, y1: float, x2: float, y2: float, color: Color, thick: int = 1) -> None:
    """Digital Differential Analyzer line drawing."""
    clipped = cohen_sutherland_clip(x1, y1, x2, y2, 0, 0, surface.get_width() - 1, surface.get_height() - 1)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    dx = x2 - x1
    dy = y2 - y1
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        plot(surface, x1, y1, color)
        return
    x_inc = dx / steps
    y_inc = dy / steps
    x, y = float(x1), float(y1)
    for _ in range(steps + 1):
        ix, iy = int(round(x)), int(round(y))
        if thick <= 1:
            plot(surface, ix, iy, color)
        else:
            r = thick // 2
            for oy in range(-r, r + 1):
                for ox in range(-r, r + 1):
                    plot(surface, ix + ox, iy + oy, color)
        x += x_inc
        y += y_inc


def bresenham_line(surface: pygame.Surface, x1: float, y1: float, x2: float, y2: float, color: Color, thick: int = 1) -> None:
    """Bresenham integer line drawing."""
    clipped = cohen_sutherland_clip(x1, y1, x2, y2, 0, 0, surface.get_width() - 1, surface.get_height() - 1)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    r = max(0, thick // 2)
    while True:
        if r == 0:
            plot(surface, x1, y1, color)
        else:
            for oy in range(-r, r + 1):
                for ox in range(-r, r + 1):
                    plot(surface, x1 + ox, y1 + oy, color)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def midpoint_circle(surface: pygame.Surface, xc: int, yc: int, radius: int, color: Color, filled: bool = False) -> None:
    """Midpoint circle algorithm (optional scan-fill)."""
    if radius <= 0:
        plot(surface, xc, yc, color)
        return
    x, y = 0, radius
    p = 1 - radius

    def octants(px: int, py: int) -> None:
        pts = (
            (xc + px, yc + py), (xc - px, yc + py),
            (xc + px, yc - py), (xc - px, yc - py),
            (xc + py, yc + px), (xc - py, yc + px),
            (xc + py, yc - px), (xc - py, yc - px),
        )
        if filled:
            # Scan-convert using the SAME midpoint (x,y) pairs (horizontal spans).
            pygame.draw.line(surface, color, (xc - px, yc + py), (xc + px, yc + py))
            pygame.draw.line(surface, color, (xc - px, yc - py), (xc + px, yc - py))
            pygame.draw.line(surface, color, (xc - py, yc + px), (xc + py, yc + px))
            pygame.draw.line(surface, color, (xc - py, yc - px), (xc + py, yc - px))
        else:
            for a, b in pts:
                plot(surface, a, b, color)

    octants(x, y)
    while x < y:
        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1
        octants(x, y)


def midpoint_ellipse(surface: pygame.Surface, xc: int, yc: int, rx: int, ry: int, color: Color, filled: bool = False) -> None:
    """Midpoint ellipse algorithm (optional scan-fill)."""
    if rx <= 0 or ry <= 0:
        return
    x, y = 0, ry
    rx2, ry2 = rx * rx, ry * ry
    p1 = ry2 - rx2 * ry + 0.25 * rx2

    def plot_sym(px: int, py: int) -> None:
        if filled:
            pygame.draw.line(surface, color, (xc - px, yc + py), (xc + px, yc + py))
            pygame.draw.line(surface, color, (xc - px, yc - py), (xc + px, yc - py))
        else:
            plot(surface, xc + px, yc + py, color)
            plot(surface, xc - px, yc + py, color)
            plot(surface, xc + px, yc - py, color)
            plot(surface, xc - px, yc - py, color)

    plot_sym(x, y)
    while 2 * ry2 * x < 2 * rx2 * y:
        x += 1
        if p1 < 0:
            p1 += 2 * ry2 * x + ry2
        else:
            y -= 1
            p1 += 2 * ry2 * x - 2 * rx2 * y + ry2
        plot_sym(x, y)

    p2 = ry2 * (x + 0.5) ** 2 + rx2 * (y - 1) ** 2 - rx2 * ry2
    while y > 0:
        y -= 1
        if p2 > 0:
            p2 += -2 * rx2 * y + rx2
        else:
            x += 1
            p2 += 2 * ry2 * x - 2 * rx2 * y + rx2
        plot_sym(x, y)


# =============================================================================
# 3. MATHEMATICAL TRANSFORMATIONS (2D homogeneous coordinates)
#    Point (x, y, 1).  Composite pose: T(tx,ty) · R(θ)
#    x' = x cosθ − y sinθ + tx
#    y' = x sinθ + y cosθ + ty
# =============================================================================

def identity() -> Mat3:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def pose_matrix(tx: float, ty: float, theta: float) -> Mat3:
    """Single 3×3 for translation after rotation (avoids extra matrix multiplies)."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]]


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    r = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return r


def translate(tx: float, ty: float) -> Mat3:
    m = identity()
    m[0][2] = tx
    m[1][2] = ty
    return m


def rotate(theta: float) -> Mat3:
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def scale(sx: float, sy: float) -> Mat3:
    return [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]


def reflect(axis: str) -> Mat3:
    if axis == "x":
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    if axis == "y":
        return [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]


def compose(*mats: Mat3) -> Mat3:
    """Composite transform applied right-to-left: compose(T, R, S) = T·R·S."""
    m = identity()
    for x in mats:
        m = mat_mul(m, x)
    return m


def apply_transform(m: Mat3, x: float, y: float) -> Vec2:
    """Apply 2D homogeneous transform to a point."""
    xp = m[0][0] * x + m[0][1] * y + m[0][2]
    yp = m[1][0] * x + m[1][1] * y + m[1][2]
    wp = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(wp) > 1e-9:
        xp /= wp
        yp /= wp
    return xp, yp


def transform_points(m: Mat3, pts: Sequence[Vec2]) -> List[Vec2]:
    return [apply_transform(m, p[0], p[1]) for p in pts]


def aabb_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


# =============================================================================
# 3. PALETTES & DRAWING HELPERS
# =============================================================================

DAY = {
    "sky_top": (135, 196, 245),
    "sky_bot": (214, 236, 196),
    "grass": (102, 168, 92),
    "grass_dark": (82, 142, 74),
    "asphalt": (62, 66, 72),
    "asphalt_edge": (48, 52, 58),
    "sidewalk": (186, 186, 178),
    "curb": (150, 150, 144),
    "lane": (240, 240, 230),
    "median": (232, 210, 70),
    "building_shadow": (40, 55, 40),
    "window_day": (196, 224, 236),
    "text": (28, 32, 36),
    "hud_bg": (24, 32, 44),
}

NIGHT = {
    "sky_top": (12, 18, 42),
    "sky_bot": (28, 46, 38),
    "grass": (34, 62, 40),
    "grass_dark": (24, 48, 32),
    "asphalt": (32, 34, 40),
    "asphalt_edge": (22, 24, 28),
    "sidewalk": (86, 86, 92),
    "curb": (60, 60, 66),
    "lane": (210, 210, 190),
    "median": (180, 160, 40),
    "building_shadow": (8, 10, 16),
    "window_day": (255, 214, 110),
    "text": (230, 234, 240),
    "hud_bg": (10, 14, 22),
}


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def fill_rect_scan(surface: pygame.Surface, x: int, y: int, w: int, h: int, color: Color) -> None:
    """Axis-aligned fill (scan conversion). Outlines still use Bresenham/DDA."""
    pygame.draw.rect(surface, color, pygame.Rect(int(x), int(y), int(w), int(h)))


def fill_poly(surface: pygame.Surface, pts: Sequence[Vec2], color: Color) -> None:
    """Filled polygon (scan conversion). Call outline_poly for Bresenham edges."""
    ip = [(int(round(p[0])), int(round(p[1]))) for p in pts]
    if len(ip) >= 3:
        pygame.draw.polygon(surface, color, ip)


def outline_poly(surface: pygame.Surface, pts: Sequence[Vec2], color: Color, thick: int = 1) -> None:
    ip = [(int(round(p[0])), int(round(p[1]))) for p in pts]
    for i in range(len(ip)):
        x1, y1 = ip[i]
        x2, y2 = ip[(i + 1) % len(ip)]
        bresenham_line(surface, x1, y1, x2, y2, color, thick)


def sky_gradient(surface: pygame.Surface, pal: dict, height: int = 110) -> None:
    """Sky ramp via DDA scanlines (limited height — full-frame set_at is too slow)."""
    h = max(1, min(height, surface.get_height()))
    for y in range(h):
        t = y / max(1, h - 1)
        c = lerp_color(pal["sky_top"], pal["sky_bot"], t)
        pygame.draw.line(surface, c, (0, y), (surface.get_width() - 1, y))


# =============================================================================
# 4. PATHS (no random wandering)
# =============================================================================

class Path:
    """Polyline path with arc-length parameterization."""

    def __init__(self, points: Sequence[Vec2], loop: bool = True, name: str = ""):
        self.name = name
        self.loop = loop
        self.points = [(float(x), float(y)) for x, y in points]
        self.seg_len: List[float] = []
        self.cum: List[float] = [0.0]
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            d = math.hypot(x2 - x1, y2 - y1)
            self.seg_len.append(d)
            self.cum.append(self.cum[-1] + d)
        self.length = self.cum[-1] if self.cum else 0.0
        # Arc-length of first entry into the intersection (computed once).
        self.inter_s: Optional[float] = None

    def sample(self, s: float) -> Tuple[float, float, float]:
        if self.length <= 1e-6:
            x, y = self.points[0]
            return x, y, 0.0
        if self.loop:
            s %= self.length
        else:
            s = max(0.0, min(self.length, s))
        i = bisect.bisect_right(self.cum, s) - 1
        i = max(0, min(i, len(self.seg_len) - 1))
        seg = self.seg_len[i] if self.seg_len[i] > 1e-6 else 1e-6
        t = (s - self.cum[i]) / seg
        x1, y1 = self.points[i]
        x2, y2 = self.points[i + 1]
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        heading = math.atan2(y2 - y1, x2 - x1)
        return x, y, heading

    def dist_to_intersection(self, s: float) -> float:
        """Remaining distance along the path to the stop/intersection region."""
        if self.inter_s is None:
            return 1e9
        if self.loop:
            return (self.inter_s - (s % self.length)) % self.length
        d = self.inter_s - s
        return d if d > 0.0 else 1e9

    def remaining_to_xband(self, s: float, *_args) -> float:
        """Compatibility helper: stop-line distance uses the precomputed entry s."""
        return self.dist_to_intersection(s)


def make_city_paths() -> dict:
    """Deterministic lane paths. Vehicles wrap so traffic is continuous."""
    eb = Path([(-80, LANE_EB_Y), (CITY_W + 80, LANE_EB_Y)], True, "EB")
    wb = Path([(CITY_W + 80, LANE_WB_Y), (-80, LANE_WB_Y)], True, "WB")
    sb = Path([(LANE_SB_X, -80), (LANE_SB_X, CITY_H + 80)], True, "SB")
    nb = Path([(LANE_NB_X, CITY_H + 80), (LANE_NB_X, -80)], True, "NB")

    # Turning routes through the intersection (smooth polylines).
    eb_nb = Path([
        (-80, LANE_EB_Y),
        (V_ROAD_X - 10, LANE_EB_Y),
        (LANE_NB_X, H_ROAD_Y + H_ROAD_H - 8),
        (LANE_NB_X, -80),
    ], False, "EB_NB")
    sb_eb = Path([
        (LANE_SB_X, -80),
        (LANE_SB_X, H_ROAD_Y - 10),
        (V_ROAD_X + V_ROAD_W - 8, LANE_EB_Y),
        (CITY_W + 80, LANE_EB_Y),
    ], False, "SB_EB")
    wb_sb = Path([
        (CITY_W + 80, LANE_WB_Y),
        (V_ROAD_X + V_ROAD_W + 10, LANE_WB_Y),
        (LANE_SB_X, H_ROAD_Y + 8),
        (LANE_SB_X, CITY_H + 80),
    ], False, "WB_SB")
    nb_wb = Path([
        (LANE_NB_X, CITY_H + 80),
        (LANE_NB_X, H_ROAD_Y + H_ROAD_H + 10),
        (V_ROAD_X + 8, LANE_WB_Y),
        (-80, LANE_WB_Y),
    ], False, "NB_WB")

    hospital = Path([
        (1120, 720),
        (1120, LANE_WB_Y),
        (V_ROAD_X + V_ROAD_W + 20, LANE_WB_Y),
        (LANE_NB_X, H_ROAD_Y + 8),
        (LANE_NB_X, 120),
        (900, 120),
    ], False, "AMB")
    hospital_return = Path([
        (900, 120),
        (LANE_SB_X, 120),
        (LANE_SB_X, LANE_EB_Y),
        (1120, LANE_EB_Y),
        (1120, 720),
    ], False, "AMB_RET")

    park_in = Path([
        (40, LANE_EB_Y),
        (160, LANE_EB_Y),
        (160, 700),
    ], False, "PARK_IN")

    paths = {
        "EB": eb, "WB": wb, "SB": sb, "NB": nb,
        "EB_NB": eb_nb, "SB_EB": sb_eb, "WB_SB": wb_sb, "NB_WB": nb_wb,
        "AMB": hospital, "AMB_RET": hospital_return, "PARK_IN": park_in,
    }
    for path in paths.values():
        path.inter_s = _first_intersection_s(path)
    return paths


def _first_intersection_s(path: Path) -> Optional[float]:
    """Precompute arc-length of first intersection entry (O(length) once)."""
    step = 6.0
    s = 0.0
    x0, y0, x1, y1 = INTER_X, INTER_Y, INTER_X + INTER_W, INTER_Y + INTER_H
    while s <= path.length:
        x, y, _ = path.sample(s)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return s
        s += step
    return None


# =============================================================================
# 5. CITY FEATURES (STATIC + SMART OBJECTS)
# =============================================================================

def draw_windows(surface: pygame.Surface, x: int, y: int, w: int, h: int, pal: dict, night: bool, cols: int = 4, rows: int = 3) -> None:
    pad_x, pad_y = 8, 10
    gap_x, gap_y = 6, 8
    ww = max(4, (w - 2 * pad_x - (cols - 1) * gap_x) // cols)
    hh = max(4, (h - 2 * pad_y - (rows - 1) * gap_y) // rows)
    for r in range(rows):
        for c in range(cols):
            wx = x + pad_x + c * (ww + gap_x)
            wy = y + pad_y + r * (hh + gap_y)
            lit = night and ((r + c + x + y) % 5 != 1)
            col = pal["window_day"] if (not night or lit) else (28, 36, 52)
            fill_rect_scan(surface, wx, wy, ww, hh, col)
            bresenham_line(surface, wx, wy, wx + ww, wy, (30, 30, 40))
            bresenham_line(surface, wx, wy + hh, wx + ww, wy + hh, (30, 30, 40))
            bresenham_line(surface, wx, wy, wx, wy + hh, (30, 30, 40))
            bresenham_line(surface, wx + ww, wy, wx + ww, wy + hh, (30, 30, 40))


def draw_building(
    surface: pygame.Surface, x: int, y: int, w: int, h: int,
    wall: Color, roof: Color, pal: dict, night: bool,
    depth: int = 14, label: str = "", window_cols: int = 5, window_rows: int = 4,
) -> None:
    fill_rect_scan(surface, x + 8, y + 8, w, h, pal["building_shadow"])
    # Isometric top (translation + implicit scale of the footprint)
    top = [(x + depth, y - depth), (x + w + depth, y - depth), (x + w, y), (x, y)]
    fill_poly(surface, top, roof)
    outline_poly(surface, top, (40, 40, 50))
    side = [(x + w, y), (x + w + depth, y - depth), (x + w + depth, y + h - depth), (x + w, y + h)]
    fill_poly(surface, side, lerp_color(wall, (0, 0, 0), 0.18))
    outline_poly(surface, side, (40, 40, 50))
    fill_rect_scan(surface, x, y, w, h, wall)
    bresenham_line(surface, x, y, x + w, y, (30, 30, 40), 2)
    bresenham_line(surface, x, y + h, x + w, y + h, (30, 30, 40), 2)
    bresenham_line(surface, x, y, x, y + h, (30, 30, 40), 2)
    bresenham_line(surface, x + w, y, x + w, y + h, (30, 30, 40), 2)
    draw_windows(surface, x, y, w, h, pal, night, window_cols, window_rows)
    if label and "label14" in FONTS:
        surface.blit(FONTS["label14"].render(label, True, (255, 255, 255)), (x + 8, y + h - 22))


def draw_solar_array(surface: pygame.Surface, x: int, y: int, cols: int, rows: int, night: bool) -> None:
    cell_w, cell_h = 14, 9
    panel = (28, 54, 96) if not night else (16, 28, 48)
    hi = (90, 160, 210)
    for r in range(rows):
        for c in range(cols):
            px = x + c * (cell_w + 3)
            py = y + r * (cell_h + 3)
            fill_rect_scan(surface, px, py, cell_w, cell_h, panel)
            dda_line(surface, px, py, px + cell_w, py, hi)
            bresenham_line(surface, px, py, px, py + cell_h, (10, 16, 28))
            bresenham_line(surface, px + cell_w, py, px + cell_w, py + cell_h, (10, 16, 28))


def draw_tree(surface: pygame.Surface, x: int, y: int, night: bool) -> None:
    trunk = (96, 62, 36) if not night else (60, 40, 24)
    leaf = (46, 130, 62) if not night else (22, 70, 38)
    fill_rect_scan(surface, x - 3, y, 6, 16, trunk)
    midpoint_circle(surface, x, y - 6, 12, leaf, filled=True)
    midpoint_circle(surface, x - 7, y - 2, 8, lerp_color(leaf, (20, 80, 30), 0.2), filled=True)
    midpoint_circle(surface, x + 7, y - 2, 8, lerp_color(leaf, (80, 180, 80), 0.15), filled=True)


def draw_dustbin(surface: pygame.Surface, x: int, y: int, fill: float, night: bool) -> None:
    body = (52, 130, 86) if not night else (28, 80, 54)
    fill_rect_scan(surface, x, y, 16, 22, body)
    bresenham_line(surface, x, y, x + 16, y, (20, 40, 30), 2)
    midpoint_ellipse(surface, x + 8, y, 8, 3, (30, 40, 32), filled=True)
    fh = max(1, int(18 * fill))
    fill_rect_scan(surface, x + 3, y + 20 - fh, 10, fh, (210, 200, 70))
    lab = SPRITES.get("bin_label")
    if lab:
        surface.blit(lab, (x - 2, y + 22))


def draw_streetlamp(surface: pygame.Surface, x: int, y: int, night: bool) -> None:
    bresenham_line(surface, x, y, x, y - 28, (70, 74, 80), 2)
    dda_line(surface, x, y - 28, x + 10, y - 28, (70, 74, 80), 2)
    midpoint_circle(surface, x + 10, y - 28, 4, (255, 230, 140) if night else (200, 200, 190), filled=True)


class Road:
    """Arterial roads: DDA edges, Bresenham medians/stop-lines, dashed lanes."""

    def paint(self, surface: pygame.Surface, pal: dict) -> None:
        paint_roads(surface, pal)


class Building:
    def __init__(
        self, x: int, y: int, w: int, h: int, wall: Color, roof: Color, label: str,
        cols: int = 5, rows: int = 4, depth: int = 12,
    ):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.wall, self.roof, self.label = wall, roof, label
        self.cols, self.rows, self.depth = cols, rows, depth

    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        draw_building(
            surface, self.x, self.y, self.w, self.h, self.wall, self.roof, pal, night,
            self.depth, self.label, self.cols, self.rows,
        )


class School(Building):
    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        super().draw(surface, pal, night)
        midpoint_circle(surface, self.x + self.w + 40, self.y + self.h // 2, 36,
                        (80, 150, 90) if not night else (40, 80, 50), True)
        midpoint_circle(surface, self.x + self.w + 40, self.y + self.h // 2, 36, (240, 240, 240), False)
        font = FONTS.get("tiny")
        if font:
            surface.blit(font.render("PLAYGROUND", True, (255, 255, 255)),
                         (self.x + self.w + 8, self.y + self.h // 2 - 6))


class College(Building):
    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        super().draw(surface, pal, night)
        midpoint_ellipse(surface, self.x + self.w // 2, self.y - 22, 70, 16,
                         (70, 150, 90) if not night else (30, 70, 45), True)
        font = FONTS.get("tiny")
        if font:
            surface.blit(font.render("CAMPUS GREEN", True, (255, 255, 255)),
                         (self.x + 70, self.y - 28))


class Mall(Building):
    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        super().draw(surface, pal, night)
        midpoint_ellipse(surface, self.x + self.w // 2, self.y + 50, 40, 22,
                         (140, 200, 220) if not night else (255, 200, 90), True)


class Hospital(Building):
    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        super().draw(surface, pal, night)
        cx, cy = self.x + self.w // 2, self.y + 28
        fill_rect_scan(surface, cx - 20, cy, 40, 14, (200, 40, 40))
        fill_rect_scan(surface, cx - 7, cy - 13, 14, 40, (200, 40, 40))
        fill_rect_scan(surface, 1080, 640, 80, 110, (70, 70, 40))
        for i in range(6):
            dda_line(surface, 1080, 650 + i * 16, 1160, 640 + i * 16, (230, 200, 40), 3)
        font = FONTS.get("tiny")
        if font:
            surface.blit(font.render("EMERGENCY", True, (255, 220, 80)), (1084, 752))


class Restaurant(Building):
    def draw(self, surface: pygame.Surface, pal: dict, night: bool) -> None:
        super().draw(surface, pal, night)
        for i, ox in enumerate((20, 55, 90)):
            tx, ty = self.x + ox, self.y + self.h + 18
            midpoint_ellipse(surface, tx, ty, 11, 6, (150, 100, 60), True)
            fill_rect_scan(surface, tx - 16, ty - 4, 6, 10, (120, 80, 50))
            fill_rect_scan(surface, tx + 10, ty - 4, 6, 10, (120, 80, 50))
            midpoint_circle(surface, tx - 4, ty - 10, 3, (255, 214, 170), True)
            if i % 2 == 0:
                midpoint_circle(surface, tx + 5, ty - 10, 3, (255, 214, 170), True)


class Cafe(Restaurant):
    pass


class SolarPanel:
    def __init__(self, x: int, y: int, cols: int, rows: int):
        self.x, self.y, self.cols, self.rows = x, y, cols, rows

    def draw(self, surface: pygame.Surface, night: bool) -> None:
        draw_solar_array(surface, self.x, self.y, self.cols, self.rows, night)


class Dustbin:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y
        self.fill = 0.4

    def update(self, dt: float, t: float, i: int) -> None:
        self.fill = 0.25 + 0.5 * abs(math.sin(t / 1.8 + i))

    def draw(self, surface: pygame.Surface, night: bool) -> None:
        draw_dustbin(surface, self.x, self.y, self.fill, night)


def draw_crosswalk(surface: pygame.Surface, x: int, y: int, w: int, h: int, vertical: bool, pal: dict) -> None:
    stripe = pal["lane"]
    if vertical:
        i = 0
        while i < w:
            fill_rect_scan(surface, x + i, y, 7, h, stripe)
            i += 12
    else:
        i = 0
        while i < h:
            fill_rect_scan(surface, x, y + i, w, 7, stripe)
            i += 12


def draw_dashed(surface: pygame.Surface, x1: int, y1: int, x2: int, y2: int, color: Color, dash: int = 14, gap: int = 12) -> None:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    d = 0.0
    on = True
    while d < length:
        seg = dash if on else gap
        nxt = min(length, d + seg)
        if on:
            bresenham_line(
                surface,
                int(x1 + ux * d), int(y1 + uy * d),
                int(x1 + ux * nxt), int(y1 + uy * nxt),
                color, 2,
            )
        d = nxt
        on = not on


class Cloud:
    def __init__(self, x: float, y: float, s: float, speed: float):
        self.x, self.y, self.s, self.speed = x, y, s, speed

    def update(self, dt: float) -> None:
        self.x += self.speed * dt
        if self.x > CITY_W + 80:
            self.x = -80

    def draw(self, surface: pygame.Surface, night: bool) -> None:
        col = (236, 242, 250) if not night else (90, 100, 130)
        k = self.s
        midpoint_ellipse(surface, int(self.x), int(self.y), int(28 * k), int(12 * k), col, True)
        midpoint_ellipse(surface, int(self.x + 18 * k), int(self.y + 2), int(22 * k), int(10 * k), col, True)
        midpoint_ellipse(surface, int(self.x - 16 * k), int(self.y + 3), int(18 * k), int(9 * k), col, True)


class Airplane:
    def __init__(self):
        self.x = -60.0
        self.y = 48.0
        self.speed = 70.0
        self.heading = 0.0

    def update(self, dt: float) -> None:
        self.x += self.speed * dt
        if self.x > CITY_W + 80:
            self.x = -80
            self.y = 36 + (int(self.x) % 40)

    def draw(self, surface: pygame.Surface, night: bool) -> None:
        body = [(18, 0), (8, -5), (-16, -4), (-20, 0), (-16, 4), (8, 5)]
        wing = [(-2, 0), (4, -16), (8, -16), (2, 0), (8, 16), (4, 16)]
        tail = [(-16, 0), (-20, -10), (-14, -10), (-12, 0)]
        m = mat_mul(translate(self.x, self.y), rotate(self.heading))
        col = (230, 232, 236) if not night else (180, 184, 196)
        fill_poly(surface, transform_points(m, body), col)
        fill_poly(surface, transform_points(m, wing), (70, 110, 170))
        fill_poly(surface, transform_points(m, tail), (190, 60, 60))
        # Reflection of the plane silhouette on a local x-axis (CG reflection)
        m2 = mat_mul(translate(self.x, self.y + 18), mat_mul(scale(1.0, 0.35), reflect("x")))
        fill_poly(surface, transform_points(m2, body), (180, 200, 220) if not night else (40, 50, 70))
        cx, cy = apply_transform(m, 10, 0)
        midpoint_ellipse(surface, int(cx), int(cy), 4, 3, (80, 160, 210) if not night else (255, 220, 80), True)


class TrafficLight:
    def __init__(self, x: int, y: int, axis: str):
        self.x, self.y = x, y
        self.axis = axis  # "EW" or "NS"
        self.state = LightState.RED
        self.timer = 0.0

    def set_state(self, state: LightState) -> None:
        self.state = state

    def draw(self, surface: pygame.Surface) -> None:
        fill_rect_scan(surface, self.x - 8, self.y - 28, 16, 44, (30, 30, 34))
        colors = {
            LightState.RED: ((220, 40, 40), (60, 60, 20), (20, 60, 20)),
            LightState.YELLOW: ((80, 30, 30), (230, 190, 40), (20, 60, 20)),
            LightState.GREEN: ((80, 30, 30), (60, 60, 20), (40, 210, 70)),
        }
        r, y, g = colors[self.state]
        midpoint_circle(surface, self.x, self.y - 16, 5, r, True)
        midpoint_circle(surface, self.x, self.y - 4, 5, y, True)
        midpoint_circle(surface, self.x, self.y + 8, 5, g, True)


class TrafficController:
    """Two-phase intersection: EW green while NS red, then yellow, swap."""

    def __init__(self):
        self.lights = {
            "EW": TrafficLight(V_ROAD_X - 18, H_ROAD_Y - 8, "EW"),
            "NS": TrafficLight(V_ROAD_X + V_ROAD_W + 18, H_ROAD_Y - 8, "NS"),
            "EW2": TrafficLight(V_ROAD_X + V_ROAD_W + 18, H_ROAD_Y + H_ROAD_H + 8, "EW"),
            "NS2": TrafficLight(V_ROAD_X - 18, H_ROAD_Y + H_ROAD_H + 8, "NS"),
        }
        self.phase = 0  # 0 EW green, 1 EW yellow, 2 NS green, 3 NS yellow
        self.t = 0.0
        self.durations = [6.5, 1.8, 6.5, 1.8]
        self.emergency = False
        self._apply()

    def _apply(self) -> None:
        if self.emergency:
            for k, L in self.lights.items():
                L.set_state(LightState.RED)
            return
        ew = LightState.GREEN if self.phase == 0 else LightState.YELLOW if self.phase == 1 else LightState.RED
        ns = LightState.GREEN if self.phase == 2 else LightState.YELLOW if self.phase == 3 else LightState.RED
        self.lights["EW"].set_state(ew)
        self.lights["EW2"].set_state(ew)
        self.lights["NS"].set_state(ns)
        self.lights["NS2"].set_state(ns)

    def update(self, dt: float, emergency: bool) -> None:
        self.emergency = emergency
        if emergency:
            self._apply()
            return
        self.t += dt
        if self.t >= self.durations[self.phase]:
            self.t = 0.0
            self.phase = (self.phase + 1) % 4
        self._apply()

    def state_for_lane(self, lane: str) -> LightState:
        if lane in ("EB", "WB", "EB_NB", "NB_WB", "AMB", "AMB_RET", "PARK_IN"):
            if lane in ("EB_NB",):
                return self.lights["EW"].state
            if lane in ("NB_WB",):
                return self.lights["NS"].state
            if lane.startswith("AMB"):
                return LightState.GREEN if self.emergency else self.lights["EW"].state
            return self.lights["EW"].state
        return self.lights["NS"].state

    def draw(self, surface: pygame.Surface) -> None:
        for L in self.lights.values():
            L.draw(surface)


class ParkingSlot:
    def __init__(self, x: int, y: int, index: int, angle: float = -math.pi / 2):
        self.x, self.y, self.index = x, y, index
        self.occupied = False
        self.vehicle_color: Color = (200, 80, 80)
        self.hold = 0.0
        self.angle = angle
        self.phase = "empty"  # empty | entering | occupied | leaving
        self.anim = 0.0
        self.car_x = float(x)
        self.car_y = float(y)
        self.entry = (160.0, LANE_EB_Y)

    def begin_enter(self, color: Color) -> None:
        self.vehicle_color = color
        self.phase = "entering"
        self.anim = 0.0
        self.occupied = True
        self.car_x, self.car_y = self.entry
        self.hold = random.uniform(8.0, 18.0)

    def begin_leave(self) -> None:
        self.phase = "leaving"
        self.anim = 0.0

    def update(self, dt: float) -> None:
        if self.phase == "entering":
            self.anim = min(1.0, self.anim + dt * 0.55)
            ex, ey = self.entry
            # Drive in via aisle (translation along a two-segment path)
            if self.anim < 0.45:
                t = self.anim / 0.45
                self.car_x = ex
                self.car_y = ey + (self.y - ey) * t
                self.angle = math.pi / 2
            else:
                t = (self.anim - 0.45) / 0.55
                self.car_x = ex + (self.x - ex) * t
                self.car_y = self.y
                self.angle = 0.0 if self.x > ex else math.pi
            if self.anim >= 1.0:
                self.phase = "occupied"
                self.car_x, self.car_y = float(self.x), float(self.y)
                self.angle = -math.pi / 2
        elif self.phase == "occupied":
            self.hold -= dt
            if self.hold <= 0:
                self.begin_leave()
        elif self.phase == "leaving":
            self.anim = min(1.0, self.anim + dt * 0.6)
            ex, ey = self.entry
            if self.anim < 0.5:
                t = self.anim / 0.5
                self.car_x = self.x + (ex - self.x) * t
                self.car_y = self.y
            else:
                t = (self.anim - 0.5) / 0.5
                self.car_x = ex
                self.car_y = self.y + (ey - self.y) * t
                self.angle = -math.pi / 2
            if self.anim >= 1.0:
                self.phase = "empty"
                self.occupied = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        col = (56, 160, 90) if not self.occupied else (180, 64, 64)
        bresenham_line(surface, self.x - 16, self.y - 28, self.x + 16, self.y - 28, col, 2)
        bresenham_line(surface, self.x - 16, self.y + 28, self.x + 16, self.y + 28, col, 2)
        bresenham_line(surface, self.x - 16, self.y - 28, self.x - 16, self.y + 28, col, 2)
        bresenham_line(surface, self.x + 16, self.y - 28, self.x + 16, self.y + 28, col, 2)
        surface.blit(font.render(str(self.index + 1), True, (240, 240, 240)), (self.x - 4, self.y - 40))
        if self.phase != "empty":
            m = compose(translate(self.car_x, self.car_y), rotate(self.angle), scale(1.0, 1.0))
            fill_poly(surface, transform_points(m, PARK_BODY), self.vehicle_color)
            outline_poly(surface, transform_points(m, PARK_BODY), (20, 20, 20))


class ParkingLot:
    def __init__(self):
        self.slots = [
            ParkingSlot(70 + i * 42, 700, i) for i in range(5)
        ] + [
            ParkingSlot(70 + i * 42, 780, 5 + i) for i in range(5)
        ]
        self.assign_timer = 3.0
        self.release_timer = 5.0
        self.message = "Parking ready"

    @property
    def available(self) -> int:
        return sum(1 for s in self.slots if not s.occupied)

    def assign_free(self) -> Optional[ParkingSlot]:
        free = [s for s in self.slots if s.phase == "empty"]
        if not free:
            return None
        slot = free[0]
        slot.begin_enter(random.choice([
            (210, 70, 70), (70, 120, 210), (230, 180, 50), (50, 160, 120), (160, 80, 180),
        ]))
        self.message = f"Assigned slot {slot.index + 1}"
        return slot

    def update(self, dt: float) -> None:
        self.assign_timer -= dt
        if self.assign_timer <= 0:
            self.assign_timer = random.uniform(4.0, 9.0)
            if self.available:
                self.assign_free()
            else:
                self.message = "Parking FULL"
        for s in self.slots:
            prev = s.phase
            s.update(dt)
            if prev != "empty" and s.phase == "empty":
                self.message = f"Slot {s.index + 1} vacated"

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        fill_rect_scan(surface, 28, 640, 250, 180, (48, 52, 58))
        for s in self.slots:
            s.draw(surface, font)
        title = font.render(f"SMART PARKING  {self.available}/10 FREE", True, (240, 240, 240))
        surface.blit(title, (40, 648))


class Pedestrian:
    def __init__(self, path: Path, speed: float, color: Color, s: float = 0.0):
        self.path = path
        self.speed = speed
        self.color = color
        self.s = s
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.state = "walk"

    def update(self, dt: float) -> None:
        self.s += self.speed * dt
        if self.s > self.path.length:
            self.s = 0.0
        self.x, self.y, self.heading = self.path.sample(self.s)

    def draw(self, surface: pygame.Surface) -> None:
        m = mat_mul(translate(self.x, self.y), rotate(self.heading))
        body = [(0, 0), (6, 0)]
        p0, p1 = transform_points(m, body)
        bresenham_line(surface, int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), self.color, 2)
        midpoint_circle(surface, int(self.x), int(self.y) - 5, 3, (255, 214, 170), True)


# =============================================================================
# 6. VEHICLES + TRANSFORMATIONS + TRAFFIC
# =============================================================================


class Vehicle:
    def __init__(self, path: Path, speed: float, color: Color, kind: str = "car", s: float = 0.0):
        self.path = path
        self.base_speed = speed
        self.speed = speed
        self.color = color
        self.kind = kind
        self.s = s
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.state = "CRUISE"
        self.length = 40 if kind in ("bus", "college") else 36 if kind == "ambulance" else 28
        self.yield_timer = 0.0
        self.lights_on = False
        self.scale = 1.18 if kind in ("bus", "college") else 1.12 if kind == "ambulance" else 1.0

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x - self.length * 0.45, self.y - 10, self.length * 0.9, 20)

    def approaching_intersection(self) -> float:
        return self.path.dist_to_intersection(self.s)

    def in_intersection(self) -> bool:
        return aabb_overlap(self.bbox(), (INTER_X, INTER_Y, INTER_W, INTER_H))

    def stop_line_distance(self) -> float:
        d = self.approaching_intersection()
        return max(0.0, d - STOP_MARGIN)

    def update(self, dt: float, sim: "City") -> None:
        desired = self.base_speed * sim.time_scale
        light = sim.traffic.state_for_lane(self.path.name.split("_")[0] if self.path.name[:2] in ("EB", "WB", "SB", "NB") else self.path.name)
        # Map turning paths onto axis lights
        if self.path.name in ("EB", "WB", "EB_NB", "PARK_IN"):
            light = sim.traffic.lights["EW"].state
        elif self.path.name in ("SB", "NB", "SB_EB", "WB_SB", "NB_WB"):
            light = sim.traffic.lights["NS"].state

        dist_inter = self.approaching_intersection()
        stop_d = dist_inter - STOP_MARGIN

        emergency = sim.emergency_active and self.kind != "ambulance"
        if emergency:
            ax, ay = sim.ambulance.x, sim.ambulance.y
            dist_a = math.hypot(self.x - ax, self.y - ay)
            amb_in = sim.ambulance.in_intersection()
            if dist_a < 100 and not self.in_intersection():
                desired = 0.0
                self.state = "YIELD"
                self.yield_timer = 0.4
            elif (amb_in or sim.ambulance.approaching_intersection() < 90) and not self.in_intersection() and stop_d < 70:
                desired = 0.0
                self.state = "YIELD"
                self.yield_timer = 0.35

        if self.yield_timer > 0:
            self.yield_timer -= dt
        yielding = self.state == "YIELD" or self.yield_timer > 0

        if self.kind != "ambulance" and not self.in_intersection() and 0 < stop_d < 160:
            if light == LightState.RED:
                desired = 0.0 if stop_d < 22 else min(desired, stop_d * 1.2)
                self.state = "STOP"
            elif light == LightState.YELLOW:
                # Dilemma zone: stop if we can, else clear
                if stop_d / max(self.speed, 1) < 1.4 and stop_d > 18:
                    desired = max(40.0, desired)
                    self.state = "CLEAR"
                else:
                    desired = 0.0 if stop_d < 22 else min(desired, stop_d)
                    self.state = "SLOW"
            else:
                self.state = "CRUISE"

        # Following distance on same path
        leader = sim.leader_ahead(self)
        if leader is not None:
            gap = leader.s - self.s
            if gap < 0 and self.path.loop:
                gap += self.path.length
            safe = 38 + 0.35 * self.speed
            if 0 < gap < safe:
                desired = min(desired, leader.speed * 0.85)
                self.state = "FOLLOW"
            if 0 < gap < 26:
                desired = 0.0
                self.state = "HOLD"

        # Intersection mutex for conflicting directions
        if self.kind != "ambulance" and stop_d < 20 and not self.in_intersection():
            if light != LightState.GREEN and light != LightState.YELLOW:
                desired = 0.0
            elif sim.intersection_blocked_for(self):
                desired = 0.0
                self.state = "WAIT_INT"

        if yielding and self.kind != "ambulance":
            desired = 0.0
            self.state = "YIELD"

        if self.kind == "ambulance" and sim.emergency_active:
            desired = max(desired, 140)
            self.state = "EMERGENCY"
            self.lights_on = True
        elif self.kind == "ambulance":
            self.lights_on = False
            if self.path.name == "AMB":
                desired = 0.0
                self.speed = 0.0
                self.state = "STANDBY"
                self.s = 0.0
                self.x, self.y, self.heading = self.path.sample(self.s)
                return
            desired = min(desired, 70)

        accel = 90.0 if desired > self.speed else 140.0
        if desired > self.speed:
            self.speed = min(desired, self.speed + accel * dt)
        else:
            self.speed = max(desired, self.speed - accel * dt)

        self.s += self.speed * dt
        if self.path.loop:
            if self.s >= self.path.length:
                self.s -= self.path.length
        else:
            if self.s >= self.path.length:
                if self.kind == "ambulance":
                    if self.path.name == "AMB":
                        self.path = sim.paths["AMB_RET"]
                        self.s = 0.0
                        sim.finish_emergency_if_ready()
                    else:
                        self.path = sim.paths["AMB"]
                        self.s = 0.0
                        sim.emergency_active = False
                        self.lights_on = False
                        self.speed = 0.0
                        self.state = "STANDBY"
                else:
                    # Resume a looping arterial after a turning route
                    nxt = {"EB_NB": "NB", "SB_EB": "EB", "WB_SB": "SB", "NB_WB": "WB"}.get(self.path.name)
                    if nxt:
                        self.path = sim.paths[nxt]
                    self.s = 0.0

        self.x, self.y, self.heading = self.path.sample(self.s)

    def local_poly(self) -> List[Vec2]:
        return list(BUS_BODY) if self.kind in ("bus", "college", "ambulance") else list(CAR_BODY)

    def draw(self, surface: pygame.Surface, night: bool) -> None:
        # COMPOSITE TRANSFORM: M = T(x,y) · R(heading) · S(scale)
        m = compose(translate(self.x, self.y), rotate(self.heading), scale(self.scale, self.scale))
        pts = transform_points(m, self.local_poly())
        fill_poly(surface, pts, self.color)
        outline_poly(surface, pts, (20, 20, 24), 1)
        wheels = BUS_WHEELS if self.kind in ("bus", "college", "ambulance") else CAR_WHEELS
        for lx, ly in wheels:
            wx, wy = apply_transform(m, lx, ly)
            midpoint_circle(surface, int(wx), int(wy), 3, (20, 20, 20), True)
        if night:
            hx, hy = apply_transform(m, 18, 0)
            cone = transform_points(m, HEADLIGHT)
            fill_poly(surface, cone, (255, 240, 160))
            midpoint_circle(surface, int(hx), int(hy), 2, (255, 255, 200), True)
        if self.kind == "ambulance" and self.lights_on:
            flash = pygame.time.get_ticks() // 120 % 2 == 0
            c1 = (255, 40, 40) if flash else (40, 80, 255)
            c2 = (40, 80, 255) if flash else (255, 40, 40)
            a, b = apply_transform(m, -8, 0), apply_transform(m, 8, 0)
            midpoint_circle(surface, int(a[0]), int(a[1] - 8), 4, c1, True)
            midpoint_circle(surface, int(b[0]), int(b[1] - 8), 4, c2, True)
        label = {"bus": "CITY BUS", "college": "COLLEGE", "ambulance": "AMB", "car": ""}.get(self.kind, "")
        font = FONTS.get("tiny")
        if label and font:
            surface.blit(font.render(label, True, (255, 255, 255)), (int(self.x) - 18, int(self.y) - 20))


class Car(Vehicle):
    def __init__(self, path: Path, speed: float, color: Color, s: float = 0.0):
        super().__init__(path, speed, color, "car", s)


class Bus(Vehicle):
    def __init__(self, path: Path, speed: float, s: float = 0.0):
        super().__init__(path, speed, (40, 90, 170), "bus", s)


class CollegeBus(Vehicle):
    def __init__(self, path: Path, speed: float, s: float = 0.0):
        super().__init__(path, speed, (40, 130, 90), "college", s)


class Ambulance(Vehicle):
    def __init__(self, path: Path, speed: float, s: float = 0.0):
        super().__init__(path, speed, (245, 245, 245), "ambulance", s)
        self.color = (250, 250, 252)


# =============================================================================
# 7. SCENE BAKE  (city / roads / buildings)
# =============================================================================

def paint_roads(surface: pygame.Surface, pal: dict) -> None:
    # Sidewalks
    fill_rect_scan(surface, 0, H_ROAD_Y - 18, CITY_W, H_ROAD_H + 36, pal["sidewalk"])
    fill_rect_scan(surface, V_ROAD_X - 18, 0, V_ROAD_W + 36, CITY_H, pal["sidewalk"])
    # Asphalt
    fill_rect_scan(surface, 0, H_ROAD_Y, CITY_W, H_ROAD_H, pal["asphalt"])
    fill_rect_scan(surface, V_ROAD_X, 0, V_ROAD_W, CITY_H, pal["asphalt"])
    # Road edges via DDA
    dda_line(surface, 0, H_ROAD_Y, CITY_W, H_ROAD_Y, pal["asphalt_edge"], 2)
    dda_line(surface, 0, H_ROAD_Y + H_ROAD_H, CITY_W, H_ROAD_Y + H_ROAD_H, pal["asphalt_edge"], 2)
    dda_line(surface, V_ROAD_X, 0, V_ROAD_X, CITY_H, pal["asphalt_edge"], 2)
    dda_line(surface, V_ROAD_X + V_ROAD_W, 0, V_ROAD_X + V_ROAD_W, CITY_H, pal["asphalt_edge"], 2)
    # Median (double yellow) — Bresenham
    bresenham_line(surface, 0, H_ROAD_Y + H_ROAD_H // 2, V_ROAD_X, H_ROAD_Y + H_ROAD_H // 2, pal["median"], 2)
    bresenham_line(surface, V_ROAD_X + V_ROAD_W, H_ROAD_Y + H_ROAD_H // 2, CITY_W, H_ROAD_Y + H_ROAD_H // 2, pal["median"], 2)
    bresenham_line(surface, V_ROAD_X + V_ROAD_W // 2, 0, V_ROAD_X + V_ROAD_W // 2, H_ROAD_Y, pal["median"], 2)
    bresenham_line(surface, V_ROAD_X + V_ROAD_W // 2, H_ROAD_Y + H_ROAD_H, V_ROAD_X + V_ROAD_W // 2, CITY_H, pal["median"], 2)
    # Dashed lane marks (skip intersection)
    draw_dashed(surface, 0, LANE_EB_Y - 18, V_ROAD_X - 4, LANE_EB_Y - 18, pal["lane"])
    draw_dashed(surface, V_ROAD_X + V_ROAD_W + 4, LANE_EB_Y - 18, CITY_W, LANE_EB_Y - 18, pal["lane"])
    draw_dashed(surface, 0, LANE_WB_Y + 18, V_ROAD_X - 4, LANE_WB_Y + 18, pal["lane"])
    draw_dashed(surface, V_ROAD_X + V_ROAD_W + 4, LANE_WB_Y + 18, CITY_W, LANE_WB_Y + 18, pal["lane"])
    # Stop lines
    bresenham_line(surface, V_ROAD_X - 6, H_ROAD_Y + 8, V_ROAD_X - 6, H_ROAD_Y + H_ROAD_H - 8, pal["lane"], 3)
    bresenham_line(surface, V_ROAD_X + V_ROAD_W + 6, H_ROAD_Y + 8, V_ROAD_X + V_ROAD_W + 6, H_ROAD_Y + H_ROAD_H - 8, pal["lane"], 3)
    bresenham_line(surface, V_ROAD_X + 8, H_ROAD_Y - 6, V_ROAD_X + V_ROAD_W - 8, H_ROAD_Y - 6, pal["lane"], 3)
    bresenham_line(surface, V_ROAD_X + 8, H_ROAD_Y + H_ROAD_H + 6, V_ROAD_X + V_ROAD_W - 8, H_ROAD_Y + H_ROAD_H + 6, pal["lane"], 3)
    # Crossings
    draw_crosswalk(surface, V_ROAD_X - 22, H_ROAD_Y + 10, 16, H_ROAD_H - 20, True, pal)
    draw_crosswalk(surface, V_ROAD_X + V_ROAD_W + 6, H_ROAD_Y + 10, 16, H_ROAD_H - 20, True, pal)
    draw_crosswalk(surface, V_ROAD_X + 10, H_ROAD_Y - 22, V_ROAD_W - 20, 16, False, pal)
    draw_crosswalk(surface, V_ROAD_X + 10, H_ROAD_Y + H_ROAD_H + 6, V_ROAD_W - 20, 16, False, pal)


def bake_city(night: bool) -> pygame.Surface:
    pal = NIGHT if night else DAY
    surf = pygame.Surface((CITY_W, CITY_H))
    fill_rect_scan(surf, 0, 0, CITY_W, CITY_H, pal["grass"])
    sky_gradient(surf, pal, 118)
    Road().paint(surf, pal)

    # Distant skyline (depth layer behind the arterial city)
    sil = (70, 96, 88) if not night else (18, 28, 36)
    x = 8
    rng = random.Random(7)
    while x < CITY_W - 8:
        bw, bh = rng.randint(18, 42), rng.randint(16, 52)
        fill_rect_scan(surf, x, 118 - bh, bw, bh, sil)
        if night:
            for i in range(rng.randint(0, 3)):
                fill_rect_scan(surf, x + 4 + (i % 3) * 8, 118 - bh + 6 + (i // 3) * 8, 3, 4, (255, 210, 120))
        x += bw + rng.randint(4, 10)

    # --- landmarks (named CG objects) ---
    Building(36, 128, 142, 108, (196, 122, 96), (150, 78, 64), "RESIDENCE A", 4, 3).draw(surf, pal, night)
    Building(196, 118, 128, 118, (214, 178, 126), (168, 120, 72), "APARTMENTS", 4, 4).draw(surf, pal, night)
    Building(344, 136, 150, 102, (186, 92, 92), (140, 60, 60), "RESIDENCE B", 5, 3).draw(surf, pal, night)
    Building(500, 168, 70, 88, (170, 150, 132), (120, 100, 88), "TOWER", 2, 4, 10).draw(surf, pal, night)
    School(36, 268, 210, 118, (92, 140, 186), (60, 90, 140), "GREENFIELD SCHOOL", 6, 3, 14).draw(surf, pal, night)
    SolarPanel(268, 258, 8, 3).draw(surf, night)

    College(730, 108, 236, 138, (86, 110, 168), (50, 70, 120), "CITY UNIVERSITY", 7, 4, 16).draw(surf, pal, night)
    SolarPanel(748, 96, 10, 2).draw(surf, night)
    Mall(990, 140, 236, 148, (168, 92, 140), (120, 50, 100), "METRO MALL", 8, 4, 16).draw(surf, pal, night)

    Cafe(300, 548, 150, 86, (196, 140, 88), (150, 96, 50), "CAFE AURUM", 4, 2).draw(surf, pal, night)
    Restaurant(468, 548, 110, 86, (88, 150, 130), (50, 110, 90), "BISTRO 42", 3, 2, 10).draw(surf, pal, night)
    Hospital(730, 548, 200, 126, (236, 236, 240), (180, 70, 70), "CITY HOSPITAL", 6, 3, 14).draw(surf, pal, night)
    Restaurant(968, 548, 140, 68, (210, 120, 80), (160, 80, 50), "FOOD COURT", 4, 2, 10).draw(surf, pal, night)

    # Trees
    for t in (
        (24, 250), (190, 260), (350, 270), (540, 250), (540, 380),
        (720, 280), (980, 320), (1240, 260), (1240, 380),
        (24, 540), (260, 620), (560, 800), (720, 800), (1240, 540), (1240, 820),
        (400, 820), (20, 820),
    ):
        draw_tree(surf, t[0], t[1], night)

    # Street lamps
    for lx, ly in (
        (80, H_ROAD_Y - 4), (260, H_ROAD_Y - 4), (460, H_ROAD_Y - 4),
        (760, H_ROAD_Y - 4), (960, H_ROAD_Y - 4), (1180, H_ROAD_Y - 4),
        (80, H_ROAD_Y + H_ROAD_H + 30), (460, H_ROAD_Y + H_ROAD_H + 30),
        (960, H_ROAD_Y + H_ROAD_H + 30),
        (V_ROAD_X - 8, 160), (V_ROAD_X - 8, 320), (V_ROAD_X + V_ROAD_W + 20, 160),
        (V_ROAD_X + V_ROAD_W + 20, 740), (V_ROAD_X - 8, 740),
    ):
        draw_streetlamp(surf, lx, ly, night)

    # Sun / moon
    if night:
        midpoint_circle(surf, 1200, 48, 22, (230, 230, 210), True)
        midpoint_circle(surf, 1192, 44, 8, pal["sky_top"], True)
    else:
        midpoint_circle(surf, 80, 48, 24, (255, 210, 70), True)
        midpoint_circle(surf, 80, 48, 18, (255, 230, 120), True)

    return surf


# =============================================================================
# 8. CITY SIMULATION ENGINE
# =============================================================================

class City:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("segoe ui", 16)
        self.font_b = pygame.font.SysFont("segoe ui", 18, bold=True)
        self.font_s = pygame.font.SysFont("segoe ui", 13)
        self.font_title = pygame.font.SysFont("segoe ui", 20, bold=True)
        FONTS["label14"] = pygame.font.SysFont("segoe ui", 14, bold=True)
        FONTS["tiny"] = pygame.font.SysFont("segoe ui", 11, bold=True)
        SPRITES["bin_label"] = FONTS["tiny"].render("IOT BIN", True, (210, 230, 210))
        self.mode = TimeMode.DAY
        self.paused = False
        self.time_scale = 1.0
        self.show_parking_info = False
        self.show_traffic_info = False
        self.emergency_active = False
        self.paths = make_city_paths()
        self.traffic = TrafficController()
        self.parking = ParkingLot()
        self.clouds = [
            Cloud(120, 40, 1.1, 12), Cloud(420, 28, 0.9, 8),
            Cloud(760, 50, 1.3, 10), Cloud(1040, 34, 0.8, 7),
        ]
        self.plane = Airplane()
        self.day_cache = bake_city(False)
        self.night_cache = bake_city(True)
        self.bins = [Dustbin(x, y) for x, y in BIN_SITES]
        self.solar = 82.0
        self._spawn_actors()

    def _spawn_actors(self) -> None:
        p = self.paths
        colors = [(210, 70, 70), (70, 120, 210), (230, 180, 50), (50, 160, 120),
                  (160, 80, 180), (230, 120, 50), (90, 90, 90), (40, 160, 180)]
        self.vehicles: List[Vehicle] = [
            Car(p["EB"], 78, colors[0], 40),
            Car(p["EB"], 70, colors[1], 220),
            Car(p["EB"], 74, colors[2], 520),
            Car(p["WB"], 76, colors[3], 80),
            Car(p["WB"], 68, colors[4], 360),
            Car(p["WB"], 72, colors[5], 700),
            Car(p["SB"], 70, colors[6], 100),
            Car(p["SB"], 66, colors[7], 380),
            Car(p["NB"], 72, colors[0], 150),
            Car(p["NB"], 68, colors[2], 430),
            Car(p["EB_NB"], 64, colors[3], 20),
            Car(p["SB_EB"], 64, colors[5], 40),
            Bus(p["EB"], 60, 800),
            Bus(p["NB"], 58, 260),
            CollegeBus(p["WB"], 58, 500),
            CollegeBus(p["SB"], 56, 200),
            Car(p["WB_SB"], 62, colors[1], 30),
            Car(p["NB_WB"], 62, colors[6], 50),
        ]
        self.ambulance = Ambulance(p["AMB"], 90, 0)
        self.ambulance.speed = 0
        self.ambulance.state = "STANDBY"
        self.vehicles.append(self.ambulance)
        for v in self.vehicles:
            v.x, v.y, v.heading = v.path.sample(v.s)

        walk_n = Path([(40, H_ROAD_Y - 12), (CITY_W - 40, H_ROAD_Y - 12)], True, "WALK_N")
        walk_s = Path([(CITY_W - 40, H_ROAD_Y + H_ROAD_H + 12), (40, H_ROAD_Y + H_ROAD_H + 12)], True, "WALK_S")
        cafe = Path([(320, 680), (560, 680), (560, 650), (320, 650)], True, "CAFE")
        self.pedestrians = [
            Pedestrian(walk_n, 28, (40, 40, 50), 10),
            Pedestrian(walk_n, 24, (80, 50, 40), 300),
            Pedestrian(walk_n, 26, (30, 60, 80), 700),
            Pedestrian(walk_s, 25, (50, 40, 60), 80),
            Pedestrian(walk_s, 22, (90, 40, 40), 500),
            Pedestrian(cafe, 16, (40, 40, 40), 0),
            Pedestrian(cafe, 14, (70, 40, 50), 80),
        ]

    def reset(self) -> None:
        self.paused = False
        self.time_scale = 1.0
        self.emergency_active = False
        self.traffic = TrafficController()
        self.parking = ParkingLot()
        self._spawn_actors()

    def leader_ahead(self, v: Vehicle) -> Optional[Vehicle]:
        best = None
        best_gap = 1e9
        for o in self.vehicles:
            if o is v or o.path.name != v.path.name:
                continue
            gap = o.s - v.s
            if gap <= 1:
                if v.path.loop:
                    gap += v.path.length
                else:
                    continue
            if gap < best_gap:
                best_gap = gap
                best = o
        return best

    def intersection_blocked_for(self, v: Vehicle) -> bool:
        v_ns = v.path.name.startswith("N") or v.path.name.startswith("S") or v.path.name in ("SB_EB", "NB_WB", "WB_SB")
        for o in self.vehicles:
            if o is v or not o.in_intersection():
                continue
            o_ns = o.path.name.startswith("N") or o.path.name.startswith("S") or o.path.name in ("SB_EB", "NB_WB", "WB_SB")
            if v_ns != o_ns:
                return True
        return False

    def finish_emergency_if_ready(self) -> None:
        # Called when ambulance reaches college; stay in emergency until return starts
        pass

    def activate_emergency(self) -> None:
        self.emergency_active = True
        self.ambulance.lights_on = True
        self.ambulance.path = self.paths["AMB"]
        if self.ambulance.state == "STANDBY" or self.ambulance.s < 8:
            self.ambulance.s = 0.0
        self.ambulance.speed = max(self.ambulance.speed, 55.0)
        self.ambulance.state = "EMERGENCY"

    def update(self, dt: float) -> None:
        if self.paused:
            return
        dt = dt * self.time_scale
        self.traffic.update(dt, self.emergency_active)
        self.parking.update(dt)
        for v in self.vehicles:
            v.update(dt, self)
        for p in self.pedestrians:
            p.update(dt)
        for c in self.clouds:
            c.update(dt)
        self.plane.update(dt)
        t = pygame.time.get_ticks() / 1000.0
        for i, b in enumerate(self.bins):
            b.update(dt, t, i)
        if self.mode == TimeMode.DAY:
            self.solar = min(98.0, 70.0 + 20.0 * abs(math.sin(t / 4.0)))
        else:
            self.solar = max(12.0, self.solar - 8.0 * dt)

    def traffic_status(self) -> str:
        if self.emergency_active:
            return "EMERGENCY"
        stopped = sum(1 for v in self.vehicles if v.speed < 8 and v.kind != "ambulance")
        if stopped >= 6:
            return "CONGESTED"
        return "NORMAL"

    def draw_night_glows(self, surface: pygame.Surface) -> None:
        glow = pygame.Surface((CITY_W, CITY_H), pygame.SRCALPHA)
        for lx, ly in LAMP_GLOW:
            pygame.draw.circle(glow, (255, 210, 80, 38), (lx, ly), 34)
        surface.blit(glow, (0, 0))

    def draw(self, screen: pygame.Surface) -> None:
        night = self.mode == TimeMode.NIGHT
        screen.blit(self.night_cache if night else self.day_cache, (0, 0))
        if night:
            self.draw_night_glows(screen)
        for c in self.clouds:
            c.draw(screen, night)
        self.plane.draw(screen, night)
        self.parking.draw(screen, self.font_s)
        for b in self.bins:
            b.draw(screen, night)
        self.traffic.draw(screen)
        for p in self.pedestrians:
            p.draw(screen)
        # Draw vehicles sorted by y for simple depth
        for v in sorted(self.vehicles, key=lambda z: z.y):
            v.draw(screen, night)
        self.draw_hud(screen)
        if self.show_parking_info:
            self.draw_overlay(screen, "PARKING", [
                f"Available: {self.parking.available} / {len(self.parking.slots)}",
                f"Status: {self.parking.message}",
                "Automatic assignment: nearest free slot",
                "Vehicles enter and vacate on timers",
            ])
        if self.show_traffic_info:
            ew = self.traffic.lights["EW"].state.value
            ns = self.traffic.lights["NS"].state.value
            self.draw_overlay(screen, "TRAFFIC", [
                f"EW lights: {ew}",
                f"NS lights: {ns}",
                f"Network: {self.traffic_status()}",
                f"Vehicles: {len(self.vehicles)}",
                "Ambulance has intersection priority",
            ])
        if self.paused:
            t = self.font_title.render("PAUSED", True, (255, 220, 80))
            screen.blit(t, (CITY_W // 2 - 40, 16))

    def draw_overlay(self, screen: pygame.Surface, title: str, lines: List[str]) -> None:
        box = pygame.Surface((360, 160), pygame.SRCALPHA)
        box.fill((10, 16, 28, 210))
        screen.blit(box, (40, 40))
        screen.blit(self.font_b.render(title, True, (255, 220, 80)), (56, 50))
        for i, line in enumerate(lines):
            screen.blit(self.font_s.render(line, True, (230, 230, 230)), (56, 80 + i * 20))

    def draw_hud(self, screen: pygame.Surface) -> None:
        hud = pygame.Rect(HUD_X, 0, WIDTH - HUD_X, HEIGHT)
        pygame.draw.rect(screen, (18, 24, 34), hud)
        pygame.draw.line(screen, (80, 160, 220), (HUD_X, 0), (HUD_X, HEIGHT), 2)
        y = 24
        screen.blit(self.font_title.render("SMART CITY", True, (80, 200, 255)), (HUD_X + 24, y))
        screen.blit(self.font_b.render("CONTROL CENTER", True, (230, 230, 230)), (HUD_X + 24, y + 28))
        y = 90
        rows = [
            ("MODE", "NIGHT" if self.mode == TimeMode.NIGHT else "DAY"),
            ("Traffic", self.traffic_status()),
            ("Vehicles", str(len(self.vehicles))),
            ("Parking", f"{self.parking.available} AVAILABLE"),
            ("Emergency", "ACTIVE" if self.emergency_active else "CLEAR"),
            ("Solar Energy", f"{self.solar:.0f}%"),
            ("Speed", f"{self.time_scale:.2f}x"),
            ("EW Signal", self.traffic.lights["EW"].state.value),
            ("NS Signal", self.traffic.lights["NS"].state.value),
        ]
        for label, val in rows:
            screen.blit(self.font_s.render(label.upper(), True, (120, 140, 160)), (HUD_X + 24, y))
            col = (255, 90, 90) if val in ("ACTIVE", "EMERGENCY", "CONGESTED", "RED") else (80, 220, 120) if val in ("CLEAR", "NORMAL", "GREEN", "DAY") else (230, 230, 230)
            screen.blit(self.font_b.render(str(val), True, col), (HUD_X + 24, y + 16))
            y += 52
        y += 8
        screen.blit(self.font_s.render("CONTROLS", True, (80, 200, 255)), (HUD_X + 24, y))
        y += 24
        for line in (
            "SPACE  Pause",
            "N      Day / Night",
            "A      Ambulance",
            "P      Parking info",
            "T      Traffic info",
            "R      Reset",
            "+ / -  Speed",
            "ESC    Exit",
        ):
            screen.blit(self.font_s.render(line, True, (180, 190, 200)), (HUD_X + 24, y))
            y += 20
        screen.blit(self.font_s.render("CG: DDA · Bresenham · Midpoint · T·R·S", True, (120, 180, 200)),
                    (HUD_X + 16, HEIGHT - 28))


def handle_key(city: City, key: int) -> bool:
    if key in (pygame.K_ESCAPE,):
        return False
    if key == pygame.K_SPACE:
        city.paused = not city.paused
    elif key == pygame.K_n:
        city.mode = TimeMode.NIGHT if city.mode == TimeMode.DAY else TimeMode.DAY
    elif key == pygame.K_a:
        city.activate_emergency()
    elif key == pygame.K_p:
        city.show_parking_info = not city.show_parking_info
        city.show_traffic_info = False
    elif key == pygame.K_t:
        city.show_traffic_info = not city.show_traffic_info
        city.show_parking_info = False
    elif key == pygame.K_r:
        city.reset()
    elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        city.time_scale = min(3.0, city.time_scale + 0.25)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        city.time_scale = max(0.25, city.time_scale - 0.25)
    return True


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Smart City — Computer Graphics Simulation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    city = City()
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                running = handle_key(city, event.key)
        city.update(dt)
        city.draw(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
