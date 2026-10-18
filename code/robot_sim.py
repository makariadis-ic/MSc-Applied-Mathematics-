"""robot_sim.py — differential-drive navigation benchmark for the hybrid FIS thesis.

Physics upgrades over a kinematic toy: first-order actuator lag, wheel slip
under high commanded acceleration, and velocity damping (friction). Sensors
are ray-cast rangefinders with Gaussian noise. Every run records a JSON-able
trace for the 3D/WebXR replay viewer.

Arenas: cluttered, corridor, u_trap, moving. All geometry = segments (walls)
+ circles (obstacles, possibly moving).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ------------------------------------------------------------------ geometry

def ray_circle(px, py, dx, dy, cx, cy, r):
    """Distance along ray (px,py)+t(dx,dy) to circle (cx,cy,r); inf if miss."""
    ox, oy = px - cx, py - cy
    b = ox * dx + oy * dy
    c = ox * ox + oy * oy - r * r
    disc = b * b - c
    if disc < 0:
        return np.inf
    t = -b - np.sqrt(disc)
    return t if t > 1e-9 else np.inf


def ray_segment(px, py, dx, dy, x1, y1, x2, y2):
    """Distance along ray to segment; inf if miss."""
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return np.inf
    t = ((x1 - px) * ey - (y1 - py) * ex) / den
    s = ((x1 - px) * dy - (y1 - py) * dx) / den
    return t if (t > 1e-9 and 0.0 <= s <= 1.0) else np.inf

# ------------------------------------------------------------------ arena

@dataclass
class Arena:
    name: str
    segments: list                 # [(x1,y1,x2,y2), ...]
    circles: list                  # [{"c":[x,y], "r":r, "v":[vx,vy] or None}, ...]
    start: tuple                   # (x, y, theta)
    goal: tuple                    # (x, y)
    size: tuple = (10.0, 10.0)

    def obstacles_at(self, t):
        """Circle positions at time t (moving circles bounce inside the box)."""
        out = []
        W, H = self.size
        for ob in self.circles:
            if ob.get("v"):
                # bounce: reflect position with period 2*span
                x0, y0 = ob["c"]; vx, vy = ob["v"]; r = ob["r"]
                x = _bounce(x0 + vx * t, r, W - r)
                y = _bounce(y0 + vy * t, r, H - r)
                out.append((x, y, r))
            else:
                out.append((ob["c"][0], ob["c"][1], ob["r"]))
        return out


def _bounce(x, lo, hi):
    span = hi - lo
    if span <= 0:
        return lo
    ph = (x - lo) % (2 * span)
    return lo + (ph if ph <= span else 2 * span - ph)


def make_arena(kind: str, rng: np.random.Generator) -> Arena:
    W = H = 10.0
    walls = [(0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0)]
    if kind == "cluttered":
        circles = []
        while len(circles) < 8:
            c = rng.uniform(1.5, 8.5, 2); r = rng.uniform(0.35, 0.7)
            if np.hypot(*(c - [1.2, 1.2])) < r + 1.0:  # keep start clear
                continue
            if np.hypot(*(c - [8.8, 8.8])) < r + 1.0:  # keep goal clear
                continue
            circles.append({"c": c.tolist(), "r": float(r), "v": None})
        return Arena("cluttered", walls, circles, (1.2, 1.2, np.pi / 4), (8.8, 8.8))
    if kind == "corridor":
        # snake corridor built from wall segments (wall-following pressure)
        segs = walls + [(2.5, 0, 2.5, 7.0), (5.0, 3.0, 5.0, 10.0), (7.5, 0, 7.5, 7.0)]
        return Arena("corridor", segs, [], (1.2, 1.2, np.pi / 2), (8.8, 8.8))
    if kind == "u_trap":
        # concave trap directly between start and goal
        segs = walls + [(3.5, 3.5, 6.5, 3.5), (6.5, 3.5, 6.5, 6.5), (3.5, 3.5, 3.5, 6.5)]
        return Arena("u_trap", segs, [], (5.0, 1.2, np.pi / 2), (5.0, 8.8))
    if kind == "moving":
        circles = [{"c": [rng.uniform(3, 7), rng.uniform(3, 7)], "r": 0.45,
                    "v": (rng.uniform(0.2, 0.45) * _rand_dir(rng)).tolist()} for _ in range(3)]
        return Arena("moving", walls, circles, (1.2, 1.2, np.pi / 4), (8.8, 8.8))
    raise ValueError(kind)


def _rand_dir(rng):
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a), np.sin(a)])

# ------------------------------------------------------------------ robot

@dataclass
class RobotParams:
    radius: float = 0.25
    v_max: float = 1.0             # m/s
    w_max: float = 2.0             # rad/s
    tau_v: float = 0.25            # actuator lag time constants [s]
    tau_w: float = 0.15
    slip_acc: float = 1.5          # accel threshold beyond which wheels slip [m/s^2]
    slip_gain: float = 0.5         # fraction of excess accel actually delivered
    damping: float = 0.06          # per-second velocity damping (friction)
    n_rays: int = 9
    fov: float = np.pi             # front 180 degrees
    ray_max: float = 4.0


def sense(arena, t, x, y, th, p: RobotParams, noise_sigma, rng):
    """Ray distances (noisy) over the front FOV. noise_sigma is relative to ray_max."""
    angles = th + np.linspace(-p.fov / 2, p.fov / 2, p.n_rays)
    obs = arena.obstacles_at(t)
    d = np.empty(p.n_rays)
    for i, a in enumerate(angles):
        dx, dy = np.cos(a), np.sin(a)
        best = p.ray_max
        for (x1, y1, x2, y2) in arena.segments:
            best = min(best, ray_segment(x, y, dx, dy, x1, y1, x2, y2))
        for (cx, cy, r) in obs:
            best = min(best, ray_circle(x, y, dx, dy, cx, cy, r))
        d[i] = min(best, p.ray_max)
    if noise_sigma > 0:
        d = np.clip(d + rng.normal(0, noise_sigma * p.ray_max, p.n_rays), 0.0, p.ray_max)
    return d, angles


def simulate(arena: Arena, controller, p: RobotParams = None, *,
             noise_sigma=0.0, seed=0, dt=0.05, t_max=90.0, record=False):
    """Run one navigation episode.

    controller(sensors_dict) -> (v_cmd, w_cmd), where sensors_dict carries
    normalized ray distances, heading error and goal distance.
    Returns metrics dict (+ trace when record=True).
    """
    p = p or RobotParams()
    rng = np.random.default_rng(seed)
    x, y, th = arena.start
    gx, gy = arena.goal
    v = w = 0.0
    n_steps = int(t_max / dt)
    path_len = 0.0
    min_clear = np.inf
    w_hist = []
    trace = [] if record else None
    success = False
    collided = False

    for k in range(n_steps):
        t = k * dt
        d, angles = sense(arena, t, x, y, th, p, noise_sigma, rng)
        gd = np.hypot(gx - x, gy - y)
        herr = _wrap(np.arctan2(gy - y, gx - x) - th)
        sensors = {"rays": d / p.ray_max, "heading_err": herr, "goal_dist": gd,
                   "noise_sigma": noise_sigma}
        v_cmd, w_cmd = controller(sensors)
        v_cmd = float(np.clip(v_cmd, 0.0, p.v_max))
        w_cmd = float(np.clip(w_cmd, -p.w_max, p.w_max))

        # first-order actuator lag
        v_new = v + (dt / p.tau_v) * (v_cmd - v)
        w_new = w + (dt / p.tau_w) * (w_cmd - w)
        # wheel slip: cap delivered linear acceleration
        acc = (v_new - v) / dt
        if abs(acc) > p.slip_acc:
            v_new = v + np.sign(acc) * (p.slip_acc + p.slip_gain * (abs(acc) - p.slip_acc)) * dt
        # friction damping
        v_new *= (1.0 - p.damping * dt)
        v, w = v_new, w_new

        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = _wrap(th + w * dt)
        path_len += abs(v) * dt
        w_hist.append(w)

        # clearance & collision
        clear = _clearance(arena, t, x, y) - p.radius
        min_clear = min(min_clear, clear)
        if record and k % 2 == 0:
            trace.append({"t": round(t, 3), "x": round(x, 4), "y": round(y, 4),
                          "th": round(th, 4),
                          "rays": [round(float(q), 3) for q in d],
                          "obs": [[round(a, 3), round(b, 3), c] for a, b, c in arena.obstacles_at(t)]})
        if clear <= 0.0:
            collided = True
            break
        if gd < 0.35:
            success = True
            break

    w_arr = np.array(w_hist) if w_hist else np.zeros(1)
    out = {
        "success": success, "collided": collided,
        "time": (k + 1) * dt, "path_len": path_len,
        "path_ratio": path_len / max(np.hypot(gx - arena.start[0], gy - arena.start[1]), 1e-9),
        "min_clearance": float(min_clear),
        "smoothness_tv": float(np.sum(np.abs(np.diff(w_arr)))),   # total variation of omega
    }
    if record:
        out["trace"] = trace
        out["arena"] = {"name": arena.name, "segments": arena.segments,
                        "circles": arena.circles, "start": arena.start,
                        "goal": arena.goal, "size": arena.size}
    return out


def _clearance(arena, t, x, y):
    best = np.inf
    for (x1, y1, x2, y2) in arena.segments:
        best = min(best, _pt_seg(x, y, x1, y1, x2, y2))
    for (cx, cy, r) in arena.obstacles_at(t):
        best = min(best, np.hypot(x - cx, y - cy) - r)
    return best


def _pt_seg(px, py, x1, y1, x2, y2):
    ex, ey = x2 - x1, y2 - y1
    L2 = ex * ex + ey * ey
    u = 0.0 if L2 == 0 else np.clip(((px - x1) * ex + (py - y1) * ey) / L2, 0, 1)
    return np.hypot(px - (x1 + u * ex), py - (y1 + u * ey))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
