"""fis_controllers.py — Mamdani, Sugeno, and implication-enriched hybrid controllers.

All three share ONE rule base and ONE set of antecedent membership functions,
so the only manipulated variable is the inference semantics:

  Mamdani : activation = min(antecedents); clip triangle consequents; centroid.
  Sugeno  : activation = min(antecedents); constant consequents; weighted mean.
  Hybrid  : activation = T( mu_r, I(mu_r, c_r) )  — modus-ponens modulation with
            an axiomatic fuzzy implication I and rule confidence c_r; consequents
            and aggregation identical to Sugeno.

With c_r = 1 the hybrid equals Sugeno exactly (I(x,1)=1 for every axiomatic
implication), which is the degeneracy check used in the tests.

Inputs (normalized): d_front, d_left, d_right in [0,1] (min over ray sectors),
heading error in [-pi,pi]. Outputs: v in [0, v_max], w in [-w_max, w_max].
"""
from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------ membership functions

def tri(x, a, b, c):
    x = np.asarray(x, dtype=float)
    return np.maximum(np.minimum((x - a) / max(b - a, 1e-9),
                                 (c - x) / max(c - b, 1e-9)), 0.0)

def trap_l(x, a, b):     # left shoulder: 1 until a, down to 0 at b
    x = np.asarray(x, dtype=float)
    return np.clip((b - x) / max(b - a, 1e-9), 0.0, 1.0)

def trap_r(x, a, b):     # right shoulder: 0 until a, up to 1 at b
    x = np.asarray(x, dtype=float)
    return np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)

# distance terms (on normalized ray distance; ray_max = 4 m, robot radius 0.25 m)
D_NEAR = lambda d: trap_l(d, 0.18, 0.55)      # engages ~2.2 m out, saturates at 0.7 m
D_FAR  = lambda d: trap_r(d, 0.45, 0.75)
# heading error terms (rad)
H_NEG  = lambda h: trap_l(h, -0.9, -0.15)
H_ZERO = lambda h: tri(h, -0.5, 0.0, 0.5)
H_POS  = lambda h: trap_r(h, 0.15, 0.9)

# ------------------------------------------------------------------ rule base
# Each rule: (label, antecedent_fn(sensors)->mu, v_consequent, w_consequent, kind)
# kind in {"avoid", "goal"} — used for confidence assignment.
V_MAX, W_MAX = 1.0, 2.0

def _sectors(rays):
    n = len(rays)
    third = max(n // 3, 1)
    right = float(np.min(rays[:third]))
    front = float(np.min(rays[third:n - third]))
    left  = float(np.min(rays[n - third:]))
    return front, left, right

D_VNEAR = lambda d: trap_l(d, 0.1, 0.28)      # emergency band: < ~1.1 m

# "goal clear": front is far OR the goal lies nearer than whatever blocks the ray.
# g = normalized goal distance (gd / ray_max). Fires when f - g > 0 (obstruction
# is behind the goal), ramping over a robot diameter.
G_CLEAR = lambda f, g: max(float(D_FAR(f)), float(trap_r(f - g, 0.0, 0.12)))

RULES = [
    # obstacle avoidance: turn toward the freer side, brake with proximity
    ("front_near_left_freer",  lambda f,l,r,h,g: min(D_NEAR(f), trap_r(l - r, 0.0, 0.25), 1-trap_r(f - g, 0.0, 0.12)), 0.15,  1.8, "avoid"),
    ("front_near_right_freer", lambda f,l,r,h,g: min(D_NEAR(f), trap_r(r - l, 0.0, 0.25), 1-trap_r(f - g, 0.0, 0.12)), 0.15, -1.8, "avoid"),
    ("front_near_symmetric",   lambda f,l,r,h,g: min(D_NEAR(f), tri(l - r, -0.15, 0.0, 0.15), 1-trap_r(f - g, 0.0, 0.12)), 0.1, 1.8, "avoid"),
    ("front_emergency",        lambda f,l,r,h,g: min(D_VNEAR(f), 1-trap_r(f - g, 0.0, 0.12)), 0.0,  2.0, "avoid"),
    ("left_near",              lambda f,l,r,h,g: min(D_NEAR(l), D_FAR(f)),  0.5, -0.9, "avoid"),
    ("right_near",             lambda f,l,r,h,g: min(D_NEAR(r), D_FAR(f)),  0.5,  0.9, "avoid"),
    # goal seeking (heading-driven, needs clear front OR goal nearer than obstruction)
    ("goal_ahead",             lambda f,l,r,h,g: min(G_CLEAR(f,g), H_ZERO(h)),  0.95,  0.0, "goal"),
    ("goal_left",              lambda f,l,r,h,g: min(G_CLEAR(f,g), max(D_FAR(l), trap_r(f-g,0.0,0.12)), H_POS(h)), 0.5,  1.4, "goal"),
    ("goal_right",             lambda f,l,r,h,g: min(G_CLEAR(f,g), max(D_FAR(r), trap_r(f-g,0.0,0.12)), H_NEG(h)), 0.5, -1.4, "goal"),
    ("goal_hard_left",         lambda f,l,r,h,g: min(trap_r(f, 0.2, 0.45), trap_r(l, 0.25, 0.5), trap_r(h,  0.8, 1.6)), 0.1,  1.9, "goal"),
    ("goal_hard_right",        lambda f,l,r,h,g: min(trap_r(f, 0.2, 0.45), trap_r(r, 0.25, 0.5), trap_l(h, -1.6, -0.8)), 0.1, -1.9, "goal"),
]

# near-goal approach: scale speed down so the turn radius v/w stays inside
# the success threshold (prevents orbiting the goal).
G_NEAR = lambda gd: trap_l(gd, 0.25, 0.8)

def _antecedents(sensors):
    f, l, r = _sectors(sensors["rays"])
    h = sensors["heading_err"]
    g = min(sensors["goal_dist"] / 4.0, 1.0)   # normalized by ray_max
    return np.array([max(0.0, min(1.0, rule[1](f, l, r, h, g))) for rule in RULES])



def _goal_scale(sensors, v):
    """Slow down near the goal: v <- v * (0.35 + 0.65*(1 - G_NEAR))."""
    gn = float(G_NEAR(sensors["goal_dist"]))
    return v * (0.18 + 0.82 * (1.0 - gn))

# ------------------------------------------------------------------ controllers

def sugeno_controller():
    def ctl(sensors):
        mu = _antecedents(sensors)
        s = mu.sum()
        if s < 1e-9:
            return 0.3, 0.5
        v = float(np.dot(mu, [r[2] for r in RULES]) / s)
        w = float(np.dot(mu, [r[3] for r in RULES]) / s)
        return _goal_scale(sensors, v), w
    ctl.name = "sugeno"
    return ctl


def mamdani_controller(n_disc=101):
    """Same rule base; triangle output terms centered on the Sugeno singletons;
    min-activation clip, max aggregation, centroid defuzzification."""
    v_grid = np.linspace(0.0, V_MAX, n_disc)
    w_grid = np.linspace(-W_MAX, W_MAX, n_disc)
    HALF_V, HALF_W = 0.25, 0.6
    v_terms = [tri(v_grid, r[2] - HALF_V, r[2], r[2] + HALF_V) for r in RULES]
    w_terms = [tri(w_grid, r[3] - HALF_W, r[3], r[3] + HALF_W) for r in RULES]

    def ctl(sensors):
        mu = _antecedents(sensors)
        if mu.max() < 1e-9:
            return 0.3, 0.5
        agg_v = np.max([np.minimum(m, t) for m, t in zip(mu, v_terms)], axis=0)
        agg_w = np.max([np.minimum(m, t) for m, t in zip(mu, w_terms)], axis=0)
        v = float(np.trapezoid(agg_v * v_grid, v_grid) / max(np.trapezoid(agg_v, v_grid), 1e-9))
        w = float(np.trapezoid(agg_w * w_grid, w_grid) / max(np.trapezoid(agg_w, w_grid), 1e-9))
        return _goal_scale(sensors, v), w
    ctl.name = "mamdani"
    return ctl


def hybrid_controller(I, T_norm, *, conf_mode="noise_adaptive", gamma=3.0,
                      c_avoid=1.0, c_goal=1.0):
    """Implication-enriched Sugeno.

    activation w_r = T( mu_r, I(mu_r, c_r) ).
    conf_mode:
      "uniform"        : c_r = (c_avoid | c_goal) constants per rule kind.
      "noise_adaptive" : goal rules lose confidence with sensor noise
                         c_goal(t) = max(0.5, 1 - gamma*sigma); avoidance rules
                         keep c=1 (safety-critical, conservative).
    """
    kinds = [r[4] for r in RULES]

    def ctl(sensors):
        mu = _antecedents(sensors)
        if conf_mode == "noise_adaptive":
            cg = max(0.5, 1.0 - gamma * sensors.get("noise_sigma", 0.0))
            c = np.array([1.0 if k == "avoid" else cg for k in kinds])
        else:
            c = np.array([c_avoid if k == "avoid" else c_goal for k in kinds])
        w_act = np.asarray(T_norm(mu, np.clip(I(mu, c), 0.0, 1.0)))
        s = w_act.sum()
        if s < 1e-9:
            return 0.3, 0.5
        v = float(np.dot(w_act, [r[2] for r in RULES]) / s)
        w = float(np.dot(w_act, [r[3] for r in RULES]) / s)
        return _goal_scale(sensors, v), w
    ctl.name = "hybrid"
    return ctl


def hybrid_controller_v2(I, T_norm, *, alpha=6.0, floor=0.35):
    """Implication-enriched Sugeno, v2: per-rule, state-dependent confidence
    estimated from data the robot actually has (no oracle noise).

    Confidence model: each rule's premise reliability decays with the *estimated*
    local sensor noise on the sectors that rule reads. The estimate is the
    median absolute deviation between successive scans (a standard proxy the
    robot can compute online), kept in a short EMA.

      c_r = max(floor, 1 - alpha * sigma_hat * s_r)

    where s_r in [0,1] is the rule's sensor dependence (1 for distance-driven
    avoid rules, ray-sector weighted; heading-driven goal rules have s_r from
    the front sector only, since heading error comes from odometry/goal fix).

    activation: w_r = T(mu_r, I(mu_r, c_r)); output = weighted average (Sugeno).
    Degeneracy: alpha=0 (or sigma_hat=0) -> c_r=1 -> classical Sugeno.
    """
    kinds = [r[4] for r in RULES]
    # Attenuation policy: under estimated sensor noise, *goal* rules lose
    # confidence (aggressive, rely on the front-clearance gate being right),
    # while avoid rules stay near-fully trusted (safety-critical, conservative
    # direction). This mirrors v1 but is per-state and estimator-driven.
    s_dep = np.array([0.25 if k == "avoid" else 1.0 for k in kinds])
    state = {"prev": None, "ema": 0.0}

    def ctl(sensors):
        rays = np.asarray(sensors["rays"], dtype=float)
        if state["prev"] is not None:
            # scan-to-scan MAD, robust to real geometry change of ~1 ray
            d = np.abs(rays - state["prev"])
            mad = float(np.median(d))
            state["ema"] = 0.9 * state["ema"] + 0.1 * mad
        state["prev"] = rays.copy()
        # scale: successive-scan diff of iid N(0,s) has MAD ~ 0.954*s*sqrt(2)
        sigma_hat = state["ema"] / 1.35
        mu = _antecedents(sensors)
        c = np.maximum(floor, 1.0 - alpha * sigma_hat * s_dep)
        w_act = np.asarray(T_norm(mu, np.clip(I(mu, c), 0.0, 1.0)))
        s = w_act.sum()
        if s < 1e-9:
            return 0.3, 0.5
        v = float(np.dot(w_act, [r[2] for r in RULES]) / s)
        w = float(np.dot(w_act, [r[3] for r in RULES]) / s)
        return _goal_scale(sensors, v), w
    ctl.name = "hybrid_v2"
    return ctl


def hybrid_controller_v3(I, T_norm, *, beta=2.0, kappa=4.0, floor=0.2,
                         brake_gain=0.6, safety_asym=False, floor_avoid=None,
                         act_mode="impl", brake_mode="impl"):
    """v3: state-rich per-rule confidence + implication-governed speed.

    Two confidence channels, both computable on-robot and both varying
    strongly within an episode (so the implication surface I(mu, c) is
    exercised across its whole domain, not at a single operating point):

      * avoid rules:  c = local scan consistency. Adjacent rays over the
        rule's sector should agree if the percept is real; disagreement
        (normalized total variation of the sector rays) lowers confidence.
        c_avoid = max(floor, 1 - beta * TV_sector)
      * goal rules:   c = kinematic safety margin. Confidence that pursuing
        the goal is compatible with stopping in time:
        c_goal = max(floor, min(1, kappa * (front - d_stop) )), where
        d_stop = v^2/(2 a_max) + reaction margin, normalized by ray_max.

    Speed governor: the final v is additionally scaled by the *inferred
    permission* p = I(danger, c_front) with danger = 1 - front clearance:
        v <- v * (1 - brake_gain * (1 - p))
    For OP implications p==1 whenever danger <= c_front (crisp permission);
    for smooth implications permission degrades gradually — the implication
    shape directly shapes the braking law.

    Degeneracy: beta=0, kappa large, brake_gain=0  ->  classical Sugeno.
    """
    kinds = [r[4] for r in RULES]
    state = {"v": 0.0}

    def ctl(sensors):
        rays = np.asarray(sensors["rays"], dtype=float)
        n = len(rays)
        third = max(n // 3, 1)
        sec = {"right": rays[:third], "front": rays[third:n-third], "left": rays[n-third:]}
        f = float(np.min(sec["front"]))
        # scan-consistency channel (per sector TV, normalized by sector span)
        fl_av = floor if floor_avoid is None else floor_avoid
        def cons(a):
            return max(fl_av, 1.0 - beta * float(np.mean(np.abs(np.diff(a)))))
        c_av = {"left": cons(sec["left"]), "right": cons(sec["right"]),
                "front": cons(sec["front"])}
        # kinematic-margin channel
        v_now = state["v"]
        d_stop = (v_now * v_now / (2.0 * 1.5) + 0.3) / 4.0   # a_max=1.5, margin 0.3 m, /ray_max
        c_go = max(floor, min(1.0, kappa * (f - d_stop)))
        c = []
        for name, k in zip([r[0] for r in RULES], kinds):
            if k == "goal":
                c.append(c_go)
            elif "left" in name:
                c.append(c_av["left"])
            elif "right" in name:
                c.append(c_av["right"])
            else:
                c.append(c_av["front"])
        c = np.array(c)
        mu = _antecedents(sensors)
        if act_mode == "impl":
            w_act = np.asarray(T_norm(mu, np.clip(I(mu, c), 0.0, 1.0)))
        elif act_mode == "gain":      # ablation: confidence as plain multiplicative gain
            w_act = mu * c
        else:                          # "plain": classical Sugeno activation
            w_act = mu
        if safety_asym:
            # avoid rules use T=min (weak signals survive); goal rules keep T_norm
            is_goal = np.array([k == "goal" for k in kinds])
            w_min = np.minimum(mu, np.clip(I(mu, c), 0.0, 1.0))
            w_act = np.where(is_goal, w_act, w_min)
        s = w_act.sum()
        if s < 1e-9:
            return 0.3, 0.5
        v = float(np.dot(w_act, [r[2] for r in RULES]) / s)
        w = float(np.dot(w_act, [r[3] for r in RULES]) / s)
        # implication-governed brake
        danger = 1.0 - f
        if brake_mode == "impl":
            p = float(np.clip(I(np.array(danger), np.array(c_av["front"])), 0.0, 1.0))
        elif brake_mode == "linear":   # ablation: non-implication brake, same inputs
            p = float(np.clip(1.0 - danger * (1.0 - c_av["front"]) - 0.5*max(0.0, danger - c_av["front"]), 0.0, 1.0))
        else:                          # "none"
            p = 1.0
        v = v * (1.0 - brake_gain * (1.0 - p))
        v = _goal_scale(sensors, v)
        state["v"] = v
        return v, w
    ctl.name = "hybrid_v3"
    return ctl


def hybrid_controller_v4(I, T_norm, *, beta=2.0, kappa=4.0, brake_gain=0.4,
                         floor=0.2, interact=True):
    """v4 = v3 + implication-mediated rule interaction.

    Beyond per-rule confidence (v3 channels), the implication arbitrates
    *between* rule groups: goal rules are additionally weighted by the degree
    to which the avoidance evidence PERMITS goal pursuit,

        permit = I( mu_avoid_max , c_front )

    i.e. "if the avoidance premise holds, how compatible is trusting the
    front percept with continuing?" For an OP implication this is a crisp
    gate (permit=1 iff mu_avoid_max <= c_front); for smooth implications a
    graded arbitration. This is inference the weighted average cannot
    express: rule groups talk to each other THROUGH the implication.

        w_goal <- T(w_goal, permit)

    Degeneracy: interact=False, beta=0, kappa->inf, brake_gain=0 -> Sugeno.
    """
    kinds = [r[4] for r in RULES]
    state = {"v": 0.0}

    def ctl(sensors):
        rays = np.asarray(sensors["rays"], dtype=float)
        n = len(rays)
        third = max(n // 3, 1)
        sec = {"right": rays[:third], "front": rays[third:n-third], "left": rays[n-third:]}
        f = float(np.min(sec["front"]))
        fl_av = floor if floor_avoid is None else floor_avoid
        def cons(a):
            return max(fl_av, 1.0 - beta * float(np.mean(np.abs(np.diff(a)))))
        c_av = {"left": cons(sec["left"]), "right": cons(sec["right"]),
                "front": cons(sec["front"])}
        v_now = state["v"]
        d_stop = (v_now * v_now / (2.0 * 1.5) + 0.3) / 4.0
        c_go = max(floor, min(1.0, kappa * (f - d_stop)))
        c, mu = [], _antecedents(sensors)
        for name, k in zip([r[0] for r in RULES], kinds):
            if k == "goal":
                c.append(c_go)
            elif "left" in name:
                c.append(c_av["left"])
            elif "right" in name:
                c.append(c_av["right"])
            else:
                c.append(c_av["front"])
        c = np.array(c)
        w_act = np.asarray(T_norm(mu, np.clip(I(mu, c), 0.0, 1.0)))
        if interact:
            is_goal = np.array([k == "goal" for k in kinds])
            mu_avoid_max = float(np.max(mu[~is_goal])) if (~is_goal).any() else 0.0
            permit = float(np.clip(I(np.array(mu_avoid_max), np.array(c_av["front"])), 0.0, 1.0))
            w_act = np.where(is_goal, T_norm(w_act, permit), w_act)
        s = w_act.sum()
        if s < 1e-9:
            return 0.3, 0.5
        v = float(np.dot(w_act, [r[2] for r in RULES]) / s)
        w = float(np.dot(w_act, [r[3] for r in RULES]) / s)
        danger = 1.0 - f
        if brake_mode == "impl":
            p = float(np.clip(I(np.array(danger), np.array(c_av["front"])), 0.0, 1.0))
        elif brake_mode == "linear":   # ablation: non-implication brake, same inputs
            p = float(np.clip(1.0 - danger * (1.0 - c_av["front"]) - 0.5*max(0.0, danger - c_av["front"]), 0.0, 1.0))
        else:                          # "none"
            p = 1.0
        v = v * (1.0 - brake_gain * (1.0 - p))
        v = _goal_scale(sensors, v)
        state["v"] = v
        return v, w
    ctl.name = "hybrid_v4"
    return ctl


def hybrid_controller_v7(I, T_norm, *, beta=2.803, kappa=5.665, floor=0.2,
                         brake_gain=0.35, dt=0.05, conf_tau=0.3,
                         prog_bias=0.6, prog_gain=0.6, prog_tau=6.0,
                         progress=True, fallback="sugeno"):
    """v7 = v3 with the two defects of its confidence stage repaired.

    The inference semantics are IDENTICAL to v3 and to the thesis:

        w_r = T( mu_r , I(mu_r, c_r) ) ,  Sugeno weighted average,
        implication-governed brake  v <- v (1 - g (1 - I(1-f, c_front))).

    Same rule base, same membership functions, same defuzzification. What
    changes is only the estimator that produces c_r, plus what happens when the
    Lukasiewicz dead zone leaves nothing behind.

    (1) Progress confidence on the goal rules.  v3's goal confidence was the
        kinematic stopping margin alone, which saturates at 1 almost everywhere
        (measured: mean 0.95-1.00 on all four arenas), so the goal rules ran
        unattenuated while the avoid rules were throttled — the weighted
        average tilted toward v=0.95, w=0 and drove the robot into walls and
        into dead ends. v7 conjoins the margin with the degree to which chasing
        the goal is actually working:

            ratio = EMA(goal-distance reduction) / EMA(distance travelled)
            c_prog = clip(prog_bias + prog_gain * ratio, floor, 1)
            c_goal = min(c_kinematic, c_prog)

        Both terms are odometry-level quantities the robot already has. In the
        top pocket of the snake corridor the ratio collapses, the dead zone
        removes the goal rules, the avoidance rules wall-follow the robot out,
        and progress restores the confidence. A graded go-to-goal / wall-follow
        alternation obtained from the confidence argument of the implication,
        with no rule added and no explicit mode switch.

    (2) Graceful fallback.  At sigma=0.3 the dead zone empties the activation
        set in 26-33 % of the steps in the corridor, and v3 then returns the
        hard-coded constant (0.3, 0.5) — a fixed right turn it was never
        designed to execute for a third of an episode. v7 instead defers to the
        classical reading of the same rule base, which is exactly this
        architecture's own c_r = 1 degeneracy: when no rule has
        confidence-backed activation, the system has no modulation to apply, so
        it applies none.

    (3) The confidence estimate is filtered (conf_tau) and the brake gain
        raised. c_r is an estimate; filtering it stops the dead zone turning
        estimator chatter into rules snapping on and off.

    Degeneracy: floor=1 forces every c_r = 1, hence I(mu,1)=1 and T(mu,1)=mu,
    hence classical Sugeno (verified exactly on all four arenas).

    Ablation switches: progress=False and fallback="const" restore v3 exactly.
    """
    kinds = [r[4] for r in RULES]
    names = [r[0] for r in RULES]
    lam = dt / max(prog_tau, 1e-6)
    clam = 1.0 if conf_tau <= 0 else dt / conf_tau
    st = {"v": 0.0, "gd": None, "dgd": 0.0, "trav": 0.0, "warm": 0, "rel": None}

    def ctl(sensors):
        rays = np.asarray(sensors["rays"], dtype=float)
        n = len(rays)
        third = max(n // 3, 1)
        f = float(np.min(rays[third:n - third]))

        # --- percept reliability per sector (v3's scan-consistency channel) ---
        tv = lambda a: float(np.mean(np.abs(np.diff(a))))
        rel = {"right": max(floor, 1.0 - beta * tv(rays[:third])),
               "front": max(floor, 1.0 - beta * tv(rays[third:n - third])),
               "left":  max(floor, 1.0 - beta * tv(rays[n - third:]))}
        if st["rel"] is None or clam >= 1.0:
            st["rel"] = dict(rel)
        else:
            for k in rel:
                st["rel"][k] += clam * (rel[k] - st["rel"][k])
        rel = st["rel"]

        # --- goal confidence: kinematic margin AND progress ---
        d_stop = (st["v"] ** 2 / (2.0 * 1.5) + 0.3) / 4.0
        c_goal = max(floor, min(1.0, kappa * (f - d_stop)))
        gd = float(sensors["goal_dist"])
        if progress:
            if st["gd"] is not None:
                st["dgd"] += lam * ((st["gd"] - gd) - st["dgd"])
                st["trav"] += lam * (st["v"] * dt - st["trav"])
                st["warm"] += 1
            st["gd"] = gd
            if st["warm"] > int(prog_tau / dt):        # wait for evidence
                ratio = st["dgd"] / max(st["trav"], 1e-6)
                c_goal = min(c_goal, min(1.0, max(floor, prog_bias + prog_gain * ratio)))

        c = np.array([c_goal if k == "goal" else
                      rel["left"] if "left" in nm else
                      rel["right"] if "right" in nm else rel["front"]
                      for nm, k in zip(names, kinds)])

        mu = _antecedents(sensors)
        w_act = np.asarray(T_norm(mu, np.clip(I(mu, c), 0.0, 1.0)))
        s = w_act.sum()
        if s < 1e-9:
            if fallback != "sugeno":
                return 0.3, 0.5
            w_act, s = mu, mu.sum()                     # c_r = 1 degeneracy
            if s < 1e-9:
                return 0.3, 0.5
        v = float(np.dot(w_act, [r[2] for r in RULES]) / s)
        w = float(np.dot(w_act, [r[3] for r in RULES]) / s)
        p = float(np.clip(I(np.array(1.0 - f), np.array(rel["front"])), 0.0, 1.0))
        v = _goal_scale(sensors, v * (1.0 - brake_gain * (1.0 - p)))
        st["v"] = v
        return v, w

    ctl.name = "hybrid_v7"
    return ctl
