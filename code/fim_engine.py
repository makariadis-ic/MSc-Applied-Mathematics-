"""fim_engine.py — Python port of the FoFIMT implication engine (fim*.m).

Faithful reimplementation of fimBuildHandle / fimConnectiveHandles /
fimEvalCustom from the MATLAB toolkit `Implication_Application`:
  * connective formula strings over x,y with numeric lambda,w parameters,
  * the 8 structure templates + arbitrary custom expressions on the grammar
        expr := 'x' | 'y' | 'N(expr)' | 'S(expr,expr)' | 'T(expr,expr)'
  * final clamp of the core to [0,1].

Everything is vectorized over numpy arrays.
"""
from __future__ import annotations
import numpy as np

# ------------------------------------------------------------------ menus (fimConnectiveLists)
S_MENU = ["max(x,y)", "x+y-x*y", "min(x+y,1)"]                    # t-conorms
T_MENU = ["min(x,y)", "x*y", "max(x+y-1,0)"]                       # t-norms
N_MENU = ["1-x", "(1-x)/(1+lambda*x)", "(1-x^w)^(1/w)"]            # negations (Sugeno, Yager)

STRUCTURES = [
    "S(N(x),y)",                                # (S1) S-implication
    "S(N(x),T(x,y))",                           # (S2) QL-type
    "S(T(N(x),N(y)),y)",                        # (S3)
    "T(S(N(x),y),S(N(y),x))",                   # (S4) equivalence-type
    "N(T(x,N(y)))",                             # (S5) D-type dual
    "S(T(N(x),N(y)),S(T(N(x),y),T(x,y)))",      # (S6)
    "S(T(N(x),S(N(y),y)),T(x,y))",              # (S7)
    "S(T(S(N(x),x),y),T(N(y),N(x)))",           # (S8)
]

# ------------------------------------------------------------------ connective compilation
_ALLOWED = {"min": np.minimum, "max": np.maximum}

def _compile_formula(formula: str, nvars: int, lam: float = 1.0, w: float = 1.0):
    """Compile a connective formula string to a vectorized function.

    Mirrors fimConnectiveHandles: x,y are the arguments; ^ is element-wise
    power; lambda/w are substituted numerically. Uses a restricted eval
    namespace (min,max only) on numpy arrays.
    """
    expr = formula.replace("lambda", repr(float(lam)))
    # replace standalone w (not part of a name) — formulas only use w as the Yager parameter
    import re as _re
    expr = _re.sub(r"\bw\b", repr(float(w)), expr)
    expr = expr.replace("^", "**")
    code = compile(expr, f"<connective:{formula}>", "eval")
    if nvars == 2:
        def h(u, v):
            return eval(code, {"__builtins__": {}},
                        {"x": u, "y": v, "min": np.minimum, "max": np.maximum})
    else:
        def h(u):
            return eval(code, {"__builtins__": {}},
                        {"x": u, "min": np.minimum, "max": np.maximum})
    h.formula = formula
    return h


def connective_handles(Ss: str, Ts: str, Ns: str, lam: float = 1.0, w: float = 1.0):
    """Port of fimConnectiveHandles: returns (S_h, T_h, N_h)."""
    return (_compile_formula(Ss, 2), _compile_formula(Ts, 2),
            _compile_formula(Ns, 1, lam=lam, w=w))

# ------------------------------------------------------------------ expression parser (fimEvalCustom)

def _split_top_comma(s: str) -> int:
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return i
    raise ValueError(f"Comma not found in segment: {s}")


def _eval_custom(expr: str, u, v, S_h, T_h, N_h):
    if expr == "x":
        return u
    if expr == "y":
        return v
    if expr.startswith("N(") and expr.endswith(")"):
        return N_h(_eval_custom(expr[2:-1], u, v, S_h, T_h, N_h))
    if (expr.startswith("S(") or expr.startswith("T(")) and expr.endswith(")"):
        op = expr[0]
        inside = expr[2:-1]
        p = _split_top_comma(inside)
        L = _eval_custom(inside[:p], u, v, S_h, T_h, N_h)
        R = _eval_custom(inside[p + 1:], u, v, S_h, T_h, N_h)
        return S_h(L, R) if op == "S" else T_h(L, R)
    raise ValueError(f"Unknown token in expression: {expr}")


def build_implication(expr: str, Ss: str, Ts: str, Ns: str,
                      lam: float = 1.0, w: float = 1.0):
    """Port of fimBuildHandle: compile I(u,v), clamped to [0,1]."""
    S_h, T_h, N_h = connective_handles(Ss, Ts, Ns, lam, w)
    e = expr.replace(" ", "")

    def I(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.clip(_eval_custom(e, u, v, S_h, T_h, N_h), 0.0, 1.0)

    I.expr, I.S, I.T, I.N, I.lam, I.w = expr, Ss, Ts, Ns, lam, w
    return I

# ------------------------------------------------------------------ axiom & property screening

def screen_implication(I, n: int = 201, tol: float = 1e-9) -> dict:
    """Numerically verify the implication axioms and optional properties
    on an n×n grid over [0,1]^2. Returns a dict of booleans + diagnostics.

    Axioms (Fodor–Roubens / Baczyński–Jayaram):
      I1: decreasing in the first argument
      I2: increasing in the second argument
      I3: I(0,0)=1,  I4: I(1,1)=1,  I5: I(1,0)=0
    Optional: NP, IP, OP, EP, CP(N), LI (law of importation, w.r.t. its own T),
    continuity proxy (max grid jump).
    """
    g = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(g, g, indexing="ij")
    M = I(U, V)                          # M[i,j] = I(g[i], g[j])

    out = {}
    out["I3_00"] = abs(float(I(0.0, 0.0)) - 1.0) <= tol
    out["I4_11"] = abs(float(I(1.0, 1.0)) - 1.0) <= tol
    out["I5_10"] = abs(float(I(1.0, 0.0)) - 0.0) <= tol
    out["I1_dec_x"] = bool(np.all(np.diff(M, axis=0) <= tol))
    out["I2_inc_y"] = bool(np.all(np.diff(M, axis=1) >= -tol))
    out["is_implication"] = all(out[k] for k in
                                ("I1_dec_x", "I2_inc_y", "I3_00", "I4_11", "I5_10"))

    # optional properties
    out["NP"] = bool(np.max(np.abs(M[-1, :] - g)) <= 1e-6)              # I(1,y)=y
    d = np.diag(M)
    out["IP"] = bool(np.max(np.abs(d - 1.0)) <= 1e-6)                   # I(x,x)=1
    ones = M >= 1.0 - 1e-6
    leq = U <= V + 1e-12
    out["OP"] = bool(np.all(ones == leq))                               # I(x,y)=1 iff x<=y

    # EP / CP / LI on a coarser grid (cost control)
    gc = np.linspace(0.0, 1.0, 41)
    Xc, Yc, Zc = np.meshgrid(gc, gc, gc, indexing="ij")
    lhs = I(Xc, I(Yc, Zc)); rhs = I(Yc, I(Xc, Zc))
    out["EP"] = bool(np.max(np.abs(lhs - rhs)) <= 1e-6)

    N_h = _compile_formula(I.N, 1, lam=I.lam, w=I.w)
    Uc, Vc = np.meshgrid(gc, gc, indexing="ij")
    out["CP"] = bool(np.max(np.abs(I(Uc, Vc) - I(np.clip(N_h(Vc),0,1),
                                                 np.clip(N_h(Uc),0,1)))) <= 1e-6)

    T_h = _compile_formula(I.T, 2)
    li_l = I(np.clip(T_h(Xc, Yc), 0, 1), Zc); li_r = I(Xc, I(Yc, Zc))
    out["LI"] = bool(np.max(np.abs(li_l - li_r)) <= 1e-6)

    jump = max(float(np.max(np.abs(np.diff(M, axis=0)))),
               float(np.max(np.abs(np.diff(M, axis=1)))))
    out["max_grid_jump"] = round(jump, 6)
    out["continuous"] = jump <= 3.0 / (n - 1)      # heuristic: jump ~ O(grid step)
    return out
