/* JS port of hybrid_controller_v3 (campaign "hybridB") — the block that replaces
   hybridControl inside the viewer's reSimulate IIFE. Kept in one place so the port
   can be validated head-to-head against the Python original before it ships. */

/* ---- champion implication #8 and the Łukasiewicz t-norm -------------------- */
/* I8 = N(T(x,N(y))) with T_M, N_std  ->  1 - min(x, 1-y); signature 1001111.
   This is the same operator the I(x,y) panel plots as "Παραγόμενη #8". */
const I8 = (x,y) => 1 - Math.min(x, 1-y);
const T_LUK = (a,b) => Math.max(a + b - 1, 0);
const V3P = {beta:2.803, kappa:5.665, floor:0.2, brakeGain:0.153};

/* v3 carries state across steps (the previous commanded v feeds the stopping
   distance), so the controller is an instance, not a pure function. */
function makeHybridV3(){
  let vPrev = 0;                                  // python: state["v"]
  return function(rays, herr, gd){
    const n = rays.length, third = Math.max(Math.floor(n/3), 1);
    const right = rays.slice(0, third),
          front = rays.slice(third, n-third),
          left  = rays.slice(n-third);
    const f = Math.min(...front);
    // scan consistency: normalized total variation across a sector's rays
    const cons = a => {
      if(a.length < 2) return Math.max(V3P.floor, 1);
      let s = 0;
      for(let i=1;i<a.length;i++) s += Math.abs(a[i]-a[i-1]);
      return Math.max(V3P.floor, 1 - V3P.beta*(s/(a.length-1)));
    };
    const cL = cons(left), cR = cons(right), cF = cons(front);
    // kinematic margin: can we still stop before the front obstacle?
    const dStop = (vPrev*vPrev/(2*1.5) + 0.3)/4.0;      // a_max=1.5, 0.3 m margin, /ray_max
    const cGo = Math.max(V3P.floor, Math.min(1, V3P.kappa*(f - dStop)));

    const mu = antecedents(rays, herr, gd);
    let s = 0, sv = 0, sw = 0;
    for(let i=0;i<mu.length;i++){
      const label = RULES[i][0];
      const c = KINDS[i] === 'goal' ? cGo
              : label.indexOf('left')  >= 0 ? cL
              : label.indexOf('right') >= 0 ? cR : cF;
      const wAct = T_LUK(mu[i], clip(I8(mu[i], c), 0, 1));
      s += wAct; sv += wAct*V_CONS[i]; sw += wAct*W_CONS[i];
    }
    // python returns early WITHOUT updating state["v"] — match that exactly
    if(s < 1e-9) return [0.3, 0.5];

    let v = sv/s; const w = sw/s;
    // implication-governed brake: permission p = I(danger, front consistency)
    const p = clip(I8(1 - f, cF), 0, 1);
    v = v*(1 - V3P.brakeGain*(1 - p));
    v = v*(0.18 + 0.82*(1 - G_NEAR(gd)));               // _goal_scale
    vPrev = v;
    return [v, w];
  };
}
