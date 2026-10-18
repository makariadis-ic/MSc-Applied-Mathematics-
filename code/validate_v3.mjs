/* Validate the JS v3 port against the Python original.
   Both run noise-free on the recorded arenas, so any divergence is the controller
   itself rather than a different RNG draw. */
import fs from 'fs';

const HTML = "../viewer/robot_viewer_v2.html";
const src = fs.readFileSync(HTML, 'utf8');

// pull the existing reSimulate IIFE body out of the viewer
const m = src.match(/const reSimulate = \(\(\) => \{([\s\S]*?)\n\}\)\(\);/);
if(!m) throw new Error('reSimulate IIFE not found');
let body = m[1];

// swap the base-hybrid control for the v3 port
const port = fs.readFileSync('v3_port.js', 'utf8')
  .replace(/^\/\*[\s\S]*?\*\/\n/, '');            // drop the file header comment
body = body.replace(/\/\* Łukasiewicz implication[\s\S]*?\n\}\n/, port + '\n');
if(!/makeHybridV3/.test(body)) throw new Error('controller swap failed');

// simulate() calls hybridControl(...); route it through a per-episode v3 instance
body = body.replace(
  'let [vc, wc] = hybridControl(d.map(q=>q/P.ray_max), herr, gd, sigma, gamma);',
  'let [vc, wc] = _ctl(d.map(q=>q/P.ray_max), herr, gd);');
body = body.replace('let [x, y, th] = arena.start;',
                    'const _ctl = makeHybridV3();\n  let [x, y, th] = arena.start;');

const simulate = new Function(body + '\nreturn simulate;')();

const EPS = JSON.parse(src.match(
  /<script type="application\/json" id="epdata">([\s\S]*?)<\/script>/)[1]);
const REF = JSON.parse(fs.readFileSync('v3_noisefree_python.json', 'utf8'));

console.log('episode    |            JS v3 port            |          Python v3            | max dev');
for(const e of EPS){
  const r = simulate(e.arena, {sigma:0.05, dt:0.05, tMax:90, noise:null, record:true});
  const p = REF[e.arena.name];
  const n = Math.min(r.trace.length, p.path.length);
  let dev = 0;
  for(let i=0;i<n;i++)
    dev = Math.max(dev, Math.hypot(r.trace[i].x - p.path[i][0], r.trace[i].y - p.path[i][1]));
  const f = (o) => `${o.time.toFixed(2)}s ${o.path_len.toFixed(3)}m tv=${o.smoothness_tv.toFixed(2)}`;
  console.log(`${e.arena.name.padEnd(10)} | ${f(r.metrics).padEnd(32)} | ${f(p.metrics).padEnd(29)} | ${dev.toExponential(2)} m`
    + `  frames ${r.trace.length}/${p.path.length}`);
}
