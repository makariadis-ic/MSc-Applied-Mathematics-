import fs from 'fs';
const HTML="../viewer/robot_viewer_v2.html";
const src=fs.readFileSync(HTML,'utf8');
const m=src.match(/const reSimulate = \(\(\) => \{([\s\S]*?)\n\}\)\(\);/);
if(!m) throw new Error('sim IIFE not found');
const simulate=new Function(m[1]+'\nreturn simulate;')();
const EPS=JSON.parse(src.match(/<script type="application\/json" id="epdata">([\s\S]*?)<\/script>/)[1]);
const REF=JSON.parse(fs.readFileSync('base_noisefree_python.json','utf8'));
console.log('episode    ctrl     |        JS port          |        Python           | max dev');
for(const e of EPS) for(const c of ['sugeno','mamdani']){
  const r=simulate(e.arena,{dt:0.05,tMax:150,noise:null,record:true,controller:c});
  const p=REF[e.arena.name][c];
  const n=Math.min(r.trace.length,p.path.length);
  let dev=0;
  for(let i=0;i<n;i++)
    dev=Math.max(dev,Math.hypot(r.trace[i].x-p.path[i][0],r.trace[i].y-p.path[i][1]));
  const f=o=>`${o.time.toFixed(2)}s ${o.path_len.toFixed(3)}m`;
  console.log(`${e.arena.name.padEnd(10)} ${c.padEnd(8)} | ${f(r.metrics).padEnd(23)} | ${f(p.metrics).padEnd(23)} | ${dev.toExponential(2)} m  ${r.trace.length}/${p.path.length}f`);
}
