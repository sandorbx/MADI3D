// Development only. Execute the unmodified pinned source; Node is not used by
// MADI3D or routine tests. An appended export exposes the private pixel oracle.
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const folder = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(folder, 'mipsearch.mjs'));
const hash = crypto.createHash('sha256').update(source).digest('hex');
if (hash !== '22f61ae357a4e6c13f5a7c0e1a7a3e3fb576c5c28e13d417f31752ec34dbbc38') throw Error('Reference source changed');
const oracle = await import('data:text/javascript;base64,' + Buffer.from(source + '\nexport {calculatePixelGap};').toString('base64'));
const colors = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255],
  [255,255,255], [19,19,19], [20,0,0], [21,0,0], [20,19,18], [21,20,19], [1,2,3],
  [90,0,255], [0,90,255], [112,0,255], [113,0,255], [0,137,255], [0,138,255],
  [0,254,255], [0,255,254], [0,178,255], [0,255,178], [0,179,255], [0,255,179],
  [254,255,0], [255,254,0], [178,255,0], [179,255,0], [255,178,0], [255,179,0],
  [255,0,178], [255,0,179], [2,1,3], [3,2,1], [3,1,2], [1,3,2], [2,3,1],
  [128,64,0], [254,127,0], [128,0,64], [254,0,127]];
const gaps = [];
for (const a of colors) for (const b of colors) gaps.push({a,b,gap:oracle.calculatePixelGap(...a,...b)});
const cases = [];
const red = [255,120,0], blue = [90,0,255];
function add(name, query, target, parameters={}, shape=[126,620]) {
  cases.push({name,shape,query,target,parameters:{query_threshold:0,candidate_threshold:0,color_tolerance_percent:2,
    xy_shift:0,mirror:true,minimum_overlap:0,...parameters}});
}
add('same', [[420,110,...red]], [[420,110,...red]]);
add('intensity', [[420,110,...red]], [[420,110,85,40,0]], {color_tolerance_percent:0});
add('candidate threshold strict', [[420,110,...red]], [[420,110,85,40,0]], {candidate_threshold:85});
add('query threshold strict', [[420,110,20,10,0]], [[420,110,20,10,0]], {query_threshold:20});
add('any channel excludes sum', [[420,110,10,9,8]], [[420,110,10,9,8]], {query_threshold:20});
add('equal maximum components', [[420,110,255,255,0]], [[420,110,255,255,0]]);
add('zero ratio', [[420,110,255,0,0]], [[420,110,255,0,0]]);
add('signed boundary gap', [[420,110,...blue]], [[420,110,0,90,255]], {color_tolerance_percent:0});
add('mirror', [[40,110,...red]], [[579,110,...red]]);
add('mirror disabled', [[40,110,...red]], [[579,110,...red]], {mirror:false});
add('mirror tie', [[40,110,...red]], [[40,110,...red],[579,110,...red]]);
add('shift ordered tie', [[40,110,...red]], [[38,108,...red],[40,110,...red]], {xy_shift:4});
add('mirror after shift', [[40,110,...red]], [[577,112,...red]], {xy_shift:2});
add('out of bounds denominator', [[0,110,...red],[619,125,...red]], [[2,112,...red]], {xy_shift:2});
add('zero matching shift', [[40,110,...red]], [], {xy_shift:2});
const labelPoints = [[329,99,...red],[330,99,...red],[369,89,...red],[370,89,...red],[0,100,...red],[619,90,...red]];
add('label boundaries', labelPoints, labelPoints);
add('labels only', [[0,0,...red],[619,0,...red]], [[0,0,...red],[619,0,...red]]);
add('empty query', [], []);
add('empty target', [[420,110,...red]], []);
const generated = JSON.parse(fs.readFileSync(path.join(folder,'generated-query.json')));
add('MADI3D generated query', generated.pixels, generated.pixels, {}, generated.shape);
function pixels(c, list) {
  const data = new Uint8Array(c.shape[0]*c.shape[1]*3);
  for (const [x,y,r,g,b] of list) data.set([r,g,b], (y*c.shape[1]+x)*3);
  return data;
}
const savedLog = console.log;
console.log = () => {};
for (const c of cases) {
  const p=c.parameters;
  const masks=oracle.GenerateColorMIPMasks({width:c.shape[1],height:c.shape[0],queryImage:pixels(c,c.query),
    maskThreshold:p.query_threshold,negQueryImage:null,negMaskThreshold:0,xyShift:p.xy_shift,mirrorMask:p.mirror,mirrorNegMask:false});
  c.foreground=masks.maskPositions.length;
  c.expected=oracle.ColorMIPSearch(pixels(c,c.target),p.candidate_threshold,p.color_tolerance_percent/100,masks);
  c.expected_target_masks=masks.targetMasksList;
}
console.log=savedLog;
fs.writeFileSync(path.join(folder,'oracle.json'), JSON.stringify({reference_revision:'3847c602fcc5e3ae2bae47b7308ee11263ae6cef',
  source_sha256:hash,batch_search_sha256:'6455a0183e58fa390c3df02facb4762d6453ca2a472787654f7643c8527a040a',gaps,cases},null,2)+'\n');
console.log(`Generated ${gaps.length} independent pixel gaps and ${cases.length} scoring cases.`);
