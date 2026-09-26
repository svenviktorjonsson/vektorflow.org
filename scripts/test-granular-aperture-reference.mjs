import assert from 'node:assert/strict';
import {createGranularApertureReference} from '../public/previews/0.6.0/compiled/runtime-wheel-27/vf-granular-aperture-reference.mjs';

const model=createGranularApertureReference();
const initial=model.snapshot();
const startup=createGranularApertureReference();
const firstStep=startup.step(1/120);
assert.ok(firstStep.flowRate>0
  && firstStep.flowRate<firstStep.dischargeAreaPerSecond*.1,
  'the physical discharge rate must ramp up instead of jumping to full flow');
assert.ok(firstStep.upperArea<initial.upperArea,
  'fractional material must start leaving before a whole coarse guide can fall');
assert.ok(initial.opening<initial.guideDiameter,
  'the diagnostic aperture must be narrower than a guide particle');
assert.ok(initial.opening>initial.grainDiameter,
  'the aperture must pass the actual represented grains');
assert.ok(initial.dischargeAreaPerSecond>0);
const samples=[];
let flowingAt30;
let firstEmptyWithTail;
for(let frame=1;frame<=120*120;frame++){
  const state=model.step(1/120);
  assert.ok(Math.abs(state.retainedFraction-1)<1e-10,
    `mass changed at frame ${frame}: ${state.retainedFraction}`);
  const impact=Math.min(state.stream.length,Math.max(1,
    Math.ceil((0.44-state.pile[state.pile.length>>1])/(0.44/state.stream.length))));
  assert.ok(state.stream.slice(impact).every(area=>area<1e-12),
    `stream entered settled pile at frame ${frame}`);
  if(frame===120*30)flowingAt30=state;
  if(!firstEmptyWithTail&&state.upperArea===0)firstEmptyWithTail=state;
  if(frame%1200===0)samples.push({time:state.time,
    upper:state.upperArea,stream:state.streamArea,
    settling:state.settlingArea,pile:state.pileArea,
    retained:state.retainedFraction});
}
const last=model.snapshot();
assert.ok(last.upperArea<initial.upperArea&&last.pileArea>0.07);
assert.ok(firstEmptyWithTail.streamArea + firstEmptyWithTail.settlingArea > 0,
  'transport must continue after the last source parcel has released');
assert.ok(flowingAt30.streamArea / flowingAt30.dischargeAreaPerSecond > .15
  && flowingAt30.streamArea / flowingAt30.dischargeAreaPerSecond < .4,
  'falling mass should imply a gravity-scale transit time, not a one-cell speed cap');
const streamPixelScale=330/0.5,streamDy=0.44/flowingAt30.stream.length;
const laneAlphaAudits=[];
for(const widthPixels of [2,3,4]){
  const width=widthPixels/streamPixelScale;
  let visibleAlphaArea=0;
  for(const cellArea of flowingAt30.stream){
    const density=cellArea/(width*streamDy);
    const alpha=0.75*density;
    assert.ok(alpha<=1+1e-12,
      `flow width ${widthPixels}px clipped area-density alpha`);
    visibleAlphaArea+=alpha*width*streamDy;
  }
  assert.ok(Math.abs(visibleAlphaArea/0.75-flowingAt30.streamArea)<1e-12);
  laneAlphaAudits.push({widthPixels,visibleAlphaArea});
}
const dx=0.5/last.pile.length,center=Math.floor(last.pile.length/2);
const measured=[];
for(let index=center+1;index<center+23;index++){
  const left=last.pile[index],right=last.pile[index+1];
  if(left<0.02||right<0.02)continue;
  measured.push(Math.atan2(left-right,dx)*180/Math.PI);
}
const reposeMeasured=measured.reduce((sum,value)=>sum+value,0)/measured.length;
assert.ok(measured.length>8&&Math.abs(reposeMeasured-32)<2,
  `reference pile slope ${reposeMeasured}° missed 32° target`);
const narrow=createGranularApertureReference({opening:0.0016});
const blocked=createGranularApertureReference({opening:0.0015});
assert.ok(narrow.snapshot().dischargeAreaPerSecond>0,
  'an aperture only 0.1 mm wider than a represented grain must start flowing');
assert.equal(blocked.snapshot().dischargeAreaPerSecond,0,
  'a throat no wider than a represented grain must not pass material');
const benchmarkStart=performance.now();
for(let frame=0;frame<120*30;frame++){
  const state=narrow.step(1/120);
  assert.ok(Math.abs(state.retainedFraction-1)<1e-10);
  blocked.step(1/120);
}
assert.equal(blocked.snapshot().pileArea,0);
assert.ok(narrow.snapshot().pileArea>0);
console.log(JSON.stringify({passed:true,openingMm:initial.opening*1000,
  guideDiameterMm:initial.guideDiameter*1000,
  grainDiameterMm:initial.grainDiameter*1000,
  targetReposeDegrees:32,measuredReferenceReposeDegrees:reposeMeasured,
  dischargeAreaPerSecond:initial.dischargeAreaPerSecond,
  benchmark30SecondsMs:performance.now()-benchmarkStart,
  laneAlphaAudits,samples}));
