const assert=require('node:assert/strict');
const fs=require('node:fs');
const {expand,validate,validateFeed,status,copyText,MODEL,CLOUD_MODEL,MODEL_LABELS,defaultModel,newestFeed}=require('../site/high-roi.js');
const {isSelected,selectionBadge,SELECTION_RULE}=require('../site/high-roi.js');
assert.deepEqual(expand([{first:[1],second:[1,2],third:[2,3,4]},{first:[1],second:[2],third:[3]}]),['1-2-3','1-2-4']);
const data=JSON.parse(fs.readFileSync('outputs/high_roi_live.json','utf8'));
assert.equal(data.model_id,MODEL);data.races.forEach(validate);
assert.equal(data.observed.model_id,MODEL);
validateFeed(data,MODEL);
const sample=data.races.find(r=>r.forecast);
// A fully cancelled day or a first launch after closing may have no forecast.
if(sample){
assert.ok(copyText(sample).includes('100円'));
const selected=structuredClone(sample);
selected.selection={rule_id:SELECTION_RULE,selected:true,basis:'before_EX',confidence_grade:'S',created_at:new Date().toISOString(),history_through:selected.forecast.history_through,source_sha256:selected.forecast.source_sha256};
validate(selected);assert.equal(isSelected(selected),true);assert.ok(selectionBadge(selected).includes('厳選'));assert.ok(copyText(selected).includes('厳選'));
const regular=structuredClone(selected);regular.selection.selected=false;
validate(regular);assert.equal(isSelected(regular),false);assert.equal(selectionBadge(regular),'');assert.ok(copyText(regular).includes('100円'));assert.ok(!copyText(regular).includes('厳選'));
assert.deepEqual(regular.forecast,selected.forecast);
for(const change of [r=>r.selection.rule_id='wrong',r=>r.selection.selected='true',r=>r.selection.basis='after_EX',r=>r.selection.confidence_grade='A',r=>r.selection.source_sha256='wrong']){
 const invalid=structuredClone(selected);change(invalid);assert.throws(()=>validate(invalid));
}
const pending=structuredClone(sample);pending.selection={rule_id:SELECTION_RULE,selected:null,status:'unavailable'};validate(pending);assert.equal(selectionBadge(pending),'');
for(const change of [r=>r.forecast.total_yen++,r=>r.forecast.tickets.pop(),r=>r.forecast.model_id='legacy',r=>r.forecast.saved_at=new Date(r.close_at*1000+1).toISOString(),r=>r.forecast.history_through=r.date]){
 const r=structuredClone(sample);change(r);assert.throws(()=>validate(r));
}
assert.equal(status({...sample,settlement:{status:'void'}}),'返還');
assert.equal(status({...sample,settlement:null},sample.close_at*1000+1),'締切・結果待ち');
}
console.log(`High ROI site: ${data.races.length} race rows verified; formation, deadline, policy and model guards passed.`);

if(sample){
 const cloud=structuredClone(sample);
 delete cloud.selection;
 cloud.forecast.model_id=CLOUD_MODEL;
 cloud.forecast.confidence={grade:'B',status:'assessed',model_mass:.12};
 cloud.forecast.odds={};
 validate(cloud,CLOUD_MODEL);
 assert.ok(copyText(cloud).includes(MODEL_LABELS[CLOUD_MODEL]));
 assert.ok(copyText(cloud).includes('自信度B'));
 assert.equal(isSelected(cloud),false);
 const cloudFeed={schema:1,model_id:CLOUD_MODEL,today:cloud.date,generated_at:cloud.forecast.saved_at,
  observed:{model_id:CLOUD_MODEL,races:0,hits:0,stake:0,returned:0,roi_pct:null,hit_pct:null},races:[cloud]};
 validateFeed(cloudFeed,CLOUD_MODEL);
 const actual=structuredClone(cloudFeed);
 actual.races[0].forecast.odds={[cloud.forecast.tickets[0]]:123.4};
 actual.races[0].forecast.odds_at=cloud.forecast.captured_at;
 validateFeed(actual,CLOUD_MODEL);
 for(const change of [
  d=>d.observed.model_id=MODEL,
  d=>d.races[0].forecast.model_id=MODEL,
  d=>d.races.push(structuredClone(d.races[0])),
  d=>d.races[0].selection={rule_id:SELECTION_RULE,selected:true},
  d=>d.races[0].forecast.confidence.grade='C',
  d=>d.races[0].forecast.confidence.model_mass=1.1,
  d=>d.races[0].forecast.odds_at=new Date(d.races[0].close_at*1000+1).toISOString(),
  d=>d.races[0].forecast.odds['1-1-1']=100,
  d=>d.races[0].forecast.marks.pop(),
  d=>d.races[0].forecast.tickets.push(d.races[0].forecast.tickets[0]),
 ]){const bad=structuredClone(actual);change(bad);assert.throws(()=>validateFeed(bad,CLOUD_MODEL));}
 assert.throws(()=>validateFeed(cloudFeed,MODEL));
 assert.throws(()=>validateFeed(data,CLOUD_MODEL));
 // A new feed never inherits the previous model's display or metrics.
 const tomorrow=new Date(Date.parse(data.today+'T00:00:00Z')+86400000).toISOString().slice(0,10);
 const fresh={...cloudFeed,today:tomorrow,generated_at:tomorrow+'T12:00:00+09:00',races:[]};
 assert.equal(defaultModel({[MODEL]:data,[CLOUD_MODEL]:fresh},tomorrow),CLOUD_MODEL);
 assert.equal(defaultModel({[MODEL]:data},tomorrow),MODEL);
 assert.equal(defaultModel({[MODEL]:data,[CLOUD_MODEL]:cloudFeed},sample.date),MODEL);
 assert.equal(defaultModel({},tomorrow),'');
 assert.equal(newestFeed([cloudFeed,fresh]),fresh);
 const settled={...fresh,results_updated_at:tomorrow+'T13:00:00+09:00'};
 assert.equal(newestFeed([fresh,settled]),settled);
 // A publication receipt changes eligibility without regenerating the picks.
 const confirmed={...fresh,updated_at:tomorrow+'T12:02:00+09:00'};
 assert.equal(newestFeed([fresh,confirmed]),confirmed);
}

assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('EX追加前'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('98.91％'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('109.55％'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('厳選以外も'));
const html=fs.readFileSync('site/high-roi.html','utf8');
assert.ok(html.includes('id="forecast-model"'));
assert.ok(html.includes('id="legacy-research" class="research" hidden'));
assert.ok(html.includes('再構築版の公開後成績'));
assert.ok(!html.includes('daily-coverage.js'));
assert.ok(fs.readFileSync('src/site_model_navigation.py','utf8').includes('ex.html'));

if(fs.existsSync('outputs/high_roi_cloud_live.json')){
 const published=JSON.parse(fs.readFileSync('outputs/high_roi_cloud_live.json','utf8'));
 validateFeed(published,CLOUD_MODEL);
 console.log(`Reconstructed high payout feed: ${published.races.length} rows verified separately.`);
}
