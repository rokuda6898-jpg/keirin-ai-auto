const assert=require('node:assert/strict');
const fs=require('node:fs');
const {expand,validate,status,copyText,MODEL,MODEL_LABELS}=require('../site/high-roi.js');
const {isSelected,selectionBadge,SELECTION_RULE}=require('../site/high-roi.js');
assert.deepEqual(expand([{first:[1],second:[1,2],third:[2,3,4]},{first:[1],second:[2],third:[3]}]),['1-2-3','1-2-4']);
const data=JSON.parse(fs.readFileSync('outputs/high_roi_live.json','utf8'));
assert.equal(data.model_id,MODEL);data.races.forEach(validate);
assert.equal(data.observed.model_id,MODEL);
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

if(sample){for(const id of Object.keys(MODEL_LABELS)){const r=structuredClone(sample);r.forecast.model_id=id;validate(r);assert.ok(copyText(r).includes(MODEL_LABELS[id]));}}

assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('EX追加前'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('98.91％'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('109.55％'));
assert.ok(fs.readFileSync('site/high-roi.html','utf8').includes('厳選以外も'));
assert.ok(fs.readFileSync('src/site_model_navigation.py','utf8').includes('ex.html'));
