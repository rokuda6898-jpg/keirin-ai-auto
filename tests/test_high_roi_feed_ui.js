/* Exercise real feed rendering and model switches without making network calls. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const {MODEL,CLOUD_MODEL,MODEL_LABELS}=require('../site/high-roi.js');

const html=fs.readFileSync('site/high-roi.html','utf8');
const legacy=JSON.parse(fs.readFileSync('outputs/high_roi_live.json','utf8'));
const source=fs.readFileSync('site/high-roi.js','utf8');

function fixture(){
 const day=new Date(Date.parse(legacy.today+'T00:00:00Z')+86400000).toISOString().slice(0,10);
 const row=structuredClone(legacy.races.find(r=>r.forecast));
 assert.ok(row,'fixture needs one original frozen forecast');
 row.date=day;row.race_id=row.race_id.slice(0,4)+day.replaceAll('-','');
 row.close_at=Date.parse(day+'T14:00:00+09:00')/1000;
 row.start_at=row.close_at+60;row.status='forecast';row.settlement=null;row.performance_eligible=true;
 delete row.selection;
 Object.assign(row.forecast,{model_id:CLOUD_MODEL,captured_at:day+'T11:59:00+09:00',saved_at:day+'T12:00:00+09:00',
  history_through:legacy.today,confidence:{grade:'B',status:'assessed',model_mass:.12},
  odds:{[row.forecast.tickets[0]]:123.4},odds_at:day+'T11:59:00+09:00'});
 return {schema:1,model_id:CLOUD_MODEL,today:day,generated_at:day+'T12:00:00+09:00',
  health:{stage:'ready',message:'再構築版の予想を保存しました。'},races:[row],
  observed:{model_id:CLOUD_MODEL,races:0,hits:0,HIGH_hits:0,stake:0,returned:0,void:0,pending:1,hit_pct:null,roi_pct:null}};
}

function environment(cloud){
 const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
 const elements=new Map(ids.map(id=>[id,{innerHTML:'',textContent:'',hidden:false,disabled:false,value:'',
  classList:{toggle(){},add(){}},handlers:{},addEventListener(kind,fn){this.handlers[kind]=fn;},querySelectorAll(){return [];}}]));
 const copy={addEventListener(){}};
 const document={hidden:false,getElementById(id){
  if(id==='copy')return elements.get('race-detail').innerHTML.includes('id="copy"')?copy:null;
  return elements.get(id)||null;
 }};
 const now=Date.parse(cloud.today+'T12:01:00+09:00');
 class FixedDate extends Date{constructor(...args){super(...(args.length?args:[now]));}static now(){return now;}}
 const context={document,Date:FixedDate,console,AbortSignal:{timeout(){return undefined;}},
  setInterval(){return 1;},setTimeout(){return 1;},clearTimeout(){},
  fetch:async url=>({ok:true,status:200,json:async()=>structuredClone(url.includes('high_roi_cloud_live.json')?cloud:legacy)})};
 vm.runInNewContext(source,context,{filename:'high-roi.js'});
 return {elements,context};
}

(async()=>{
 const cloud=fs.existsSync('outputs/high_roi_cloud_live.json')
  ?JSON.parse(fs.readFileSync('outputs/high_roi_cloud_live.json','utf8')):fixture();
 // Synthetic clock after the original archive guarantees the intended default;
 // the separate test_high_roi_site validates whichever real feed is available.
 const generated=fixture(),app=environment(generated),el=id=>app.elements.get(id);
 await new Promise(setImmediate);
 assert.ok(el('model-note').textContent.includes('再構築版は検証中'));
 assert.ok(el('race-list').innerHTML.includes(generated.races[0].venue));
 assert.ok(el('race-detail').innerHTML.includes('3連単フォーメーション'));
 assert.ok(el('race-detail').innerHTML.includes('自信度 B'));
 assert.ok(el('race-detail').innerHTML.includes('123.4倍'));
 assert.equal(el('legacy-research').hidden,true);
 assert.equal(el('legacy-intro').hidden,true);
 assert.equal(el('cloud-method').hidden,false);
 assert.equal(el('performance-title').textContent,'再構築版の公開後成績');
 assert.ok(el('observed').innerHTML.includes('0 / 0レース'));

 el('forecast-model').handlers.change({target:{value:MODEL}});
 assert.equal(el('performance-title').textContent,'旧方式の公開後成績');
 assert.equal(el('legacy-research').hidden,false);
 assert.equal(el('legacy-intro').hidden,false);
 assert.equal(el('cloud-method').hidden,true);
 assert.ok(el('observed').innerHTML.includes(`${legacy.observed.hits.toLocaleString('ja-JP')} / ${legacy.observed.races.toLocaleString('ja-JP')}レース`));
 assert.ok(el('notice').textContent.includes('本日分は未掲載'));

 el('forecast-model').handlers.change({target:{value:CLOUD_MODEL}});
 assert.equal(el('performance-title').textContent,'再構築版の公開後成績');
 assert.equal(el('legacy-research').hidden,true);
 assert.ok(el('forecast-model').innerHTML.includes(MODEL_LABELS[CLOUD_MODEL]));
 el('reload').handlers.click();await new Promise(setImmediate);
 assert.equal(el('performance-title').textContent,'再構築版の公開後成績');
 assert.ok(el('notice').textContent.includes('1 / 1レース'));

 if(cloud.model_id===CLOUD_MODEL&&cloud.races.some(r=>r.forecast)){
  const real=environment(cloud);await new Promise(setImmediate);
  real.elements.get('forecast-model').handlers.change({target:{value:CLOUD_MODEL}});
  assert.equal(real.elements.get('performance-title').textContent,'再構築版の公開後成績');
  assert.ok(real.elements.get('race-detail').innerHTML.includes('3連単フォーメーション'));
 }
 console.log('High ROI UI: startup, model switch, formation rendering and isolated performance passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
