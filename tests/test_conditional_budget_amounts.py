from pathlib import Path
from html.parser import HTMLParser
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]

class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.styles=[]
        self.scripts=[]
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag=='link' and 'stylesheet' in attributes.get('rel','').split():
            self.styles.append(attributes.get('href','').split('?',1)[0])
        if tag=='script' and attributes.get('src'):
            self.scripts.append(attributes['src'].split('?',1)[0])

class AmountDisplayTests(unittest.TestCase):
    def test_ticket_amount_mapping_and_missing_states(self):
        if not shutil.which('node'):self.skipTest('Node unavailable')
        subprocess.run(['node',str(ROOT/'tests/test_conditional_budget_amounts.js')],check=True)
    def test_main_forecast_assets_and_legacy_redirect(self):
        html=(ROOT/'site/index.html').read_text(encoding='utf-8')
        redirect=(ROOT/'site/conditional.html').read_text(encoding='utf-8')
        assets=AssetLinks();assets.feed(html)
        self.assertIn('nexus.css',assets.styles)
        self.assertIn('nexus-core.js',assets.scripts)
        self.assertIn('nexus.js',assets.scripts)
        self.assertLess(assets.scripts.index('nexus-core.js'),assets.scripts.index('nexus.js'))
        for asset in ('nexus.css','nexus-core.js','nexus.js'):
            self.assertTrue((ROOT/'site'/asset).is_file(),asset)
        self.assertEqual(html.count('id="races"'),1)
        self.assertIn('id="stats"',html)
        self.assertIn('data-tab="results"',html)
        self.assertIn('data-tab="model"',html)
        self.assertIn('id="model-view"',html)
        self.assertIn('location.replace',redirect)
        self.assertIn('./index.html',redirect)
        self.assertIn('条件付き3連単',redirect)
    def test_forecast_settlement_and_meeting_boundaries(self):
        if not shutil.which('node'):self.skipTest('Node unavailable')
        script=r"""
const assert=require('node:assert/strict');
const ui=require('./site/forecast.js');
const start=s=>Date.parse('2026-10-19T'+s+':00+09:00')/1000;
const meetings=[{date:'2026-10-19',venue:'夜場',start_at:start('15:30')},{date:'2026-10-19',venue:'夜場',start_at:start('20:30')},{date:'2026-10-19',venue:'朝場',start_at:start('08:30')},{date:'2026-10-19',venue:'朝場',start_at:start('10:40')},{date:'2026-10-19',venue:'昼場',start_at:start('10:45')},{date:'2026-10-19',venue:'昼場',start_at:start('16:15')},{date:'2026-10-19',venue:'深夜場',start_at:start('20:40')},{date:'2026-10-19',venue:'深夜場',start_at:start('23:30')}];
const bands=ui.meetingBands(meetings);
assert.deepEqual([...bands.values()],['night','morning','day','midnight']);
const race={shadow:{tickets:['7-5-2'],snapshot_at:'2026-10-19T08:00:00+09:00'},close_at:start('10:00'),cancelled_cars:['3'],actual:['3-1-2','7-5-2'],payouts:{'3-1-2':100,'7-5-2':1700}};
assert.deepEqual(ui.officialResults(race),['7-5-2']);
assert.equal(ui.baseSettlement(race).paid,1700);
assert.equal(ui.baseSettlement({...race,actual:['3-1-2']}).paid,null);
assert.equal(ui.baseSettlement({...race,payouts:{}}).status,'held');
assert.equal(ui.baseSettlement({...race,close_at:null}).status,'invalid');
assert.equal(ui.baseSettlement({...race,shadow:{tickets:['7-5-2']}}).status,'invalid');
const waiting={...race,actual:[],cancelled_cars:[]};
assert.equal(ui.aggregate([waiting],'uniform').roi,null);
assert.equal(ui.aggregate([waiting],'uniform').paid,null);
const closed={...waiting,shadow:{tickets:['7-5-2'],snapshot_at:0},close_at:1};
assert.equal(ui.aggregate([closed],'uniform').awaitingResult,1);
assert.equal(ui.aggregate([closed],'uniform').open,0);
const budget={status:'allocated',total_stake_yen:5000,allocations:[{buy:'7-5-2',stake_yen:5000}]};
const allocated={...race,budget,budget_settlement:{status:'settled',invested_yen:5000,payout_yen:85000,hit:true}};
assert.equal(ui.allocatedSettlement(allocated).net,80000);
assert.equal(ui.allocation({...allocated,budget:{...budget,total_stake_yen:4900}}),null);
assert.equal(ui.allocation({...allocated,budget:{...budget,allocations:[null]}}),null);
assert.equal(ui.allocatedSettlement({...allocated,budget_settlement:{status:'payout_pending'}}).paid,null);
assert.equal(ui.aggregate([allocated,waiting],'allocated').settled,1);
"""
        subprocess.run(['node','-e',script],cwd=ROOT,check=True)
    def test_pipeline_initializes_then_settles(self):
        text=(ROOT/'.github/workflows/conditional-live.yml').read_text(encoding='utf-8')
        self.assertLess(text.index('python src/conditional_budget_initial.py'),text.index('python src/conditional_live.py'))
        self.assertIn("src/conditional_budget*.py",text)
        self.assertIn('git add outputs/conditional_budget_initialization.json',text)

if __name__=='__main__':unittest.main()

