import os
import unittest
from datetime import datetime
from unittest.mock import patch
from management_audit import audit, workflow_findings
from management_notify import deliver, native_notice


class ManagementTests(unittest.TestCase):
    def test_native_alert_needs_no_smtp_and_recovers_without_fake_failure(self):
        report={'findings':[{'code':'missing_shadow','message':'missing'}]}
        state,status,alert=native_notice(report,{},100)
        self.assertTrue(alert)
        self.assertFalse(native_notice(report,state,101)[2])
        self.assertTrue(native_notice(report,state,86501)[2])
        recovered,status,alert=native_notice({'findings':[]},state,102)
        self.assertFalse(alert)
        self.assertFalse(recovered['active'])
        self.assertTrue(native_notice(report,recovered,103)[2])
    def setUp(self):
        self.now = datetime.fromisoformat('2026-10-09T12:00:00+09:00')
        self.feed = {'schema':1,'schedule_date':'2026-10-09','updated_at':self.now.isoformat(),'races':[]}

    def race(self):
        return {'id':'r','date':'2026-10-09','close_at':self.now.timestamp()+600,
                'start_at':self.now.timestamp()+900,'riders':[{'car':str(i)} for i in (1,2,3)],
                'company':{'tickets':['1-2-3'],'snapshot_at':self.now.isoformat()},
                'shadow':None,'actual':[],'payouts':{}}

    def test_missing_independent_is_not_hidden_by_company(self):
        self.feed['races']=[self.race()]
        r=audit(self.feed,self.now)
        self.assertIn('missing_shadow',{x['code'] for x in r['findings']})
        self.assertEqual(r['coverage']['shadow']['missing'],1)

    def test_late_snapshot_and_invalid_car_are_detected(self):
        r=self.race();r['company']['snapshot_at']='2026-10-09T12:11:00+09:00'
        r['company']['tickets']=['1-2-9'];self.feed['races']=[r]
        codes={x['code'] for x in audit(self.feed,self.now)['findings']}
        self.assertTrue({'late_snapshot','invalid_tickets'}<=codes)

    def test_unknown_payoff_does_not_become_zero_return(self):
        r=self.race();r['actual']=['1-2-3'];r['shadow']=dict(r['company']);self.feed['races']=[r]
        result=audit(self.feed,self.now)
        self.assertIsNone(result['matched_comparison']['company']['return_yen'])
        self.assertEqual(result['matched_comparison']['company']['hits'],1)

    def test_email_dedup_recovery_and_failed_delivery(self):
        report=audit({**self.feed,'races':[self.race()]},self.now)
        env={k:'test' for k in ('SMTP_HOST','SMTP_USER','SMTP_PASSWORD','ALERT_FROM','ALERT_TO')}
        with patch.dict(os.environ,env):
            sent=[]
            state,status=deliver(report,{},100,sent.append)
            self.assertEqual(status,'sent')
            self.assertEqual(deliver(report,state,101,sent.append)[1],'suppressed_duplicate')
            healthy={**report,'findings':[]}
            self.assertEqual(deliver(healthy,state,102,sent.append)[1],'sent')
            self.assertEqual(len(sent),2)
            with self.assertRaises(RuntimeError):
                deliver(report,{},100,lambda m: (_ for _ in ()).throw(RuntimeError()))

    def test_unconfigured_mail_is_not_claimed_sent(self):
        with patch.dict(os.environ,{},clear=True):
            report=audit({**self.feed,'races':[self.race()]},self.now)
            self.assertEqual(deliver(report,{},100)[1],'not_configured')

    def test_failure_does_not_recover_just_because_retry_started(self):
        failed={'status':'completed','conclusion':'failure'}
        active={'status':'in_progress','conclusion':None,'created_at':self.now.isoformat()}
        self.assertTrue(workflow_findings('daily.yml',[active,failed],self.now.timestamp()))
        self.assertFalse(workflow_findings('daily.yml',[{'status':'completed','conclusion':'success'},failed],self.now.timestamp()))

    def test_training_and_settlement_use_their_longer_time_budget(self):
        run={'status':'in_progress','created_at':'2026-10-09T11:00:00+09:00'}
        self.assertFalse(workflow_findings('fusion-shadow-live.yml',[run],self.now.timestamp()))
        self.assertTrue(workflow_findings('site-manager.yml',[run],self.now.timestamp()))
