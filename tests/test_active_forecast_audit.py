import unittest
from datetime import datetime
from management_audit import audit


class ActiveForecastAuditTests(unittest.TestCase):
    def test_company_missing_is_still_an_error_after_independent_retirement(self):
        now=datetime.fromisoformat('2026-10-11T10:00:00+09:00')
        race=dict(id='r1',date='2026-10-11',close_at=now.timestamp()+600,
                  start_at=now.timestamp()+900,company=None,shadow=None,actual=[],payouts={})
        feed=dict(schema=1,schedule_date='2026-10-11',updated_at=now.isoformat(),races=[race])
        report=audit(feed,now,modes=('company',))
        codes={r['code'] for r in report['findings']}
        self.assertIn('missing_company',codes)
        self.assertNotIn('missing_shadow',codes)
        self.assertEqual(set(report['coverage']),{'company'})

    def test_old_independent_records_do_not_exclude_valid_company_results(self):
        now=datetime.fromisoformat('2026-10-11T10:00:00+09:00')
        race=dict(id='r1',date='2026-10-11',close_at=now.timestamp()+600,
                  start_at=now.timestamp()+900,riders=[dict(car=str(i)) for i in (1,2,3)],
                  company=dict(tickets=['1-2-3'],snapshot_at=now.isoformat()),
                  shadow=dict(tickets=['9-9-9'],snapshot_at='bad'),actual=['1-2-3'],payouts={'1-2-3':1000})
        feed=dict(schema=1,schedule_date='2026-10-11',updated_at=now.isoformat(),races=[race])
        report=audit(feed,now,modes=('company',))
        self.assertEqual(report['matched_comparison']['company']['races'],1)
        self.assertEqual(report['status'],'healthy')


if __name__=='__main__': unittest.main()
