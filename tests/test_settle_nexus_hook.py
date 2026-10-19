"""Exercise the real integration functions without importing legacy data clients."""
import argparse
import ast
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def integration(outputs, events, sync, dry_run=False):
    # Load the functions from the production source.  The unrelated historical
    # training/fetch imports are deliberately absent from this integration test.
    source = ast.parse((ROOT / 'src/settle_results.py').read_text(encoding='utf-8'))
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('main', 'sync_nexus_results_after_settlement')]
    parser = Mock()
    parser.parse_args.return_value = argparse.Namespace(min_expected_profit=0.0, dry_run=dry_run)
    scope = dict(OUTPUT_DIR=outputs, datetime=datetime, ZoneInfo=ZoneInfo, json=json,
                 argparse=types.SimpleNamespace(ArgumentParser=lambda: parser),
                 run_settlement=lambda args: events.append('legacy') or 'legacy-result')
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'settle_results_hook', 'exec'), scope)
    nexus = types.ModuleType('nexus_results_sync')
    nexus.sync_latest_results = sync
    manager = types.ModuleType('site_manager')
    manager.refresh_monitor_after_results = lambda: events.append('monitor')
    return scope, {'nexus_results_sync': nexus, 'site_manager': manager}


class SettlementNexusIntegration(unittest.TestCase):
    def test_sync_runs_after_official_settlement_before_monitor(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs, events = Path(folder), []

            def sync(path):
                self.assertEqual(path, outputs)
                events.append('nexus')
                return {'changed': True, 'updated_races': 2}

            scope, modules = integration(outputs, events, sync)
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scope['main'](), 'legacy-result')
            self.assertEqual(events, ['legacy', 'nexus', 'monitor'])
            status = json.loads((outputs / 'nexus_results_sync_status.json').read_text())
            self.assertEqual(status['status'], 'ok')
            self.assertEqual(status['report']['updated_races'], 2)

    def test_failed_sync_is_recorded_and_next_cycle_retries_without_losing_forecasts(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs, events = Path(folder), []
            forecast = outputs / 'nexus_live.json'
            frozen = b'{"prediction":"frozen-before-close"}'
            forecast.write_bytes(frozen)

            def failing_sync(path):
                events.append('nexus-error')
                raise ValueError('invalid result JSON')

            scope, modules = integration(outputs, events, failing_sync)
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(scope['main'](), 'legacy-result')
            self.assertEqual(events, ['legacy', 'nexus-error', 'monitor'])
            status_path = outputs / 'nexus_results_sync_status.json'
            status = json.loads(status_path.read_text())
            self.assertEqual(status['status'], 'error')
            self.assertEqual(status['error_type'], 'ValueError')
            self.assertIn('::warning::', log.getvalue())
            self.assertEqual(forecast.read_bytes(), frozen)

            def recovered_sync(path):
                events.append('nexus-ok')
                return {'changed': False, 'updated_races': 0}

            modules['nexus_results_sync'].sync_latest_results = recovered_sync
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scope['main'](), 'legacy-result')
            self.assertEqual(events[-3:], ['legacy', 'nexus-ok', 'monitor'])
            self.assertEqual(json.loads(status_path.read_text())['status'], 'ok')
            self.assertEqual(forecast.read_bytes(), frozen)

    def test_existing_dry_run_flag_skips_result_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs, events, sync = Path(folder), [], Mock()
            scope, modules = integration(outputs, events, sync, dry_run=True)
            with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                scope['main']()
            sync.assert_not_called()
            self.assertEqual(events, ['legacy', 'monitor'])
            self.assertFalse((outputs / 'nexus_results_sync_status.json').exists())


if __name__ == '__main__':
    unittest.main()
