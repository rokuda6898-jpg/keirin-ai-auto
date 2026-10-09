import unittest
from reconcile_live_only_provenance import unchanged_training_ast


class LiveOnlyProvenanceTests(unittest.TestCase):
    def test_live_only_change_allowed_but_training_and_globals_rejected(self):
        old='X=1\ndef train():\n return X\ndef live():\n return 1\n'
        self.assertTrue(unchanged_training_ast(old,old.replace('return 1','return 2'),'live'))
        for modified in (old.replace('return X','return X+1'),old.replace('X=1','X=2')):
            with self.assertRaises(ValueError):
                unchanged_training_ast(old,modified,'live')
