import unittest
from race_features import verify_line_prediction
class LineVerificationTests(unittest.TestCase):
    def test_complete_duplicate_cancelled_and_contested(self):
        p={'lines':[{'entries':[{'numbers':[1]},{'numbers':[2]}]},{'entries':[{'numbers':[3]}]}]}
        self.assertEqual(verify_line_prediction(p,[1,2,3]),'verified')
        self.assertEqual(verify_line_prediction(p,[1,2]),'incomplete')
        self.assertEqual(verify_line_prediction({'lines':[{'entries':[{'numbers':[1]},{'numbers':[1]}]}]},[1,2]),'incomplete')
        self.assertEqual(verify_line_prediction({'lines':[{'entries':[{'numbers':[1,2]}]}]},[1,2]),'ambiguous')
        self.assertEqual(verify_line_prediction(None,[1,2]),'missing')
