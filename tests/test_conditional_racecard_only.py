import inspect
import unittest
from unittest.mock import patch

import pandas as pd

import conditional_input_refresh as capture

URL = "https://www.winticket.jp/keirin/test/racecard/2026101100/1/1"


def source(size=7):
    return {"schedule": {"date": "20261011"},
            "race": {"id": "010020261011", "number": 1, "entriesNumber": size},
            "entries": [{"number": car} for car in range(1, size + 1)]}


def entrants(size=7):
    return [{"race_id": "010020261011", "date": "2026-10-11", "car_no": car,
             "player_id": str(100 + car), "entries_number": size,
             "close_at": 1791682800} for car in range(1, size + 1)]


class ConditionalRacecardOnlyTests(unittest.TestCase):
    def capture_fixture(self, data, rows):
        with patch.object(capture, "http_get", return_value="html") as http, \
             patch.object(capture, "extract_preloaded_state", return_value={}), \
             patch.object(capture, "find_query_data", return_value=data), \
             patch.object(capture, "get_venue_name", return_value="test"), \
             patch.object(capture, "build_entry_rows", return_value=rows) as builder, \
             patch("market_odds_sources.verify_market", side_effect=AssertionError("market access")) as market:
            actual, odds = capture.fetch_racecard_only(URL)
            http.assert_called_once_with(URL, attempts=2)
            self.assertFalse(builder.call_args.kwargs["include_results"])
            market.assert_not_called()
            self.assertEqual(actual, rows)
            self.assertEqual(odds, [])
        return actual

    def test_full_seven_and_nine_car_cards_need_no_market(self):
        for size in (7, 9):
            with self.subTest(size=size):
                rows = self.capture_fixture(source(size), entrants(size))
                self.assertEqual(len(rows), size)
                self.assertTrue(capture.valid_race(pd.DataFrame(rows), "010020261011",
                                                  "2026-10-11", 1791682800))

    def test_absent_zero_placeholder_is_ignored_not_real_car(self):
        data = source()
        data["entries"].append({"number": 0, "absent": True})
        self.capture_fixture(data, entrants())

    def test_missing_and_duplicate_source_cars_fail_closed(self):
        for numbers in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 6]):
            data = source()
            data["entries"] = [{"number": car} for car in numbers]
            with self.subTest(numbers=numbers), self.assertRaisesRegex(ValueError, "incomplete raw"):
                self.capture_fixture(data, entrants())

    def test_incomplete_built_entries_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "incomplete race field"):
            self.capture_fixture(source(), entrants()[:-1])

    def test_unknown_field_size_and_infinite_deadline_are_rejected(self):
        frame = pd.DataFrame(entrants())
        self.assertFalse(capture.valid_race(frame.drop(columns="entries_number"),
                                           "010020261011", "2026-10-11", 1791682800))
        frame["close_at"] = float("inf")
        self.assertFalse(capture.valid_race(frame, "010020261011", "2026-10-11", 1791682800))

    def test_refresh_default_is_the_odds_free_path(self):
        self.assertIs(inspect.signature(capture.refresh).parameters["fetcher"].default,
                      capture.fetch_racecard_only)


if __name__ == "__main__":
    unittest.main()
