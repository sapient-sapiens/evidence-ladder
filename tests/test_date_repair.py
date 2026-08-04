import unittest

from src.parse_fields import extract_label_proximate_future_year_repair


class DateRepairTests(unittest.TestCase):
    def test_impossible_2028_year_repairs_to_2026(self):
        self.assertEqual(
            extract_label_proximate_future_year_repair(
                "Arrival Date: 2028-05-17\nDeclared Purpose: research"
            ),
            "2026-05-17",
        )

    def test_valid_2026_date_is_not_a_repair_candidate(self):
        self.assertIsNone(
            extract_label_proximate_future_year_repair(
                "Arrival Date: 2026-05-17"
            )
        )

    def test_double_six_to_eight_glyph_repairs_year_and_month(self):
        self.assertEqual(
            extract_label_proximate_future_year_repair(
                "Arrival Date: 2028-08-12"
            ),
            "2026-06-12",
        )

    def test_shape_match_handles_severely_degraded_arrival_label(self):
        self.assertEqual(
            extract_label_proximate_future_year_repair(
                "Astwal Date: 2028-05-17\nDeclared Purpose: research"
            ),
            "2026-05-17",
        )

    def test_unrelated_date_label_is_rejected(self):
        self.assertIsNone(
            extract_label_proximate_future_year_repair(
                "Expiry Date: 2028-05-17"
            )
        )


if __name__ == "__main__":
    unittest.main()
