from __future__ import annotations

import unittest

from solution import _reconcile_post_adjudication_risks
from src.adjudicate import adjudicate
from src.parse_fields import ParsedPacket


class PostAdjudicationRiskReconciliationTests(unittest.TestCase):
    def test_visual_disqualifier_recomputes_decision(self) -> None:
        packet = ParsedPacket(case_id="MIB-999999")
        pred = {
            "risk_flags": "biohazard_red",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.28,
        }

        _reconcile_post_adjudication_risks(pred, packet)

        self.assertEqual(pred["adjudication"], "DENIED")
        self.assertGreater(pred["confidence"], 0.28)

    def test_review_only_visual_flag_does_not_change_decision(self) -> None:
        packet = ParsedPacket(case_id="MIB-999999")
        pred = {
            "risk_flags": "illegible_biometrics",
            "adjudication": "NEEDS_REVIEW",
            "confidence": 0.72,
        }

        _reconcile_post_adjudication_risks(pred, packet)

        self.assertEqual(pred["adjudication"], "NEEDS_REVIEW")
        self.assertEqual(pred["confidence"], 0.72)

    def test_visible_clean_risk_and_paid_fee_can_resolve_ocr_uncertainty(self) -> None:
        packet = ParsedPacket(case_id="MIB-999999")
        packet.fields = {
            "applicant_name": "Test Applicant",
            "species_code": "ORION_GRAYS",
            "home_world": "Mars Dome-7",
            "visa_class": "DIP-1",
            "sponsor_id": "SPN-1234",
            "arrival_date": "2026-04-01",
            "declared_purpose": "research",
            "fee_status": "paid",
        }
        packet.field_sources = {
            "_observed_none_any": "document_role_semantic",
            "fee_status": "fee_ledger:native:paid",
        }
        packet.observed_flags_seen = True
        packet.sources_seen = {"intake", "fee"}
        packet.untrusted_conflicts = {"applicant_name"}

        result = adjudicate(packet)

        self.assertEqual(result["adjudication"], "APPROVED")

    def test_absent_fee_imputation_cannot_use_clean_risk_override(self) -> None:
        packet = ParsedPacket(case_id="MIB-999999")
        packet.fields = {
            "applicant_name": "Test Applicant",
            "species_code": "ORION_GRAYS",
            "home_world": "Mars Dome-7",
            "visa_class": "XW-1",
            "sponsor_id": "SPN-1234",
            "arrival_date": "2026-04-01",
            "declared_purpose": "research",
            "fee_status": "paid",
        }
        packet.field_sources = {
            "_observed_none_any": "document_role_semantic",
            "fee_status": "r42_fee_absent:paid",
        }
        packet.observed_flags_seen = True
        packet.sources_seen = {"intake"}

        result = adjudicate(packet)

        self.assertEqual(result["adjudication"], "NEEDS_REVIEW")


if __name__ == "__main__":
    unittest.main()
