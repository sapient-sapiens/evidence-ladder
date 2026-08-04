from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.adjudicate import adjudicate
from src.parse_fields import parse_packet
from src.text_extract import TextSources


BASE_PACKET = """\
FORM I-8090
Case ID MIB-123456
Applicant Zed Zarnax
Species Code ORION_GRAYS
Home World Kepler-186f
Visa Class XW-1
Sponsor ID SPN-1042
Arrival Date 2026-04-17
Declared Purpose research
MIB Fee Receipt
Fee Status waived
Observed flags: none
"""


class WaiverEvidenceTests(TestCase):
    def _meta_approved(self, text: str, confidence: float = 0.9):
        packet = parse_packet("MIB-123456", TextSources(native=text))
        packet.observed_flags_seen = True
        with (
            patch("src.adjudicate.feature_dict", return_value={}),
            patch("src.adjudicate.predict_has_dq", return_value=0.0),
            patch(
                "src.adjudicate.adjudication_runtime_config",
                return_value={
                    "dq_deny_threshold": 0.4,
                    "dq_approve_ceiling": 0.4,
                    "approve_threshold": 0.58,
                    "observed_approve_threshold": 0.7,
                    "deny_threshold": 0.5,
                    "review_threshold": 0.6,
                },
            ),
            patch(
                "src.adjudicate.predict_adjudication",
                return_value=("APPROVED", confidence),
            ),
            patch(
                "src.adjudicate.calibrate_confidence",
                side_effect=lambda **kwargs: kwargs["raw_confidence"],
            ),
        ):
            prediction = adjudicate(packet, pdf_path=Path("unused.pdf"))
        return packet, prediction

    def test_non_dip_waiver_without_authorization_cannot_be_approved(self):
        packet, prediction = self._meta_approved(BASE_PACKET)

        self.assertFalse(packet.positive_waiver_seen)
        self.assertEqual("NEEDS_REVIEW", prediction["adjudication"])

    def test_visible_waiver_code_preserves_meta_approval(self):
        packet, prediction = self._meta_approved(
            BASE_PACKET + "Waiver Code DIP-WAIVER\n"
        )

        self.assertTrue(packet.positive_waiver_seen)
        self.assertEqual("APPROVED", prediction["adjudication"])

    def test_dip_waiver_code_overrides_contradictory_printed_status(self):
        text = BASE_PACKET.replace("Fee Status waived", "Fee Status unpaid")
        packet = parse_packet(
            "MIB-123456",
            TextSources(native=text + "Amount $0.00\nWaiver Code DIP-WAIVER\n"),
        )

        self.assertEqual("waived", packet.fields["fee_status"])
        self.assertTrue(packet.positive_waiver_seen)

    def test_standard_charge_without_waiver_overrides_stale_status(self):
        text = BASE_PACKET.replace("Fee Status waived", "Fee Status unpaid")
        packet = parse_packet(
            "MIB-123456",
            TextSources(native=text + "Amount $809.00\nWaiver Code N/A\n"),
        )

        self.assertEqual("paid", packet.fields["fee_status"])
        self.assertTrue(packet.field_sources["fee_status"].startswith("fee_ledger:"))

    def test_zero_dollar_explicit_unknown_is_preserved(self):
        text = BASE_PACKET.replace("Fee Status waived", "Fee Status unknown")
        packet = parse_packet(
            "MIB-123456",
            TextSources(native=text + "Amount $0.00\nWaiver Code N/A\n"),
        )

        self.assertEqual("unknown", packet.fields["fee_status"])
        self.assertTrue(packet.field_sources["fee_status"].endswith(":unknown"))

    def test_non_dip_zero_dollar_waived_without_waiver_is_unknown(self):
        packet = parse_packet(
            "MIB-123456",
            TextSources(
                native=BASE_PACKET + "Amount $0.00\nWaiver Code N/A\n"
            ),
        )

        self.assertEqual("unknown", packet.fields["fee_status"])
        self.assertTrue(packet.field_sources["fee_status"].endswith(":unknown"))

    def test_manual_review_note_unknown_overrides_stale_paid_receipt(self):
        text = BASE_PACKET.replace("Fee Status waived", "Fee Status paid")
        text += (
            "Amount $809.00\nWaiver Code N/A\n"
            "Manual Adjudicator Note\n"
            "Finding: NEEDS_REVIEW. Reason: Fee status unknown.\n"
        )

        packet = parse_packet("MIB-123456", TextSources(native=text))

        self.assertEqual("unknown", packet.fields["fee_status"])
        self.assertTrue(
            packet.field_sources["fee_status"].startswith("manual_note:")
        )

    def test_observed_flags_approval_still_requires_model_threshold(self):
        # The rule engine must not reach ``clean_packet`` on its own here, or
        # the model threshold would not be what is under test: drop the fee
        # receipt role so approval can only come from the model.
        text = BASE_PACKET.replace("MIB Fee Receipt\n", "")
        packet, prediction = self._meta_approved(text, confidence=0.65)

        self.assertNotIn("fee", packet.sources_seen)
        self.assertEqual("NEEDS_REVIEW", prediction["adjudication"])

    def test_authorized_non_dip_waiver_satisfies_the_public_fee_rule(self):
        # FIELD_MANUAL: waived is acceptable for DIP-1 *or a visible hardship
        # waiver*. With that authorization visible, a non-DIP waiver is not a
        # policy review by itself.
        packet, prediction = self._meta_approved(
            BASE_PACKET + "Waiver Code DIP-WAIVER\n", confidence=0.65
        )

        self.assertTrue(packet.positive_waiver_seen)
        self.assertEqual("APPROVED", prediction["adjudication"])
