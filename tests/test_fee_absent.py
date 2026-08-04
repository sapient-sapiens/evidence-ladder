from unittest import TestCase

from src.fee_absent import apply_absent_fee_imputation
from src.parse_fields import ParsedPacket
from src.text_extract import TextSources


class AbsentFeeTests(TestCase):
    def test_absent_dip_fee_uses_crossfit_supported_paid_default(self) -> None:
        packet = ParsedPacket(
            case_id="MIB-999999",
            fields={"visa_class": "DIP-1", "fee_status": "unknown"},
        )

        changed = apply_absent_fee_imputation(
            packet, TextSources(native="Visa Class: DIP-1")
        )

        self.assertTrue(changed)
        self.assertEqual("paid", packet.fields["fee_status"])
        self.assertEqual("r42_fee_absent:paid", packet.field_sources["fee_status"])
        self.assertNotIn("fee", packet.sources_seen)

    def test_visible_fee_evidence_is_never_imputed(self) -> None:
        packet = ParsedPacket(
            case_id="MIB-999999",
            fields={"visa_class": "DIP-1", "fee_status": "unknown"},
        )

        changed = apply_absent_fee_imputation(
            packet, TextSources(native="Fee Status: waived")
        )

        self.assertFalse(changed)
        self.assertEqual("unknown", packet.fields["fee_status"])

    def test_obscured_fee_is_never_imputed(self) -> None:
        packet = ParsedPacket(
            case_id="MIB-999999",
            fields={"visa_class": "DIP-1", "fee_status": "unknown"},
        )

        changed = apply_absent_fee_imputation(
            packet, TextSources(native="[FEE STATUS OBSCURED]")
        )

        self.assertFalse(changed)
        self.assertEqual("unknown", packet.fields["fee_status"])
