from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.features import feature_dict
from src.parse_fields import ParsedPacket


class EvidenceFeatureTests(TestCase):
    def test_coarse_provenance_features_do_not_replace_legacy_schema(self) -> None:
        packet = ParsedPacket(
            case_id="MIB-999999",
            fields={
                "applicant_name": "Ariix Solul",
                "species_code": "ARCTURIAN",
                "fee_status": "waived",
            },
            trusted_text_chars=450,
            field_sources={
                "applicant_name": "native",
                "species_code": "page_ocr",
                "fee_status": "r36_tplroi:fee_status",
            },
            untrusted_conflicts={"species_code"},
            explicit_risk_flags=["illegible_biometrics"],
            r35_repair_evidence={"species_code": {"mode": "fill_missing"}},
            positive_waiver_seen=True,
        )
        with patch("src.features.pdf_page_count", return_value=3):
            feat = feature_dict(packet, Path(__file__), "NEEDS_REVIEW")

        self.assertEqual(feat["pages"], 3)
        self.assertIn("pdf_bytes", feat)
        self.assertEqual(feat["n_fields_from_native"], 1)
        self.assertEqual(feat["n_fields_from_standard_ocr"], 1)
        self.assertEqual(feat["n_fields_from_specialized_ocr"], 1)
        self.assertEqual(feat["field_ocr_species_code"], 1)
        self.assertEqual(feat["field_specialized_fee_status"], 1)
        self.assertEqual(feat["untrusted_conflict_species_code"], 1)
        self.assertEqual(feat["r35_repair_species_code"], 1)
        self.assertEqual(feat["n_explicit_risk_flags"], 1)
        self.assertEqual(feat["positive_waiver_seen"], 1)
        self.assertEqual(feat["trusted_chars_ge_120"], 1)
        self.assertEqual(feat["trusted_chars_ge_400"], 1)
        self.assertEqual(feat["trusted_chars_ge_1000"], 0)
