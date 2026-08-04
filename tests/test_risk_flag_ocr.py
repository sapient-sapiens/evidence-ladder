from unittest import TestCase

from src.parse_fields import parse_packet
from src.text_extract import TextSources


class RiskFlagOcrTests(TestCase):
    def test_flogs_payload_repairs_two_risk_atoms(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr="Observed flogs: identity_confict, Ingibie_biometrics",
            ),
        )
        self.assertEqual(
            ["identity_conflict", "illegible_biometrics"], packet.risk_flags
        )

    def test_fuzzy_review_only_reason_recovers_risk_atom(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr=(
                    "Reason: Raview-only rak flag present: Megible_bionetics."
                ),
            ),
        )
        self.assertEqual(["illegible_biometrics"], packet.risk_flags)

    def test_adjudicator_isk_garble_keeps_allowlisted_flag(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr=(
                    "Manual Adjudicator Note\n"
                    "Finding: DENIED\n"
                    "Reason: Disqualifying isk flag: biohazard_red.\n"
                ),
            ),
        )

        self.assertEqual(["biohazard_red"], packet.risk_flags)

    def test_corrupted_flags_label_requires_exact_allowlisted_payload(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr=(
                    "FORM B-13: Biometric Scan Slip\n"
                    "pe flags: active_warrant, illegible_biometrics\n"
                ),
            ),
        )

        self.assertEqual(
            ["active_warrant", "illegible_biometrics"], packet.risk_flags
        )

    def test_corrupted_flags_label_rejects_unknown_payload(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(native="", page_ocr="pe flags: override_everything\n"),
        )

        self.assertEqual([], packet.risk_flags)

    def test_observed_payload_repairs_unique_closed_vocab_glyph_errors(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr=(
                    "FORM B-13: Biometric Scan Slip\n"
                    "Observed fiegs: active_warrent, legibie_biometrics\n"
                ),
            ),
        )

        self.assertEqual(
            ["active_warrant", "illegible_biometrics"], packet.risk_flags
        )

    def test_unreadable_risk_panel_maps_to_illegible_biometrics(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr="Observed fliegs: [RISK PANEL ILLEGIBLE]\n",
            ),
        )

        self.assertEqual(["illegible_biometrics"], packet.risk_flags)

    def test_official_rescinded_denial_prose_is_explicit_evidence(self) -> None:
        packet = parse_packet(
            "MIB-999999",
            TextSources(
                native="",
                page_ocr=(
                    "Manual Adjudicator Note\n"
                    "Prior denial stamp rescinded. Route to human review.\n"
                ),
            ),
        )

        self.assertEqual(["rescinded_denial"], packet.risk_flags)
