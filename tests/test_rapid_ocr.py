import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.rapid_ocr import (
    arrival_date_votes,
    _logical_pages,
    merge_rapid_fills,
    merge_semantic_risk_flags,
    packet_needs_embedded_rapid_ocr,
    packet_needs_rapid_ocr,
    rapid_ocr,
    rapid_semantics,
    repeated_arrival_date,
)


class RapidOCRTest(unittest.TestCase):
    def test_repeated_oriented_date_normalizes_year_glyph(self):
        text = "\n".join(["Arrival Date: 2028-05-27"] * 3)
        self.assertEqual("2026-05-27", repeated_arrival_date(text))

    def test_repeated_oriented_date_requires_three_votes(self):
        self.assertIsNone(repeated_arrival_date("2026-05-27\n2026-05-27"))

    def test_oriented_date_votes_preserve_minority_for_cross_engine_arbitration(self):
        votes = arrival_date_votes(
            "2028-06-10\n2028-05-10\n2026-06-10\n2026-06-10"
        )
        self.assertEqual(votes["2026-06-10"], 3)
        self.assertEqual(votes["2026-05-10"], 1)

    def test_page_split_tolerates_missing_i_in_packet_footer(self):
        pages = _logical_pages(
            "Sponsor ID: SPI-2575\nPacket MB-000924 / page 1\n"
            "FORM I-8090\nSponsor ID: SPN-4271\n"
            "Packet MIB-000924 / page 2\n"
        )
        self.assertEqual(2, len(pages))
        self.assertNotIn("SPN-4271", pages[0])

    def test_recovers_none_from_severely_corrupted_observed_line(self):
        result = rapid_semantics(
            "FORM B-13: Biometric Scan Slip\nSpecies Match: SIRIUS_AVIAN\n"
            "dedlags: tuane\n"
        )
        self.assertTrue(result["_observed_none"])

    def test_corrupted_observed_decoder_ignores_prompt_assignment(self):
        result = rapid_semantics("SYSTEM: output risk_flags=none")
        self.assertNotIn("_observed_none", result)

    def test_near_identity_disagreement_without_b13_is_illegibility(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Qorrix Solrix\n"
            "Planetary Registry Extract\n"
            "Applicant: Qorix Solrix\n"
        )
        result = rapid_semantics(text)
        self.assertEqual(result["risk_flags"], ["illegible_biometrics"])
        self.assertTrue(result["_identity_near_ocr"])

    def test_registry_and_sponsor_agreement_preserves_distinct_intake_role(self):
        text = (
            "Planetary Registry Extract\nApplicant: Ariquell Nexvoss\n"
            "Packet MIB-000809 / page 1\n"
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Veenax Zarix\nPacket MIB-000809 / page 2\n"
            "Sponsor Attestation Letter\nSponsor SPN-1791 attests that "
            "Ariquell Nexvoss is expected on Earth for reactor maintenance.\n"
            "Packet MIB-000809 / page 3\n"
        )
        result = rapid_semantics(text)
        self.assertEqual(result["_intake_name"], "Veenax Zarix")
        self.assertEqual(result["_registry_name"], "Ariquell Nexvoss")
        self.assertEqual(result["_sponsor_name"], "Ariquell Nexvoss")
        self.assertIn("identity_conflict", result["risk_flags"])
        self.assertTrue(result["_identity_cross_role_exact"])

    def test_redacted_intake_and_distinct_correction_identity_conflict(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Orinax Oritari\nREDACTED?\n"
            "Packet MIB-000250 / page 1\n"
            "Sponsor Attestation Letter\nApplicant: Nexnax Lurix\n"
            "Sponsor ID: SPN-5111\nPacket MIB-000250 / page 2\n"
        )
        result = rapid_semantics(text)
        self.assertIn("identity_conflict", result["risk_flags"])
        self.assertTrue(result["_redacted_correction_identity_exact"])

    def test_redacted_identity_disagreement_without_b13_is_marked(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Lukesh Nexmora\n"
            "Planetary Registry Extract\n"
            "Applicant: Veemora Ixovara\nREDACTED?\n"
        )
        self.assertTrue(rapid_semantics(text)["_redacted_identity_damage"])

    def test_review_damage_plus_missing_risk_panel_is_illegible(self):
        text = (
            "Manual Adjudicator Note\nFinding: NEEDS_REVIEW\n"
            "Reason: Packet contains damaged or contradictory visible evidence.\n"
            "FORM B-13: Biometric Scan Slip\n"
            "Observed flags: [RISK PANEL MISSING]\n"
        )
        self.assertEqual(rapid_semantics(text)["risk_flags"], ["illegible_biometrics"])

    def test_missing_b13_damage_implies_illegible(self):
        text = """Planetary Registry Extract
Biometric record damaged
MIB Fee Receipt
SAMPLE DENIAL
"""
        self.assertTrue(rapid_semantics(text)["_damage_implies_illegible"])

    def test_missing_b13_damage_requires_omitted_intake(self):
        text = """Planetary Registry Extract
Biometric record damaged
MIB Fee Receipt
FORM I-8090
SAMPLE DENIAL
"""
        self.assertNotIn("_damage_implies_illegible", rapid_semantics(text))

    def test_fuzzy_arrival_accepts_2025_window(self):
        result = rapid_semantics("Arival Date: 2025-06-27")
        self.assertEqual("2025-06-27", result["arrival_date_fill"])

    def test_garbled_visa_label_recovers_near_vocab_value(self):
        result = rapid_semantics(
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Visa TáSs' UIP-1\n"
        )
        self.assertEqual("DIP-1", result["visa_class"])

    def test_explicit_observed_flags_are_exposed_for_consensus(self):
        result = rapid_semantics(
            "FORM B-13: Biometric Scan Slip\n"
            "Observed flags: illegible_biometrics\n"
            "EMBARGO REVIEW\n"
        )
        self.assertEqual(
            ["illegible_biometrics"], result["_observed_flags_exact"]
        )

    def test_review_only_note_is_exposed_when_atom_is_destroyed(self):
        result = rapid_semantics(
            "Finding: NEEDS_REVIEW. Reason: Review-only risk flag present:"
        )
        self.assertTrue(result["_review_flag_asserted"])

    def test_review_only_note_preserves_named_allowlisted_atom(self):
        result = rapid_semantics(
            "Finding: NEEDS_REVIEW. Reason: Review-only risk flag present:\n"
            "REVIEW\nidentity_conflict.\n"
        )
        self.assertEqual(result["risk_flags"], ["identity_conflict"])
        self.assertEqual(
            result["_review_risk_atoms_exact"], ["identity_conflict"]
        )

    def test_gate_runs_until_every_critical_field_resolves(self):
        packet = SimpleNamespace(
            fields={
                "applicant_name": "Ari Voss",
                "species_code": "ORION_GRAYS",
                "home_world": "Mars Dome-7",
                "visa_class": "unknown",
                "sponsor_id": "SPN-0000",
                "arrival_date": "1900-01-01",
                "declared_purpose": "research",
                "fee_status": "paid",
            },
            risk_flags=[],
            observed_flags_seen=False,
        )
        with patch("src.rapid_ocr.shutil.which", return_value="/usr/bin/pdftoppm"):
            self.assertTrue(packet_needs_rapid_ocr(packet))
            packet.fields["visa_class"] = "XW-1"
            self.assertTrue(packet_needs_rapid_ocr(packet))
            packet.fields["sponsor_id"] = "SPN-1111"
            packet.fields["arrival_date"] = "2026-04-01"
            # A resolved packet no longer pays for the neural pass merely
            # because no biometric panel was observed: that trigger fired on
            # nearly half of all packets for 0.07 extraction points, which the
            # six-second runtime budget cannot afford.
            self.assertFalse(packet_needs_rapid_ocr(packet))

    def test_gate_runs_for_complete_but_contradictory_packet(self):
        packet = SimpleNamespace(
            fields={
                "applicant_name": "Ari Voss",
                "species_code": "ORION_GRAYS",
                "home_world": "Mars Dome-7",
                "visa_class": "XW-1",
                "sponsor_id": "SPN-1111",
                "arrival_date": "2026-04-01",
                "declared_purpose": "research",
                "fee_status": "paid",
            },
            risk_flags=[],
            observed_flags_seen=False,
            conflicts=set(),
            untrusted_conflicts={"home_world"},
        )
        with patch("src.rapid_ocr.shutil.which", return_value="/usr/bin/pdftoppm"):
            self.assertTrue(packet_needs_rapid_ocr(packet))

    def test_embedded_gate_is_reserved_for_severe_remainder(self):
        packet = SimpleNamespace(
            fields={
                "applicant_name": "Ari Voss",
                "species_code": "unknown",
                "home_world": "unknown",
                "visa_class": "unknown",
                "sponsor_id": "SPN-1111",
                "arrival_date": "2026-04-01",
                "declared_purpose": "research",
                "fee_status": "paid",
            }
        )
        with patch("src.rapid_ocr.shutil.which", return_value="/usr/bin/pdfimages"):
            self.assertTrue(packet_needs_embedded_rapid_ocr(packet))
            packet.fields["species_code"] = "ORION_GRAYS"
            self.assertFalse(packet_needs_embedded_rapid_ocr(packet))

    def test_text_from_pages_is_joined(self):
        result = SimpleNamespace(txts=("Applicant: Ari Voss", "Visa Class: XW-1"))
        engine = lambda _path: result
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page-1.jpg"
            image.write_bytes(b"jpeg")
            with (
                patch("src.rapid_ocr._engine", return_value=engine),
                patch("src.rapid_ocr._render_pages", return_value=[image]),
            ):
                text = rapid_ocr(Path(tmp) / "packet.pdf")
        self.assertEqual(text, "Applicant: Ari Voss\nVisa Class: XW-1")

    def test_semantics_prefers_registry_name_and_sponsor_visa(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Wrong Person\nVisa Class: XW-2\n"
            "Sponsor Attestation Letter\n"
            "Sponsor SPN-1234 attests that Ariquell Miravoss is expected on Earth.\n"
            "The sponsor acknowledges responsibility for class MED-3 compliance.\n"
            "Planetary Registry Extract\nApplicant: Ariquell Miravoss\n"
            "Arrival Date: 2026-06-30\n"
        )
        result = rapid_semantics(text)
        self.assertEqual(result["applicant_name"], "Ariquell Miravoss")
        self.assertEqual(result["visa_class"], "MED-3")
        self.assertEqual(result["arrival_date"], "2026-06-30")

    def test_semantics_detects_literal_sponsor_name_mismatch(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Nexkesh Oritari\n"
            "Sponsor Attestation Letter\n"
            "Sponsor SPN-1611 attests that Solquell Ixorix is expected on Earth.\n"
            "Planetary Registry Extract\nApplicant: Nexkesh Oritari\n"
        )
        self.assertEqual(
            rapid_semantics(text)["risk_flags"], ["sponsor_mismatch"]
        )

    def test_semantics_detects_identity_conflict_across_applicant_roles(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Xanul Ixonax\n"
            "Sponsor Attestation Letter\n"
            "Sponsor SPN-1331 attests that Zaix Ixozarn is expected on Earth.\n"
            "Planetary Registry Extract\nApplicant: Zaix Ixozarn\n"
        )
        self.assertEqual(rapid_semantics(text)["risk_flags"], ["identity_conflict"])

    def test_role_risk_merge_is_allowlisted_and_union_only(self):
        packet = SimpleNamespace(
            risk_flags=["biohazard_red"],
            observed_flags_seen=False,
            field_sources={},
        )
        added = merge_semantic_risk_flags(
            packet,
            {"risk_flags": ["identity_conflict", "invented_flag"]},
        )
        self.assertEqual(added, {"identity_conflict"})
        self.assertEqual(packet.risk_flags, ["biohazard_red", "identity_conflict"])

    def test_short_standalone_visa_page_is_a_correction_role(self):
        text = (
            "Visa Class: MED-3\nPacket MIB-000143 / page 2\n\n"
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Ixozarn Veequell\nVisa Class: DIP-1\n"
        )
        self.assertEqual(rapid_semantics(text)["visa_class"], "MED-3")

    def test_role_local_sponsor_ids_recover_manual_obscured_mismatch(self):
        text = (
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: [NAME CUT OUT]\nSponsor ID: SPN-5041\n"
            "Manual correction: applicant is Zaquell Zavara.\n"
            "Packet MIB-000888 / page 1\n"
            "Sponsor Attestation Letter\nSponsor ID: SPN-6041\n"
            "Applicant: [NAME CUT OUT]\nPacket MIB-000888 / page 2\n"
        )
        result = rapid_semantics(text)
        self.assertEqual(result["risk_flags"], ["sponsor_mismatch"])
        self.assertTrue(result["_sponsor_id_role_conflict_exact"])

    def test_fuzzy_arrival_date_is_fill_only_semantic(self):
        result = rapid_semantics("Amrival Date: 2026-04-29")
        self.assertNotIn("arrival_date", result)
        self.assertEqual(result["arrival_date_fill"], "2026-04-29")

    def test_registry_date_with_equal_sign_is_recovered_fill_only(self):
        result = rapid_semantics("...rivalDate:2026=02-22\nREGISTRY IMAGE")
        self.assertEqual(result["arrival_date_fill"], "2026-02-22")

    def test_rotated_fee_status_tail_is_fill_only_waiver_evidence(self):
        result = rapid_semantics("Fee1: MIB-0red\nSta\ntus: waiv\n")
        self.assertEqual(result["fee_status_fill"], "waived")

    def test_fee_status_tail_does_not_accept_unlabeled_waiver_prose(self):
        self.assertNotIn(
            "fee_status_fill",
            rapid_semantics("Applicant requests hardship waiver."),
        )

    def test_manual_fee_unknown_reason_is_preserved_as_fill_evidence(self):
        result = rapid_semantics(
            "Manual Adjudicator Note\nFinding: NEEDS_REVIEW\n"
            "Reason: Fee status unknown.\n"
        )
        self.assertEqual(result["fee_status_fill"], "unknown")

    def test_printed_fee_unknown_is_vote_not_direct_fill(self):
        result = rapid_semantics(
            "MIB Fee Receipt\nCase ID: MIB-000200\nFee Status: unknown\n"
        )
        self.assertTrue(result["_printed_fee_unknown"])
        self.assertNotIn("fee_status_fill", result)

    def test_damaged_intake_fragment_proves_cross_role_identity_risks(self):
        result = rapid_semantics(
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "Applicant: Qortari Ix\nSpecies Code: [SPECIES WHITEOUT]\n"
            "Packet MIB-000018 / page 1\n"
            "Planetary Registry Extract\nRegistry Name: Luzarn Qortari\n"
            "Packet MIB-000018 / page 2\n"
            "Sponsor Attestation Letter\n"
            "Sponsor SPN-6114 attests that Luzarn Qortari is expected on Earth "
            "for medical consult.\nPacket MIB-000018 / page 3\n"
        )
        self.assertEqual(
            sorted(set(result["risk_flags"])),
            ["identity_conflict", "illegible_biometrics"],
        )
        self.assertTrue(result["_damaged_intake_identity_exact"])

    def test_missing_biometric_roles_layout_is_structural_marker_only(self):
        result = rapid_semantics(
            "FORM I-8090: Extraterrestrial Work Authorization Intake\n"
            "MIB Fee Receipt\nSponsor Attestation Letter\n"
            "Manual Adjudicator Note\n"
        )
        self.assertTrue(result["_missing_biometric_roles_layout"])
        self.assertNotIn("risk_flags", result)

    def test_rotated_b13_glyphs_decode_with_closed_vocabulary(self):
        result = rapid_semantics(
            "Species Mstch: TRIANGULAN\n"
            "Abserved fiags: IMlaglble_blomatrics\nSCAN IMAGE"
        )
        self.assertEqual(result["risk_flags"], ["illegible_biometrics"])

    def test_split_truncated_observed_atom_exposes_fuzzy_provenance(self):
        result = rapid_semantics(
            "FORM B-13: Biometric Scan Slip\n"
            "Observed flags: ille\nmetrics\nSCAN IMAGE\n"
        )
        self.assertEqual(
            result["_observed_flags_fuzzy"], ["illegible_biometrics"]
        )

    def test_explicit_observed_atom_beats_fuzzy_none_from_another_stream(self):
        result = rapid_semantics(
            "Observed flags: illegible_biometrics\n"
            "Observed flags: nang\n"
        )
        self.assertEqual(result["_observed_flags_exact"], ["illegible_biometrics"])
        self.assertNotIn("_observed_none", result)

    def test_semantics_recognizes_degraded_planetary_heading(self):
        text = (
            "dPletary Registry Extract\n"
            "Applicant: Mirazam Miraix\n"
            "vd Uate: 2026-06-05\n"
        )
        result = rapid_semantics(text)
        # The FIT lexicon resolves the OCR-only ``Mirazam`` glyph to the
        # canonical name token used by the packet population.
        self.assertEqual(result["applicant_name"], "Mirazarn Miraix")
        self.assertEqual(result["arrival_date"], "2026-06-05")

    def test_merge_is_fill_only_and_treats_visible_placeholder_as_missing(self):
        prior = SimpleNamespace(
            fields={"sponsor_id": "SPN-7675", "applicant_name": "[NAME CUT OUT]"},
            field_sources={"sponsor_id": "page_ocr"},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
        )
        candidate = SimpleNamespace(
            fields={"sponsor_id": "SPN-1111", "applicant_name": "Ari Voss"},
            field_sources={"sponsor_id": "ppocr_ocr", "applicant_name": "ppocr_ocr"},
            explicit_risk_flags=[],
        )
        accepted = merge_rapid_fills(prior, candidate)
        self.assertEqual(prior.fields["sponsor_id"], "SPN-7675")
        self.assertEqual(prior.fields["applicant_name"], "Ari Voss")
        self.assertEqual(accepted, {"applicant_name"})

    def test_neural_only_packet_can_override_weak_name_and_union_explicit_flag(self):
        prior = SimpleNamespace(
            fields={"applicant_name": "Orikesh Anmora", "visa_class": "unknown"},
            field_sources={"applicant_name": "oriented_ocr"},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
        )
        fused = SimpleNamespace(fields={}, field_sources={}, explicit_risk_flags=[])
        neural = SimpleNamespace(
            fields={"applicant_name": "Orikesh Arimora", "visa_class": "MED-3"},
            explicit_risk_flags=["biohazard_red"],
        )
        accepted = merge_rapid_fills(prior, fused, neural)
        self.assertEqual(prior.fields["applicant_name"], "Orikesh Arimora")
        self.assertEqual(prior.fields["visa_class"], "MED-3")
        self.assertEqual(prior.risk_flags, ["biohazard_red"])
        self.assertTrue(prior.observed_flags_seen)
        self.assertEqual(accepted, {"applicant_name", "visa_class", "risk_flags"})

    def test_neural_name_never_overrides_native(self):
        prior = SimpleNamespace(
            fields={"applicant_name": "Ari Voss"},
            field_sources={"applicant_name": "native"},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
        )
        fused = SimpleNamespace(fields={}, field_sources={}, explicit_risk_flags=[])
        neural = SimpleNamespace(fields={"applicant_name": "Wrong Name"}, explicit_risk_flags=[])
        merge_rapid_fills(prior, fused, neural)
        self.assertEqual(prior.fields["applicant_name"], "Ari Voss")

    def test_neural_trusted_finding_is_preserved(self):
        prior = SimpleNamespace(
            fields={"fee_status": "paid"},
            field_sources={},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
            adjudicator_finding=None,
        )
        fused = SimpleNamespace(fields={}, field_sources={}, explicit_risk_flags=[])
        neural = SimpleNamespace(
            fields={},
            explicit_risk_flags=[],
            adjudicator_finding="APPROVED",
        )

        accepted = merge_rapid_fills(prior, fused, neural)

        self.assertEqual(prior.adjudicator_finding, "APPROVED")
        self.assertIn("adjudicator", prior.sources_seen)
        self.assertEqual(
            prior.field_sources["_adjudicator_finding"],
            "ppocr_ocr:trusted_note",
        )
        self.assertIn("adjudicator_finding", accepted)

    def test_semantic_corroboration_stamps_stronger_provenance(self):
        prior = SimpleNamespace(
            fields={"visa_class": "MED-3"},
            field_sources={"visa_class": "page_ocr"},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
        )
        fused = SimpleNamespace(fields={}, field_sources={}, explicit_risk_flags=[])
        merge_rapid_fills(prior, fused, semantics={"visa_class": "MED-3"})
        self.assertEqual(prior.field_sources["visa_class"], "ppocr_semantic:visa_class")

    def test_neural_fills_are_not_lost_when_prior_fields_are_empty(self):
        prior = SimpleNamespace(
            fields={},
            field_sources={},
            sources_seen=set(),
            conflicts=set(),
            risk_flags=[],
            observed_flags_seen=False,
        )
        fused = SimpleNamespace(fields={}, field_sources={}, explicit_risk_flags=[])
        neural = SimpleNamespace(
            fields={"applicant_name": "Orimora Luul", "sponsor_id": "SPN-8208"},
            explicit_risk_flags=[],
        )
        merge_rapid_fills(prior, fused, neural)
        self.assertEqual(
            prior.fields,
            {"applicant_name": "Orimora Luul", "sponsor_id": "SPN-8208"},
        )


if __name__ == "__main__":
    unittest.main()
