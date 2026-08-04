from pathlib import Path
from unittest.mock import patch

from src.adjudicate import _counterfactual_review_denial, _rule_adjudicate
from src.parse_fields import ParsedPacket


def _packet() -> ParsedPacket:
    packet = ParsedPacket("MIB-000001")
    packet.fields = {
        "applicant_name": "Ari Voss",
        "species_code": "ORION_GRAYS",
        "home_world": "Mars Dome-7",
        "visa_class": "MED-3",
        "sponsor_id": "SPN-1111",
        "arrival_date": "2026-04-01",
        "declared_purpose": "research",
        "fee_status": "paid",
    }
    packet.risk_flags = ["identity_conflict"]
    return packet


def test_review_atom_cannot_hide_confident_meta_denial() -> None:
    packet = _packet()
    with (
        patch("src.adjudicate.feature_dict", return_value={}),
        patch("src.adjudicate.predict_adjudication", return_value=("DENIED", 0.91)),
        patch(
            "src.adjudicate.adjudication_runtime_config",
            return_value={"deny_threshold": 0.5},
        ),
    ):
        confidence = _counterfactual_review_denial(packet, Path("unused.pdf"))
    assert confidence == 0.91
    assert packet.risk_flags == ["identity_conflict"]


def test_disqualifying_atom_does_not_use_review_counterfactual() -> None:
    packet = _packet()
    packet.risk_flags.append("biohazard_red")
    assert _counterfactual_review_denial(packet, Path("unused.pdf")) is None


def test_explicit_rescission_overrides_usual_transit_denial() -> None:
    packet = _packet()
    packet.fields["visa_class"] = "TRANSIT-7"
    packet.risk_flags = ["rescinded_denial"]
    fields, reasons, hard = _rule_adjudicate(packet)
    assert hard is None
    assert fields["adjudication"] == "NEEDS_REVIEW"
    assert "rescinded_transit" in reasons


def test_other_review_atoms_do_not_override_transit_denial() -> None:
    packet = _packet()
    packet.fields["visa_class"] = "TRANSIT-7"
    fields, reasons, hard = _rule_adjudicate(packet)
    assert hard == "DENIED"
    assert "transit_visa" in reasons


def test_revoked_sponsor_requires_known_non_dip_visa() -> None:
    packet = _packet()
    packet.fields["visa_class"] = "unknown"
    packet.fields["sponsor_id"] = "SPN-0139"
    fields, reasons, hard = _rule_adjudicate(packet)
    assert hard is None
    assert fields["adjudication"] == "NEEDS_REVIEW"
    assert "missing_fields" in reasons


def test_stale_denial_requires_resolved_ocr_evidence() -> None:
    packet = _packet()
    packet.fields["arrival_date"] = "2025-01-01"
    packet.fields["declared_purpose"] = "unknown"
    packet.field_sources["visa_class"] = "ppocr_semantic:visa_class"
    packet.field_sources["arrival_date"] = "ocr_consensus:arrival_date"
    fields, reasons, hard = _rule_adjudicate(packet)
    assert hard is None
    assert fields["adjudication"] == "NEEDS_REVIEW"
    assert "unresolved_stale_evidence" in reasons


def test_native_stale_evidence_remains_denied_when_packet_is_incomplete() -> None:
    packet = _packet()
    packet.fields["arrival_date"] = "2025-01-01"
    packet.fields["declared_purpose"] = "unknown"
    packet.field_sources["visa_class"] = "native"
    packet.field_sources["arrival_date"] = "native"
    _fields, reasons, hard = _rule_adjudicate(packet)
    assert hard == "DENIED"
    assert "stale_arrival" in reasons


def test_transit_denial_requires_coherent_ocr_evidence() -> None:
    packet = _packet()
    packet.fields["visa_class"] = "TRANSIT-7"
    packet.fields["applicant_name"] = "unknown"
    packet.fields["arrival_date"] = "1900-01-01"
    packet.field_sources["visa_class"] = "ppocr_semantic:visa_class"
    packet.untrusted_conflicts = {"declared_purpose", "sponsor_id"}
    fields, reasons, hard = _rule_adjudicate(packet)
    assert hard is None
    assert fields["adjudication"] == "NEEDS_REVIEW"
    assert "unresolved_transit_evidence" in reasons


def test_incomplete_transit_without_conflicts_remains_denied() -> None:
    packet = _packet()
    packet.fields["visa_class"] = "TRANSIT-7"
    packet.fields["home_world"] = "unknown"
    packet.fields["fee_status"] = "unknown"
    packet.field_sources["visa_class"] = "ppocr_semantic:visa_class"
    _fields, reasons, hard = _rule_adjudicate(packet)
    assert hard == "DENIED"
    assert "transit_visa" in reasons
