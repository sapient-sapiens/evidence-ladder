from src.note_finding_ocr import decode_note_finding


def test_review_suffix_requires_official_reason_body() -> None:
    text = (
        "Manual Adjudicator Note\n"
        "Finding: 1 IEW\n"
        "Reason: Packet contains damaged or contradictory visible evidence.\n"
    )
    assert decode_note_finding(text, "MIB-000931")[0] == "NEEDS_REVIEW"


def test_review_suffix_without_reason_is_not_a_finding() -> None:
    text = "Manual Adjudicator Note\nFinding: 1 IEW\n"
    assert decode_note_finding(text, "MIB-000931")[0] is None


def test_archived_adjacent_review_suffix_is_rejected() -> None:
    text = (
        "Archived adjacent applicant - not active\n"
        "Case ID: MIB-000000\n"
        "Manual Adjudicator Note\n"
        "Finding: 1 IEW\n"
        "Reason: Packet contains damaged or contradictory visible evidence.\n"
    )
    assert decode_note_finding(text, "MIB-000931")[0] is None
