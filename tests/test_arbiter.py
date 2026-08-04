from src.arbiter import LABELS, UTILITY, apply_arbiter, build_features


class _ConstantBinaryModel:
    def __init__(self, positive):
        self.positive = positive

    def predict_proba(self, rows):
        return [[1.0 - self.positive, self.positive] for _ in rows]


def _state(**overrides):
    state = {
        "case_id": "MIB-000000",
        "rule_decision": "NEEDS_REVIEW",
        "hard": None,
        "final_decision": "NEEDS_REVIEW",
        "final_reasons": ["insufficient_evidence"],
        "raw_confidence": 0.28,
        "runtime_confidence": 0.3,
        "missing": [],
        "meta_label": "APPROVED",
        "meta_conf": 0.6,
        "dq_prob": 0.05,
        "adjudicator_finding": None,
        "risk_flags": [],
        "conflicts": [],
        "untrusted_conflicts": [],
        "sources_seen": ["intake", "fee"],
        "observed_flags_seen": True,
        "positive_waiver_seen": False,
        "injection_heavy": False,
        "trusted_text_chars": 900,
        "packet_fields": {
            "applicant_name": "Zed Zarnax",
            "species_code": "ORION_GRAYS",
            "home_world": "Kepler-186f",
            "visa_class": "XW-2",
            "sponsor_id": "SPN-1042",
            "arrival_date": "2026-04-17",
            "declared_purpose": "research",
            "fee_status": "paid",
        },
        "field_sources": {"applicant_name": "native", "fee_status": "page_ocr"},
    }
    state.update(overrides)
    return state


def test_features_exclude_envelope_and_identity_inputs():
    feat = build_features(_state())
    assert "pages" not in feat
    assert "pdf_bytes" not in feat
    assert not any("case_id" in name for name in feat)
    assert all(isinstance(v, float) for v in feat.values())


def test_features_describe_policy_content_and_provenance():
    feat = build_features(_state())
    assert feat["visa_XW-2"] == 1.0
    assert feat["fee_paid"] == 1.0
    assert feat["native_applicant_name"] == 1.0
    assert feat["stdocr_fee_status"] == 1.0
    assert feat["why_insufficient_evidence"] == 1.0
    assert feat["sponsor_wellformed"] == 1.0


def test_missing_artifact_leaves_prediction_untouched(monkeypatch):
    import src.arbiter as arbiter

    monkeypatch.setattr(arbiter, "load_arbiter", lambda: None)
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state())
    assert pred == {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}


def test_trusted_review_finding_keeps_precedence(monkeypatch):
    import src.arbiter as arbiter

    monkeypatch.setattr(arbiter, "load_arbiter", lambda: {"margin": 0.0})
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.99, 0.005, 0.005])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state(adjudicator_finding="NEEDS_REVIEW"))
    assert pred["adjudication"] == "NEEDS_REVIEW"


def test_confident_approval_probability_promotes_review(monkeypatch):
    import src.arbiter as arbiter

    monkeypatch.setattr(arbiter, "load_arbiter", lambda: {"margin": 2.0})
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.95, 0.01, 0.04])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state())
    assert pred["adjudication"] == "APPROVED"
    assert pred["confidence"] == 0.95


def test_uncertain_probability_keeps_review_and_reblends_confidence(monkeypatch):
    import src.arbiter as arbiter

    monkeypatch.setattr(arbiter, "load_arbiter", lambda: {"margin": 2.0})
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.5, 0.1, 0.4])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state())
    assert pred["adjudication"] == "NEEDS_REVIEW"
    assert pred["confidence"] == 0.35


def test_denials_are_never_relaxed(monkeypatch):
    import src.arbiter as arbiter

    monkeypatch.setattr(arbiter, "load_arbiter", lambda: {"margin": 0.0})
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.99, 0.005, 0.005])
    pred = {"adjudication": "DENIED", "confidence": 0.9}
    arbiter.apply_arbiter(pred, _state(final_decision="DENIED", hard="DENIED"))
    assert pred["adjudication"] == "DENIED"


def test_replace_mode_uses_exact_path_correctness_confidence(monkeypatch):
    import src.arbiter as arbiter

    blob = {"mode": "replace", "correctness_calibrator": _ConstantBinaryModel(0.73),
            "correctness_features": ["changed"]}
    monkeypatch.setattr(arbiter, "load_arbiter", lambda: blob)
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.95, 0.01, 0.04])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state())
    assert pred == {"adjudication": "APPROVED", "confidence": 0.73}


def test_runtime_pathway_keeps_decision_and_calibrates_exact_action(monkeypatch):
    import src.arbiter as arbiter

    blob = {"mode": "replace", "decision_pathway": "runtime",
            "correctness_calibrator": _ConstantBinaryModel(0.81),
            "correctness_features": ["changed"]}
    monkeypatch.setattr(arbiter, "load_arbiter", lambda: blob)
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.95, 0.01, 0.04])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    arbiter.apply_arbiter(pred, _state())
    assert pred == {"adjudication": "NEEDS_REVIEW", "confidence": 0.81}


def test_runtime_majority_pathway_uses_two_of_three_vote(monkeypatch):
    import src.arbiter as arbiter

    blob = {"mode": "replace", "decision_pathway": "runtime_majority",
            "correctness_calibrator": _ConstantBinaryModel(0.81),
            "correctness_features": ["changed"]}
    monkeypatch.setattr(arbiter, "load_arbiter", lambda: blob)
    monkeypatch.setattr(arbiter, "class_probabilities", lambda state: [0.5, 0.1, 0.4])
    pred = {"adjudication": "NEEDS_REVIEW", "confidence": 0.3}
    state = _state(rule_decision="APPROVED", meta_label="APPROVED")
    arbiter.apply_arbiter(pred, state)
    assert pred == {"adjudication": "APPROVED", "confidence": 0.81}


def test_utility_table_matches_public_scoring():
    assert UTILITY["APPROVED"]["DENIED"] == -4
    assert UTILITY["NEEDS_REVIEW"]["APPROVED"] == 2
    assert UTILITY["DENIED"]["NEEDS_REVIEW"] == 1
    assert LABELS == ("APPROVED", "DENIED", "NEEDS_REVIEW")
