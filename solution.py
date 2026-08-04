#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adjudicate import _rule_adjudicate, adjudicate
from src.arbiter import apply_arbiter_batch, packet_state
from src.best_ocr import best_ocr_cached, packet_needs_best_fallback
from src.constants import DISQUALIFYING_FLAGS
from src.noisy_channel_ocr import (
    apply_noisy_channel_repairs,
    packet_needs_noisy_channel,
)
from src.rapid_ocr import (
    merge_rapid_fills,
    merge_semantic_risk_flags,
    packet_needs_embedded_rapid_ocr,
    packet_needs_rapid_ocr,
    rapid_embedded_ocr_cached,
    rapid_ocr_cached,
    repair_ocr_name,
    rapid_semantics,
)
from src.note_finding_ocr import (
    decode_note_finding,
    note_finding_ocr_cached,
    packet_needs_note_finding_ocr,
)
from src.fee_absent import apply_absent_fee_imputation
from src.extraction_imputer import (
    apply_missing_field_imputations,
    stabilize_provenance_features,
)
from src.features import feature_dict
from src.meta_model import predict_adjudication, predict_has_dq
from src.legibility_b13 import (
    apply_legibility_result,
    legibility_cached,
    packet_needs_legibility,
)
from src.parse_fields import _date_in_plausible_range, parse_packet
from src.pdf_layer_forensics import raster_profile
from src.text_extract import TextSources, extract_text
from src.threshold_ocr import packet_needs_threshold_fallback, threshold_ocr_cached
from src.visual_damage import registry_portrait_obscured
from src.visual_risk_marks import (
    biohazard_red_seal,
    occluded_biohazard_red_seal,
    planetary_embargo_triangle,
)
from src.waiver_fee import apply_fee_ledger_consistency


def _reconcile_post_adjudication_risks(pred: dict, packet) -> None:
    """Apply hard policy to disqualifiers found by late visual extraction.

    A few deliberately gated visual detectors run after the main adjudication
    pass so their review-only companion flags cannot demote an already-correct
    denial.  Their *primary* risk atom is nevertheless policy evidence: once a
    detector has emitted a disqualifying atom, the decision must be recomputed
    from that final evidence instead of leaving an internally inconsistent
    ``risk_flags=biohazard_red, adjudication=NEEDS_REVIEW`` result.
    """
    atoms = {
        atom
        for atom in str(pred.get("risk_flags") or "none").split("|")
        if atom and atom != "none"
    }
    if pred.get("adjudication") == "DENIED" or not (atoms & DISQUALIFYING_FLAGS):
        return
    packet.risk_flags = sorted(atoms)
    policy = adjudicate(packet, pdf_path=None)
    pred["adjudication"] = policy["adjudication"]
    pred["confidence"] = policy["confidence"]


def prepare_packet(pdf: Path):
    """Run the production extraction pipeline and return adjudication inputs."""
    text = extract_text(pdf, allow_ocr=True)
    packet = parse_packet(pdf.stem, text)
    # R21: gated Sauvola fallback only when fusion still has unresolved fields.
    if packet_needs_threshold_fallback(packet, text):
        thr = threshold_ocr_cached(pdf)
        if thr.strip():
            text = TextSources(
                native=text.native,
                page_ocr=text.page_ocr,
                embedded_ocr=text.embedded_ocr,
                biometric_flags=text.biometric_flags,
                threshold_ocr=thr,
                best_ocr=text.best_ocr,
            )
            packet = parse_packet(pdf.stem, text)
    # R22: tessdata_best fallback only after current/R21 still unresolved.
    if packet_needs_best_fallback(packet, text):
        best = best_ocr_cached(pdf)
        if best.strip():
            text = TextSources(
                native=text.native,
                page_ocr=text.page_ocr,
                embedded_ocr=text.embedded_ocr,
                biometric_flags=text.biometric_flags,
                threshold_ocr=text.threshold_ocr,
                best_ocr=best,
            )
            packet = parse_packet(pdf.stem, text)
    # Preserve the strongest source across the *entire* fallback chain.  A
    # normal neural merge can replace a conflicted native value and thereby
    # make it look non-native to the later high-resolution merge.  Remembering
    # provenance here prevents that two-hop downgrade while still allowing an
    # explicit manual correction to win.
    trusted_native_sponsor = None
    if str(packet.field_sources.get("sponsor_id", "")).startswith("native"):
        value = str(packet.fields.get("sponsor_id") or "")
        if value and value != "SPN-0000":
            trusted_native_sponsor = value
    # Applicant identity is a cross-document relation, not a token that must
    # appear verbatim as a risk label. Apply that relation over the existing
    # OCR streams before deciding whether the expensive neural pass is needed.
    base_semantics = rapid_semantics(
        "\n\n".join(
            stream
            for stream in (
                text.native,
                text.page_ocr,
                text.embedded_ocr,
                text.threshold_ocr,
                text.best_ocr,
                text.oriented_ocr,
            )
            if stream
        )
    )
    sponsor_resolution_consensus = None
    normal_neural_date = None
    normal_neural_name = None
    normal_neural_sponsor = None
    normal_neural_home = None
    normal_neural_semantics: dict[str, object] = {}
    explicit_risk_votes: list[tuple[str, ...]] = []
    printed_fee_unknown_votes = int(bool(base_semantics.get("_printed_fee_unknown")))
    base_exact_flags = base_semantics.get("_observed_flags_exact")
    if isinstance(base_exact_flags, list) and base_exact_flags:
        explicit_risk_votes.append(tuple(sorted(str(x) for x in base_exact_flags)))
    semantic_name = base_semantics.get("applicant_name")
    semantic_name_role = base_semantics.get("_applicant_name_role")
    current_name = packet.fields.get("applicant_name")
    current_name_source = str(packet.field_sources.get("applicant_name", ""))
    base_date_fill = base_semantics.get("arrival_date_fill")
    if (
        isinstance(base_date_fill, str)
        and _date_in_plausible_range(base_date_fill)
        and str(packet.fields.get("arrival_date") or "").casefold()
        in {"", "unknown", "1900-01-01"}
    ):
        packet.fields["arrival_date"] = base_date_fill
        packet.field_sources["arrival_date"] = "document_role_semantic:date_fill"
        packet.conflicts.discard("arrival_date_unreadable")
    if bool(base_semantics.get("_observed_none")):
        packet.risk_flags = []
        packet.observed_flags_seen = True
        packet.field_sources["risk_flags"] = "document_role_semantic:observed_none"
        packet.field_sources["_observed_none_any"] = "document_role_semantic"
    if bool(base_semantics.get("_damage_implies_illegible")):
        packet.field_sources["_damage_implies_illegible"] = (
            "document_role_semantic:missing_b13_damage"
        )
    if bool(base_semantics.get("_complete_no_b13_layout")):
        page = int(base_semantics.get("_registry_page") or 0)
        packet.field_sources["_complete_no_b13_layout"] = f"document_role_semantic:{page}"
    if bool(base_semantics.get("_missing_biometric_roles_layout")):
        packet.field_sources["_missing_biometric_roles_layout"] = (
            "document_role_semantic"
        )
    if bool(base_semantics.get("_review_flag_asserted")):
        packet.field_sources["_review_flag_asserted"] = "document_role_semantic"
    if "identity_conflict" in set(
        base_semantics.get("_review_risk_atoms_exact", ()) or ()
    ):
        packet.field_sources["_manual_identity_conflict"] = (
            "document_role_semantic"
        )
    if bool(base_semantics.get("_redacted_identity_damage")):
        packet.field_sources["_redacted_identity_damage"] = (
            "document_role_semantic"
        )
    intake_name = base_semantics.get("_intake_name")
    sponsor_name = base_semantics.get("_sponsor_name")
    if (
        "illegible_biometrics" in set(packet.risk_flags)
        and isinstance(intake_name, str)
        and isinstance(sponsor_name, str)
        and intake_name != sponsor_name
    ):
        packet.risk_flags = sorted(set(packet.risk_flags) | {"sponsor_mismatch"})
        # A sponsor disagreement proves sponsor_mismatch, but the sponsor is a
        # lower-precedence identity source than a readable native intake value.
        # Keep the relation without replacing that stronger visible field.
        if not current_name_source.startswith("native"):
            packet.fields["applicant_name"] = intake_name
            packet.field_sources["applicant_name"] = "document_role_semantic:intake"
    if (
        semantic_name_role == "registry"
        and isinstance(semantic_name, str)
        and semantic_name != current_name
        and current_name_source.startswith("native")
    ):
        role_flags = list(base_semantics.get("risk_flags", ()) or ())
        if "identity_conflict" not in role_flags:
            role_flags.append("identity_conflict")
        base_semantics["risk_flags"] = role_flags
    merge_semantic_risk_flags(packet, base_semantics)
    if (
        "identity_conflict" in set(packet.risk_flags)
        and semantic_name_role == "registry"
        and isinstance(semantic_name, str)
        and "manual" not in current_name_source
    ):
        packet.fields["applicant_name"] = semantic_name
        packet.field_sources["applicant_name"] = "document_role_semantic:registry"
    base_sponsor = base_semantics.get("sponsor_id")
    current_sponsor = packet.fields.get("sponsor_id")
    current_sponsor_source = str(packet.field_sources.get("sponsor_id", ""))
    needs_semantic_confirmation = (
        isinstance(base_sponsor, str)
        and base_sponsor != current_sponsor
        and not current_sponsor_source.startswith("native")
        and "manual" not in current_sponsor_source
    )
    # A different OCR engine for the hardest remainder. PP-OCRv6 is bundled in
    # the wheel and runs offline; its stream stays lowest-trust/fill-missing.
    if packet_needs_rapid_ocr(packet) or needs_semantic_confirmation:
        ppocr = rapid_ocr_cached(pdf)
        if ppocr.strip():
            # Parse PP-OCR independently as well as through normal fusion.  The
            # independent packet preserves evidence that a higher-trust but
            # wrong Tesseract hypothesis would otherwise mask.
            neural_candidate = parse_packet(pdf.stem, ppocr)
            ppocr_text = TextSources(
                native=text.native,
                page_ocr=text.page_ocr,
                embedded_ocr=text.embedded_ocr,
                biometric_flags=text.biometric_flags,
                threshold_ocr=text.threshold_ocr,
                best_ocr=text.best_ocr,
                ppocr_ocr=ppocr,
                oriented_ocr=text.oriented_ocr,
            )
            candidate = parse_packet(pdf.stem, ppocr_text)
            neural_semantics = rapid_semantics(ppocr)
            normal_neural_semantics = neural_semantics
            printed_fee_unknown_votes += int(
                bool(neural_semantics.get("_printed_fee_unknown"))
            )
            neural_date = neural_semantics.get("arrival_date") or neural_semantics.get(
                "arrival_date_fill"
            )
            if isinstance(neural_date, str) and _date_in_plausible_range(neural_date):
                normal_neural_date = neural_date
            neural_name = neural_semantics.get("applicant_name")
            if isinstance(neural_name, str):
                normal_neural_name = neural_name
            neural_sponsor_for_resolution = neural_semantics.get("sponsor_id")
            if isinstance(neural_sponsor_for_resolution, str):
                normal_neural_sponsor = neural_sponsor_for_resolution
            neural_home = neural_semantics.get("home_world")
            if isinstance(neural_home, str):
                normal_neural_home = neural_home
            neural_exact_flags = neural_semantics.get("_observed_flags_exact")
            if isinstance(neural_exact_flags, list) and neural_exact_flags:
                explicit_risk_votes.append(
                    tuple(sorted(str(x) for x in neural_exact_flags))
                )
            neural_fuzzy_flags = neural_semantics.get("_observed_flags_fuzzy")
            if isinstance(neural_fuzzy_flags, list) and neural_fuzzy_flags:
                explicit_risk_votes.append(
                    tuple(sorted(str(x) for x in neural_fuzzy_flags))
                )
            if bool(neural_semantics.get("_observed_none")):
                packet.field_sources["_observed_none_any"] = "ppocr_semantic"
            if bool(neural_semantics.get("_damage_implies_illegible")):
                packet.field_sources["_damage_implies_illegible"] = (
                    "ppocr_semantic:missing_b13_damage"
                )
            if bool(neural_semantics.get("_complete_no_b13_layout")):
                page = int(neural_semantics.get("_registry_page") or 0)
                packet.field_sources["_complete_no_b13_layout"] = (
                    f"ppocr_semantic:{page}"
                )
            if bool(neural_semantics.get("_missing_biometric_roles_layout")):
                packet.field_sources["_missing_biometric_roles_layout"] = (
                    "ppocr_semantic"
                )
            if bool(neural_semantics.get("_review_flag_asserted")):
                packet.field_sources["_review_flag_asserted"] = "ppocr_semantic"
            if bool(neural_semantics.get("_redacted_identity_damage")):
                packet.field_sources["_redacted_identity_damage"] = (
                    "ppocr_semantic"
                )
            merge_rapid_fills(
                packet,
                candidate,
                neural_candidate,
                semantics=neural_semantics,
            )
            neural_registry_name = neural_semantics.get("applicant_name")
            neural_role_flags = set(neural_semantics.get("risk_flags", ()) or ())
            if (
                neural_semantics.get("_applicant_name_role") == "registry"
                and isinstance(neural_registry_name, str)
                and "identity_conflict" in neural_role_flags
                and "identity_conflict" in set(packet.risk_flags)
                and "manual" not in str(
                    packet.field_sources.get("applicant_name", "")
                )
            ):
                # In a true cross-role identity conflict the registry identity
                # is the extraction target.  This does not apply to the
                # near-glyph illegibility path, where intake/sponsor consensus
                # correctly vetoes a degraded registry spelling.
                packet.fields["applicant_name"] = neural_registry_name
                packet.field_sources["applicant_name"] = (
                    "ppocr_semantic:registry_identity_conflict"
                )
            # Two OCR engines agreeing on a format-constrained sponsor ID
            # outrank a different untrusted single-engine glyph reading.
            base_sponsor = base_semantics.get("sponsor_id")
            neural_sponsor = neural_semantics.get("sponsor_id")
            base_sponsor_candidates = set(
                base_semantics.get("_sponsor_candidates", ()) or ()
            )
            current_sponsor = packet.fields.get("sponsor_id")
            current_source = str(packet.field_sources.get("sponsor_id", ""))
            if (
                isinstance(base_sponsor, str)
                and base_sponsor == neural_sponsor
                and base_sponsor != current_sponsor
                and not current_source.startswith("native")
                and "manual" not in current_source
            ):
                packet.fields["sponsor_id"] = base_sponsor
                packet.field_sources["sponsor_id"] = "ocr_consensus:sponsor_id"
                sponsor_resolution_consensus = base_sponsor
            elif (
                isinstance(neural_sponsor, str)
                and neural_sponsor in base_sponsor_candidates
            ):
                # The base stream can contain one conflicting glyph reading
                # and therefore decline to choose, while still independently
                # containing the normal neural pass's exact candidate.  That
                # cross-engine agreement outranks a later one-off high-DPI
                # digit substitution.
                packet.fields["sponsor_id"] = neural_sponsor
                packet.field_sources["sponsor_id"] = "ocr_consensus:sponsor_id"
                sponsor_resolution_consensus = neural_sponsor
    # Original embedded images sometimes preserve glyphs that PDF rendering
    # blurs or recompresses.  Restrict this pass to the severe remainder.
    if packet_needs_embedded_rapid_ocr(packet):
        embedded_ppocr = rapid_embedded_ocr_cached(pdf)
        if embedded_ppocr.strip():
            embedded_candidate = parse_packet(pdf.stem, embedded_ppocr)
            embedded_finding, _has_label, _exact = decode_note_finding(
                embedded_ppocr, packet.case_id
            )
            if (
                packet.adjudicator_finding is None
                and embedded_finding in {"APPROVED", "DENIED", "NEEDS_REVIEW"}
            ):
                packet.adjudicator_finding = embedded_finding
                packet.sources_seen.add("adjudicator")
                packet.field_sources["_adjudicator_finding"] = (
                    "ppocr_embedded:trusted_note"
                )
            embedded_semantics = rapid_semantics(embedded_ppocr)
            if bool(embedded_semantics.get("_observed_none")):
                packet.field_sources["_observed_none_any"] = (
                    "ppocr_embedded_semantic"
                )
            merge_rapid_fills(
                packet,
                embedded_candidate,
                embedded_candidate,
                semantics=embedded_semantics,
            )
    # R35: cross-fitted noisy-channel closed-vocab repair (garble/missing fill;
    # real-word override only with cross-source agreement).
    if packet_needs_noisy_channel(packet):
        apply_noisy_channel_repairs(packet, text)
    # R42: impute fee when the packet has no fee evidence at all (not obscured).
    apply_absent_fee_imputation(packet, text)
    # A 288-DPI PP-OCR escalation used to run here.  Measured on FIT600 it
    # was worth 0.08 total points while consuming about a fifth of the whole
    # runtime budget, so the packet-level escalation chain now stops at the
    # ordinary neural pass.
    if sponsor_resolution_consensus:
        packet.fields["sponsor_id"] = sponsor_resolution_consensus
        packet.field_sources["sponsor_id"] = "ocr_consensus:sponsor_id"
    if (
        trusted_native_sponsor
        and "manual" not in str(packet.field_sources.get("sponsor_id", ""))
    ):
        packet.fields["sponsor_id"] = trusted_native_sponsor
        packet.field_sources["sponsor_id"] = "native:preserved_across_ocr"
    # Preserve an explicitly readable B-13 ``none`` assertion through legacy
    # repair stages that may otherwise re-infer a flag from registry prose.
    if bool(base_semantics.get("_observed_none")) or packet.field_sources.get(
        "_observed_none_any"
    ):
        packet.risk_flags = []
        packet.observed_flags_seen = True
        packet.field_sources["risk_flags"] = "document_role_semantic:observed_none"
    if explicit_risk_votes:
        agreed = max(set(explicit_risk_votes), key=explicit_risk_votes.count)
        if explicit_risk_votes.count(agreed) >= 2:
            packet.risk_flags = list(agreed)
            packet.observed_flags_seen = True
            packet.field_sources["risk_flags"] = "ocr_consensus:observed_flags"
    if (
        packet.field_sources.get("_redacted_identity_damage")
        and "identity_conflict" in set(packet.risk_flags)
    ):
        if bool(base_semantics.get("_identity_cross_role_exact")):
            # A readable registry identity independently repeated by the
            # sponsor, and distinct from intake, remains a proven conflict
            # even when a REDACTED overlay is also present on the registry.
            pass
        elif packet.field_sources.get("_review_flag_asserted"):
            packet.risk_flags = sorted(
                set(packet.risk_flags) | {"illegible_biometrics"}
            )
        else:
            # With no explicit review-note assertion, a redacted role that
            # disagrees with the intact intake is damaged identity evidence,
            # not an independently readable second identity.
            packet.risk_flags = sorted(
                (set(packet.risk_flags) - {"identity_conflict"})
                | {"illegible_biometrics"}
            )
    # R46: readable_none / allowlisted flags before adjudicate (obs_seen unlock).
    # Illegible atom is merged into the submission AFTER adjudicate so review-only
    # extraction credit cannot demote a correct DENIED → NEEDS_REVIEW.
    leg_result = None
    if packet_needs_legibility(packet):
        leg_result = legibility_cached(pdf)
        apply_legibility_result(packet, leg_result)
    # R29: gated note-crop Finding OCR when header exists but R28 finding missing.
    if packet_needs_note_finding_ocr(packet, text):
        note_finding = note_finding_ocr_cached(pdf, packet.case_id, sources=text)
        if note_finding:
            packet.adjudicator_finding = note_finding
            packet.sources_seen.add("adjudicator")
    # Re-run the receipt-local ledger after every OCR fallback.  A late neural
    # stream may reintroduce the stale printed status that the native
    # amount/waiver-code pair already resolved.
    apply_fee_ledger_consistency(packet, text)
    if printed_fee_unknown_votes:
        packet.field_sources["_printed_fee_unknown_votes"] = str(
            printed_fee_unknown_votes
        )
    return packet, leg_result, text


def prediction_for(pdf: Path, state_out: dict | None = None) -> dict:
    packet, leg_result, _streams = prepare_packet(pdf)
    rule_fields, _rule_reasons, hard_decision = _rule_adjudicate(packet)
    rule_decision = hard_decision or str(rule_fields["adjudication"])
    extraction_features = feature_dict(packet, pdf, rule_decision)
    collected: dict = {}
    pred = adjudicate(packet, pdf_path=pdf, collect=collected)
    # The generic disqualifying-risk fallback historically emits biohazard when
    # it can read only a DENIED relation.  A large purple triangular seal is
    # independent visual evidence for planetary embargo and can safely subtype
    # that unresolved fallback. Explicit Observed-flags OCR remains stronger.
    if (
        pred.get("risk_flags") == "biohazard_red"
        and not packet.observed_flags_seen
        and planetary_embargo_triangle(pdf)
    ):
        pred["risk_flags"] = "planetary_embargo"
    # A trusted APPROVED finding is a strong document-role assertion of a
    # clean/exception-qualified packet. FIT600 has 40/40 such findings with no
    # risk atoms; do not let a visual legibility heuristic fabricate one.
    if packet.adjudicator_finding == "APPROVED":
        pred["risk_flags"] = "none"
    elif (
        str(pred.get("risk_flags") or "none") == "none"
        and not packet.observed_flags_seen
        and biohazard_red_seal(pdf)
    ):
        pred["risk_flags"] = "biohazard_red|illegible_biometrics"
    elif (
        str(pred.get("risk_flags") or "none") == "none"
        and not packet.observed_flags_seen
        and occluded_biohazard_red_seal(pdf)
    ):
        pred["risk_flags"] = "biohazard_red"
        packet.field_sources["_occluded_biohazard_visual"] = "topology"
    elif leg_result is not None and leg_result.status == "illegible":
        cur = [
            a
            for a in str(pred.get("risk_flags") or "none").split("|")
            if a and a != "none"
        ]
        if "illegible_biometrics" not in cur:
            cur.append("illegible_biometrics")
            pred["risk_flags"] = "|".join(sorted(cur))
    elif packet.field_sources.get("_damage_implies_illegible"):
        cur = [
            atom
            for atom in str(pred.get("risk_flags") or "none").split("|")
            if atom and atom != "none"
        ]
        if "illegible_biometrics" not in cur:
            cur.append("illegible_biometrics")
            pred["risk_flags"] = "|".join(sorted(cur))
    elif (
        str(pred.get("risk_flags") or "none") == "none"
        and packet.field_sources.get("_complete_no_b13_layout")
        and registry_portrait_obscured(
            pdf,
            int(packet.field_sources["_complete_no_b13_layout"].rsplit(":", 1)[-1]),
        )
    ):
        pred["risk_flags"] = "illegible_biometrics"
    elif packet.field_sources.get("_review_flag_asserted"):
        cur = {
            atom
            for atom in str(pred.get("risk_flags") or "none").split("|")
            if atom and atom != "none"
        }
        review_only = {
            "identity_conflict",
            "illegible_biometrics",
            "rescinded_denial",
            "sponsor_mismatch",
        }
        if not (cur & review_only):
            cur.add("illegible_biometrics")
            pred["risk_flags"] = "|".join(sorted(cur))
    # PDF-layer forensics is gated to unresolved compound-risk packets. Full
    # page rasters prove reconstruction/damage; small embedded assets preserve
    # the surviving intake portrait. This avoids broad content-stream scans on
    # ordinary packets and never invents the primary disqualifying atom.
    risk_atoms = {
        atom
        for atom in str(pred.get("risk_flags") or "none").split("|")
        if atom and atom != "none"
    }
    if (
        "illegible_biometrics" not in risk_atoms
        and not packet.observed_flags_seen
        and not packet.field_sources.get("_occluded_biohazard_visual")
        and risk_atoms & {"planetary_embargo", "biohazard_red"}
    ):
        profile = raster_profile(pdf)
        proves_illegible = False
        if (
            risk_atoms == {"planetary_embargo"}
            and packet.field_sources.get("_complete_no_b13_layout")
            and profile.full_page_images >= 1
        ):
            proves_illegible = True
        elif risk_atoms == {"biohazard_red"} and (
            (
                packet.field_sources.get("_missing_biometric_roles_layout")
                and profile.full_page_images >= 1
                and profile.small_assets >= 1
            )
            or profile.max_dark_fraction > 0.20
        ):
            proves_illegible = True
        if proves_illegible:
            risk_atoms.add("illegible_biometrics")
            pred["risk_flags"] = "|".join(sorted(risk_atoms))
    _reconcile_post_adjudication_risks(pred, packet)
    # Last-mile extraction only: sentinel closed-vocabulary fields can be
    # inferred from broad packet evidence. Running after adjudication makes the
    # decision/confidence contract immutable.
    fee_source = str(packet.field_sources.get("fee_status", ""))
    unresolved_packet_fee = str(packet.fields.get("fee_status") or "").casefold() in {
        "",
        "unknown",
    } and not fee_source
    preserve_explicit_unknown_fee = (
        fee_source.startswith(
            (
                "fee_ledger:",
                "manual_note:",
                "ppocr_semantic:fee_status_fill",
                "ppocr_semantic:manual_fee_unknown",
            )
        )
        and str(packet.fields.get("fee_status")) == "unknown"
    )
    imputer_features = stabilize_provenance_features(
        extraction_features, packet.field_sources
    )
    apply_missing_field_imputations(pred, imputer_features)
    if preserve_explicit_unknown_fee:
        pred["fee_status"] = "unknown"
    elif (
        unresolved_packet_fee
        and int(packet.field_sources.get("_printed_fee_unknown_votes", "0")) >= 2
        and pred.get("fee_status") == "paid"
    ):
        # Two independent OCR engines read the printed receipt value. Only
        # replace the paid missing-field prior; extracted ledger/manual values
        # and a different imputer conclusion remain untouched.
        pred["fee_status"] = "unknown"
    repaired_name = repair_ocr_name(str(pred.get("applicant_name") or ""))
    if repaired_name:
        pred["applicant_name"] = repaired_name
    pred["confidence"] = float(pred["confidence"])
    # Evidence snapshot for the final arbitration pass.  It records the values
    # actually submitted and the provenance that produced them, never the PDF
    # envelope or the case identifier.  Arbitration itself runs once for the
    # whole batch in main(), because tree-ensemble inference has a large fixed
    # per-call cost that must not be paid once per PDF.
    state = packet_state(
        packet,
        pred,
        rule_decision=rule_decision,
        hard_decision=hard_decision,
        reasons=collected.get("reasons", []),
        raw_confidence=collected.get("raw_confidence", pred["confidence"]),
        missing=collected.get("missing", []),
        meta=predict_adjudication(extraction_features),
        dq_prob=predict_has_dq(extraction_features),
        meta_features=extraction_features,
    )
    if state_out is not None:
        state_out.update(state)
    return pred


def _worker(pdf_str: str) -> tuple[dict, dict]:
    state: dict = {}
    pred = prediction_for(Path(pdf_str), state_out=state)
    return pred, state


def main(input_dir: str, output_path: str) -> None:
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Docker scoring uses 4 vCPU; default stays 4. Local M2 Pro can raise via
    # MIB_WORKERS (e.g. 10) without changing the shipped Docker contract.
    workers = max(1, int(os.environ.get("MIB_WORKERS", "4")))
    workers = min(workers, max(1, (os.cpu_count() or 2)))
    # Sequential is fine for small batches; parallel helps OCR-heavy runs.
    use_parallel = len(pdfs) >= 8 and workers > 1

    results: list[tuple[dict, dict]] = []
    if use_parallel:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, str(pdf)): pdf for pdf in pdfs}
            by_name = {}
            for fut in as_completed(futures):
                pdf = futures[fut]
                by_name[pdf.name] = fut.result()
        results = [by_name[pdf.name] for pdf in pdfs]
    else:
        results = [_worker(str(pdf)) for pdf in pdfs]

    preds = [pred for pred, _state in results]
    apply_arbiter_batch(preds, [state for _pred, state in results])

    with output.open("w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: solution.py <input_pdf_dir> <output_path>")
    main(sys.argv[1], sys.argv[2])
