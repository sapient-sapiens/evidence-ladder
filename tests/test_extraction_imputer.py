import unittest

from src.extraction_imputer import (
    apply_missing_field_imputations,
    stabilize_provenance_features,
)


class _Vectorizer:
    def transform(self, rows):
        return rows


class _Estimator:
    def __init__(self, value):
        self.value = value

    def predict(self, _matrix):
        return [self.value]


def artifact_for(field, value):
    return {
        "models": {
            field: {"vectorizer": _Vectorizer(), "estimator": _Estimator(value)}
        }
    }


class ExtractionImputerTests(unittest.TestCase):
    def test_missing_closed_vocab_field_is_filled(self):
        prediction = {"visa_class": "unknown"}
        changed = apply_missing_field_imputations(
            prediction, {}, artifact=artifact_for("visa_class", "MED-3")
        )
        self.assertEqual(prediction["visa_class"], "MED-3")
        self.assertEqual(changed, {"visa_class"})

    def test_concrete_ocr_value_is_never_overridden(self):
        prediction = {"visa_class": "XW-1"}
        changed = apply_missing_field_imputations(
            prediction, {}, artifact=artifact_for("visa_class", "MED-3")
        )
        self.assertEqual(prediction["visa_class"], "XW-1")
        self.assertEqual(changed, set())

    def test_non_allowlisted_prediction_is_rejected(self):
        prediction = {"visa_class": "unknown"}
        changed = apply_missing_field_imputations(
            prediction, {}, artifact=artifact_for("visa_class", "SECRET-9")
        )
        self.assertEqual(prediction["visa_class"], "unknown")
        self.assertEqual(changed, set())

    def test_new_evidence_wrappers_do_not_create_provenance_drift(self):
        features = {
            "n_field_sources": 3,
            "n_fields_from_native": 0,
            "n_fields_from_standard_ocr": 0,
            "n_fields_from_specialized_ocr": 3,
        }
        stabilized = stabilize_provenance_features(
            features,
            {
                "applicant_name": "native:preserved_across_ocr",
                "arrival_date": "ocr_consensus:arrival_date",
                "fee_status": "fee_ledger:native:paid",
            },
        )

        self.assertEqual(stabilized["n_fields_from_native"], 2)
        self.assertEqual(stabilized["n_fields_from_standard_ocr"], 1)
        self.assertEqual(stabilized["n_fields_from_specialized_ocr"], 0)
        self.assertEqual(stabilized["field_ocr_arrival_date"], 1)
        self.assertEqual(stabilized["field_specialized_arrival_date"], 0)


if __name__ == "__main__":
    unittest.main()
