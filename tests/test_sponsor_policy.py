import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.sponsor_policy import learned_revoked_sponsors


class SponsorPolicyTests(TestCase):
    def setUp(self):
        learned_revoked_sponsors.cache_clear()

    def tearDown(self):
        learned_revoked_sponsors.cache_clear()

    def test_rejects_artifact_marked_as_containing_case_ids(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "policy.json"
            artifact.write_text(
                json.dumps(
                    {
                        "learned_revoked_sponsors": {"SPN-2718": {}},
                        "crossfit_selection_counts": {"SPN-2718": 6},
                        "anti_leakage": {"contains_case_ids": True},
                    }
                ),
                encoding="utf-8",
            )
            with patch("src.sponsor_policy._ARTIFACT", artifact):
                self.assertEqual(frozenset(), learned_revoked_sponsors())

    def test_loads_only_sponsor_shaped_field_values(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "policy.json"
            artifact.write_text(
                json.dumps(
                    {
                        "learned_revoked_sponsors": {
                            "SPN-2718": {},
                            "SPN-7331": {},
                            "MIB-000001": {},
                        },
                        "criteria": {"required_leave_one_fold_selections": 6},
                        "crossfit_selection_counts": {
                            "SPN-2718": 6,
                            "SPN-7331": 5,
                            "MIB-000001": 6,
                        },
                        "anti_leakage": {
                            "contains_case_ids": False,
                            "contains_filenames": False,
                            "contains_document_hashes": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("src.sponsor_policy._ARTIFACT", artifact):
                self.assertEqual(
                    frozenset({"SPN-2718"}), learned_revoked_sponsors()
                )

    def test_loads_causal_exception_with_four_fold_witnesses(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "policy.json"
            artifact.write_text(
                json.dumps(
                    {
                        "learned_revoked_sponsors": {
                            "SPN-9090": {
                                "selection_mode": "causal_exception",
                                "required_leave_one_fold_selections": 4,
                            }
                        },
                        "criteria": {"required_leave_one_fold_selections": 6},
                        "crossfit_selection_counts": {"SPN-9090": 4},
                        "anti_leakage": {
                            "contains_case_ids": False,
                            "contains_filenames": False,
                            "contains_document_hashes": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("src.sponsor_policy._ARTIFACT", artifact):
                self.assertEqual(
                    frozenset({"SPN-9090"}), learned_revoked_sponsors()
                )
