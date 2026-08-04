from unittest import TestCase

from src.meta_model import DEFAULT_RUNTIME_CONFIG


class MetaRuntimeConfigTests(TestCase):
    def test_fit_supported_dq_denial_threshold_is_frozen(self) -> None:
        self.assertEqual(0.4, DEFAULT_RUNTIME_CONFIG["dq_deny_threshold"])

    def test_fit_supported_dq_approval_ceiling_is_frozen(self) -> None:
        self.assertEqual(0.4, DEFAULT_RUNTIME_CONFIG["dq_approve_ceiling"])

    def test_fit_supported_meta_denial_threshold_is_frozen(self) -> None:
        self.assertEqual(0.5, DEFAULT_RUNTIME_CONFIG["deny_threshold"])
