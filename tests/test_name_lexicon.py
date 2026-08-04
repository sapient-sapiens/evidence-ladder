from pathlib import Path
from unittest import TestCase

from src.name_resolve import (
    load_name_lexicon,
    normalize_name_token,
    normalize_two_token_name,
    strip_name_chrome,
)


class FitNameLexiconTests(TestCase):
    def test_unified_fit_vocabulary_supports_both_name_positions(self) -> None:
        path = Path(__file__).resolve().parents[1] / "models" / "name_token_lexicon.json"
        lexicon = load_name_lexicon(str(path))
        tokens = lexicon["tokens"]
        self.assertEqual(len(tokens), 144)
        self.assertEqual(lexicon["source"], "dev800.txt")
        for token in tokens:
            first = normalize_name_token(
                token, position="first", lexicon=lexicon
            )
            last = normalize_name_token(
                token, position="last", lexicon=lexicon
            )
            self.assertEqual(first, (token, 0, True))
            self.assertEqual(last, (token, 0, True))

    def test_terminal_m_expands_only_to_exact_fit_token(self) -> None:
        lexicon = load_name_lexicon()

        normalized, cost, exact = normalize_two_token_name(
            "Orimora Qorzam", lexicon
        )

        self.assertEqual("Orimora Qorzarn", normalized)
        self.assertEqual(1, cost)
        self.assertFalse(exact)

    def test_trailing_equals_is_form_chrome(self) -> None:
        cleaned, stripped = strip_name_chrome("Ixotari Xandane =")

        self.assertEqual("Ixotari Xandane", cleaned)
        self.assertTrue(stripped)
