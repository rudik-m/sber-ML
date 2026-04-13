"""Проверки нормализации текста."""

from __future__ import annotations

import unittest

from vishing.text.normalize import maybe_fix_common_asr_artifacts, normalize_text


class NormalizeTextTestCase(unittest.TestCase):
    def test_normalize_text_compacts_spaces_and_sms(self) -> None:
        self.assertEqual(normalize_text("  Код   из С М С!!!\n"), "код из смс!")

    def test_fix_common_asr_artifacts_replaces_yo(self) -> None:
        self.assertIn("все", maybe_fix_common_asr_artifacts("всё"))


if __name__ == "__main__":
    unittest.main()
