"""Проверки regex-паттернов."""

from __future__ import annotations

import unittest

from vishing.constants import CONFIGS_DIR
from vishing.text.patterns import extract_pattern_features, load_pattern_catalog


class PatternCatalogTestCase(unittest.TestCase):
    def test_sms_and_transfer_patterns(self) -> None:
        catalog = load_pattern_catalog(CONFIGS_DIR / "patterns.yaml")
        summary = extract_pattern_features(
            "Назовите код из смс и переведите деньги на безопасный счет прямо сейчас.",
            catalog,
        )
        self.assertGreaterEqual(int(summary.features["pattern_sms_code_count"]), 1)
        self.assertGreaterEqual(int(summary.features["pattern_transfer_money_count"]), 1)
        self.assertGreaterEqual(int(summary.features["pattern_safe_account_count"]), 1)


if __name__ == "__main__":
    unittest.main()
