"""Проверки извлечения group_id."""

from __future__ import annotations

import unittest

from vishing.utils.misc import extract_group_id


class GroupIdTestCase(unittest.TestCase):
    def test_extract_group_id_examples(self) -> None:
        self.assertEqual(extract_group_id("out_a_13.wav"), "out_a")
        self.assertEqual(extract_group_id("out_d_8.wav"), "out_d")
        self.assertEqual(extract_group_id("Nout_b_33.wav"), "nout_b")
        self.assertEqual(extract_group_id("Nout_f_3.wav"), "nout_f")
        self.assertEqual(extract_group_id("mvd.wav"), "mvd")


if __name__ == "__main__":
    unittest.main()
