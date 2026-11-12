import unittest

from neuraneunet.evaluation.metrics import dice_score


class BinaryMaskContractTest(unittest.TestCase):
    def test_dice_rejects_nonbinary_masks(self):
        for label in (2, -1, 0.9, float("nan")):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "binary"):
                    dice_score([1, 0], [label, 0])

    def test_dice_preserves_binary_overlap_and_empty_foreground(self):
        self.assertEqual(dice_score([False, 0], [0, False]), 1.0)
        self.assertEqual(dice_score([True, 1, 0], [True, 0, 1]), 0.5)
