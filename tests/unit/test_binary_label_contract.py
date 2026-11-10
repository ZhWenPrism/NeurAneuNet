import unittest

from neuraneunet.evaluation.metrics import binary_metrics


class BinaryLabelContractTest(unittest.TestCase):
    def test_invalid_labels_are_not_coerced_to_positive(self):
        for invalid in (float("nan"), 2, "0", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "binary"):
                    binary_metrics([0, invalid], [False, True])
                with self.assertRaisesRegex(ValueError, "binary"):
                    binary_metrics([False, True], [0, invalid])

    def test_boolean_and_numeric_binary_labels_agree(self):
        result = binary_metrics([True, False, True, False], [1, 1, 0, 0])
        self.assertEqual((result.sensitivity, result.specificity, result.accuracy), (0.5, 0.5, 0.5))
