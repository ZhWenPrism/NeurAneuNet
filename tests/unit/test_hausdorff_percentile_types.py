from decimal import Decimal
import unittest
from neuraneunet.evaluation.metrics import percentile_hausdorff


class HausdorffPercentileTypesTest(unittest.TestCase):
    def test_decimal_percentiles_match_float_interpolation(self):
        left, right = [(0, 0, 0), (2, 0, 0)], [(1, 0, 0), (5, 0, 0)]
        for value in ("0", "50", "95", "100"):
            with self.subTest(value=value):
                expected = percentile_hausdorff(left, right, float(value))
                self.assertEqual(percentile_hausdorff(left, right, Decimal(value)), expected)

    def test_invalid_decimal_percentiles_raise_value_error(self):
        for value in ("NaN", "Infinity", "-1", "101"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                percentile_hausdorff([(0, 0, 0)], [(1, 0, 0)], Decimal(value))
