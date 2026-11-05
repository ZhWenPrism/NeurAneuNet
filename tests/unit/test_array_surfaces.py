import unittest

from neuraneunet.evaluation.metrics import percentile_hausdorff

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy is optional")
class ArraySurfaceTests(unittest.TestCase):
    def test_array_points_match_tuples(self):
        left = [(0., 0., 0.), (4., 0., 0.)]
        right = [(0., 0., 1.), (4., 0., 3.)]
        self.assertEqual(percentile_hausdorff(np.array(left), np.array(right)),
                         percentile_hausdorff(left, right))

    def test_empty_array_reports_missing_points(self):
        with self.assertRaisesRegex(ValueError, "both surfaces must contain points"):
            percentile_hausdorff(np.empty((0, 3)), np.zeros((1, 3)))
