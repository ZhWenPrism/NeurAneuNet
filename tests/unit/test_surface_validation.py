import unittest

from neuraneunet.evaluation import percentile_hausdorff


class SurfaceValidationTests(unittest.TestCase):
    def test_both_surfaces_require_finite_three_dimensional_points(self):
        valid = [(0.0, 0.0, 0.0)]
        for point in ((0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                      (float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0)):
            for left, right in (([point], valid), (valid, [point])):
                with self.subTest(point=point, left=left), self.assertRaises(ValueError):
                    percentile_hausdorff(left, right)

    def test_singleton_surfaces_use_all_three_coordinates(self):
        self.assertEqual(
            percentile_hausdorff([(0.0, 0.0, 0.0)], [(3.0, 0.0, 4.0)]),
            5.0,
        )
