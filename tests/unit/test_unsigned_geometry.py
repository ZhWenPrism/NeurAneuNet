import unittest

from neuraneunet.evaluation.metrics import mean_signed_error, percentile_hausdorff
from neuraneunet.models.geometry.centerline import (
    cumulative_arc_length,
    interpolate_along_centerline,
    project_to_centerline,
)

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class UnsignedGeometryTest(unittest.TestCase):
    def test_unsigned_sizes_preserve_negative_error(self):
        with np.errstate(over="raise"):
            error = mean_signed_error(np.array([9, 10], dtype=np.uint8),
                                      np.array([7, 9], dtype=np.uint8))
        self.assertEqual(error, -1.5)

    def test_unsigned_surface_distance_is_symmetric(self):
        left = np.array([[2, 0, 0]], dtype=np.uint8)
        right = np.array([[7, 0, 0]], dtype=np.uint8)
        with np.errstate(over="raise"):
            self.assertEqual(percentile_hausdorff(left, right), 5.0)

    def test_descending_uint_centerline_uses_physical_distances(self):
        points = np.array([[10, 0, 0], [5, 0, 0], [0, 0, 0]], dtype=np.uint8)
        with np.errstate(over="raise"):
            self.assertEqual(cumulative_arc_length(points), (0.0, 5.0, 10.0))
            self.assertEqual(interpolate_along_centerline(points, 2.5), (7.5, 0.0, 0.0))
            result = project_to_centerline(np.array([7, 2, 0], dtype=np.uint8), points)
        self.assertEqual(result.point, (7.0, 0.0, 0.0))
        self.assertEqual(result.arc_length_mm, 3.0)
        self.assertEqual(result.distance_mm, 2.0)
