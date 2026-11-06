import unittest

from neuraneunet.models.geometry.centerline import cumulative_arc_length, interpolate_along_centerline, project_to_centerline

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy is optional")
class ArrayCenterlineTests(unittest.TestCase):
    def test_numpy_paths_match_tuple_paths_with_repeated_points(self):
        points = [(0., 0., 0.), (0., 0., 0.), (3., 0., 0.), (3., 4., 0.)]
        array = np.array(points)
        self.assertEqual(cumulative_arc_length(array), cumulative_arc_length(points))
        self.assertEqual(interpolate_along_centerline(array, 5.), (3., 2., 0.))
        self.assertEqual(project_to_centerline((4., 2., 0.), array),
                         project_to_centerline((4., 2., 0.), points))

    def test_zero_length_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-zero segment"):
            cumulative_arc_length(np.zeros((3, 3)))
