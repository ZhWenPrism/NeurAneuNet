import unittest
from unittest.mock import patch

from neuraneunet.models.geometry import centerline


class LandingZoneProjectionTest(unittest.TestCase):
    def test_two_landing_points_reuse_one_centerline_distance_table(self):
        path = [(0, 0, 0), (0, 0, 0), (4, 0, 0), (4, 3, 0)]
        with patch.object(centerline, "_arc_lengths", wraps=centerline._arc_lengths) as lengths:
            offset = centerline.landing_zone_offset((1, 0, 0), (4, 2, 0), path)
        self.assertEqual(offset, 5.0)
        self.assertEqual(lengths.call_count, 1)
