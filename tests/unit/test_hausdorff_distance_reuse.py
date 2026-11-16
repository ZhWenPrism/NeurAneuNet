import math
import unittest
from unittest.mock import patch

from neuraneunet.evaluation import metrics


class HausdorffDistanceReuseTest(unittest.TestCase):
    def test_asymmetric_surfaces_preserve_percentile_with_one_pass_over_pairs(self):
        left = [(0., 0., 0.), (1., 0., 0.)]
        right = [(0., 1., 0.), (1., 2., 0.), (0., 4., 0.)]
        with patch.object(metrics, "dist", wraps=math.dist) as distance:
            forward = metrics.percentile_hausdorff(left, right, percentile=75)
        self.assertEqual(forward, 2.0)
        self.assertEqual(distance.call_count, len(left) * len(right))
        self.assertEqual(metrics.percentile_hausdorff(right, left, percentile=75), forward)
