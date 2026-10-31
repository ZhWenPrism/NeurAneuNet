import unittest

from neuraneunet.datasets import VolumeSpec


class VolumeDimensionTests(unittest.TestCase):
    def test_dimensions_must_be_positive_integers(self):
        for dimension in (1.5, 1.0, float("nan"), float("inf"), True, 0, -1):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                VolumeSpec((dimension, 2, 3), (0.5, 0.5, 0.5), "RAS")

    def test_valid_dimensions_preserve_physical_spacing(self):
        volume = VolumeSpec((1, 2, 3), (0.5, 0.5, 2.0), "RAS")
        self.assertEqual(volume.shape_zyx, (1, 2, 3))
        self.assertEqual(volume.voxel_volume_mm3, 0.5)
