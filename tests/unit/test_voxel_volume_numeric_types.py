from decimal import Decimal
import unittest
from neuraneunet.datasets.schema import VolumeSpec

try:
    import numpy as np
except ImportError:
    np = None


class VoxelVolumeNumericTypesTest(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is not installed")
    def test_unsigned_spacing_uses_physical_volume_without_overflow(self):
        volume = VolumeSpec((2, 2, 2), tuple(np.array([10, 10, 10], dtype=np.uint8)), "LPS")
        self.assertEqual(volume.voxel_volume_mm3, 1000.0)

    def test_decimal_spacing_produces_a_usable_float_volume(self):
        volume = VolumeSpec((2, 2, 2), (Decimal(".5"), Decimal(".5"), Decimal("1")), "LPS")
        self.assertEqual(volume.voxel_volume_mm3 * .5, .125)

    def test_standard_float_spacing_keeps_the_existing_result(self):
        self.assertEqual(VolumeSpec((2, 2, 2), (.5, .5, 1.), "LPS").voxel_volume_mm3, .25)
