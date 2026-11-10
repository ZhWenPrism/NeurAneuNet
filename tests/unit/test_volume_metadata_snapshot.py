import unittest

from neuraneunet.datasets import VolumeSpec


class VolumeMetadataSnapshotTest(unittest.TestCase):
    def test_mutating_input_vectors_cannot_change_validated_metadata(self):
        shape = [10, 20, 30]
        spacing = [0.5, 1.0, 2.0]
        volume = VolumeSpec(shape, spacing, "LPS")
        shape[0] = 0
        spacing[2] = float("nan")
        self.assertEqual(volume.shape_zyx, (10, 20, 30))
        self.assertEqual(volume.spacing_mm_zyx, (0.5, 1.0, 2.0))
        self.assertEqual(volume.voxel_volume_mm3, 1.0)
