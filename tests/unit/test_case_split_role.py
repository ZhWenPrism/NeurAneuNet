import unittest

from neuraneunet.datasets import CaseRecord, SplitRole, VolumeSpec


class CaseSplitRoleTest(unittest.TestCase):
    def test_valid_serialized_roles_are_normalized(self):
        volume = VolumeSpec((10, 10, 10), (1.0, 1.0, 1.0), "RAS")
        for role in SplitRole:
            with self.subTest(role=role):
                self.assertIs(CaseRecord("case-1", role.value, volume).split, role)
        with self.assertRaisesRegex(ValueError, "split"):
            CaseRecord("case-1", "external_validation", volume)
