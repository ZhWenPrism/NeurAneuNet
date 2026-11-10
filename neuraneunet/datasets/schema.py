"""Privacy-safe, framework-independent neurovascular data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from numbers import Integral
from types import MappingProxyType
from typing import Mapping


class SplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INDEPENDENT_CLINICAL = "independent_clinical"


def _positive_finite(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    """Spatial metadata without embedding the clinical image itself."""

    shape_zyx: tuple[int, int, int]
    spacing_mm_zyx: tuple[float, float, float]
    orientation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_zyx", tuple(self.shape_zyx))
        object.__setattr__(self, "spacing_mm_zyx", tuple(self.spacing_mm_zyx))
        if len(self.shape_zyx) != 3 or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in self.shape_zyx
        ):
            raise ValueError("shape_zyx must contain three positive integer dimensions")
        if len(self.spacing_mm_zyx) != 3:
            raise ValueError("spacing_mm_zyx must contain three values")
        for axis, spacing in zip("zyx", self.spacing_mm_zyx):
            _positive_finite(f"spacing_mm_{axis}", spacing)
        if not self.orientation.strip():
            raise ValueError("orientation must not be empty")

    @property
    def voxel_volume_mm3(self) -> float:
        z, y, x = self.spacing_mm_zyx
        return z * y * x


@dataclass(frozen=True, slots=True)
class DeviceTarget:
    device_family: str
    diameter_mm: float
    length_mm: float
    proximal_landing_mm: float
    distal_landing_mm: float

    def __post_init__(self) -> None:
        if not self.device_family.strip():
            raise ValueError("device_family must not be empty")
        for name in ("diameter_mm", "length_mm"):
            _positive_finite(name, getattr(self, name))
        for name in ("proximal_landing_mm", "distal_landing_mm"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """One patient-level experiment record identified only by a study key."""

    case_id: str
    split: SplitRole
    volume: VolumeSpec
    clinical_features: Mapping[str, float | None] = field(default_factory=dict)
    target: DeviceTarget | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or any(ch.isspace() for ch in self.case_id):
            raise ValueError("case_id must be a non-empty token")
        clean: dict[str, float | None] = {}
        for name, value in self.clinical_features.items():
            if not name.strip():
                raise ValueError("clinical feature names must not be empty")
            if value is not None and not isfinite(value):
                raise ValueError(f"clinical feature {name!r} must be finite or None")
            clean[name] = value
        object.__setattr__(self, "clinical_features", MappingProxyType(clean))

    @property
    def missing_clinical_features(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.clinical_features.items() if value is None)
