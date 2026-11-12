"""Dependency-light metrics for neurovascular perception and planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, isfinite, sqrt
from typing import Iterable, Sequence


def _paired_finite(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left, right = tuple(y_true), tuple(y_pred)
    if not left or len(left) != len(right):
        raise ValueError("inputs must be non-empty and have equal length")
    if any(not isfinite(value) for value in (*left, *right)):
        raise ValueError("inputs must contain finite values")
    return tuple(map(float, left)), tuple(map(float, right))


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    left, right = _paired_finite(y_true, y_pred)
    return sum(abs(actual - predicted) for actual, predicted in zip(left, right)) / len(left)


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    left, right = _paired_finite(y_true, y_pred)
    return sqrt(sum((actual - predicted) ** 2 for actual, predicted in zip(left, right)) / len(left))


def mean_signed_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean predicted-minus-observed error; positive values indicate oversizing."""

    left, right = _paired_finite(y_true, y_pred)
    return sum(predicted - actual for actual, predicted in zip(left, right)) / len(left)


def _paired_binary(
    reference: Iterable[bool | int], prediction: Iterable[bool | int], name: str
) -> tuple[tuple[bool | int, ...], tuple[bool | int, ...]]:
    left, right = tuple(reference), tuple(prediction)
    if not left or len(left) != len(right):
        raise ValueError(f"{name} must be non-empty and have equal length")
    if any(value not in (0, 1) for values in (left, right) for value in values):
        raise ValueError(f"{name} must contain binary values (0 or 1)")
    return left, right


def dice_score(reference: Iterable[bool | int], prediction: Iterable[bool | int]) -> float:
    """Dice overlap for binary masks."""
    reference_values, prediction_values = _paired_binary(reference, prediction, "masks")
    reference_set = {index for index, value in enumerate(reference_values) if bool(value)}
    prediction_set = {index for index, value in enumerate(prediction_values) if bool(value)}
    if not reference_set and not prediction_set:
        return 1.0
    return 2 * len(reference_set & prediction_set) / (len(reference_set) + len(prediction_set))


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    precision: float
    accuracy: float


def binary_metrics(reference: Iterable[bool | int], prediction: Iterable[bool | int]) -> BinaryMetrics:
    left, right = _paired_binary(reference, prediction, "labels")
    pairs = [(bool(actual), bool(predicted)) for actual, predicted in zip(left, right)]
    tp = sum(actual and predicted for actual, predicted in pairs)
    tn = sum(not actual and not predicted for actual, predicted in pairs)
    fp = sum(not actual and predicted for actual, predicted in pairs)
    fn = sum(actual and not predicted for actual, predicted in pairs)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return BinaryMetrics(sensitivity, specificity, precision, (tp + tn) / len(pairs))


Point3D = tuple[float, float, float]


def _directed_distances(source: Sequence[Point3D], target: Sequence[Point3D]) -> list[float]:
    return [min(dist(point, candidate) for candidate in target) for point in source]


def percentile_hausdorff(
    surface_a: Sequence[Point3D],
    surface_b: Sequence[Point3D],
    percentile: float = 95.0,
) -> float:
    """Symmetric percentile Hausdorff distance for physical-space surface points."""

    if len(surface_a) == 0 or len(surface_b) == 0:
        raise ValueError("both surfaces must contain points")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    for surface in (surface_a, surface_b):
        for point in surface:
            if len(point) != 3 or any(not isfinite(value) for value in point):
                raise ValueError("surface points must contain three finite coordinates")
    values = sorted(
        _directed_distances(surface_a, surface_b) + _directed_distances(surface_b, surface_a)
    )
    rank = (len(values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    fraction = rank - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def agreement_rate(reference: Iterable[object], prediction: Iterable[object]) -> float:
    left, right = tuple(reference), tuple(prediction)
    if not left or len(left) != len(right):
        raise ValueError("ratings must be non-empty and have equal length")
    return sum(actual == predicted for actual, predicted in zip(left, right)) / len(left)
