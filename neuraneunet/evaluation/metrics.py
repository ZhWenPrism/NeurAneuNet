"""Dependency-light metrics for neurovascular perception and planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, inf, isfinite, sqrt
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


def _binary_counts(reference, prediction) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for actual, predicted in zip(reference, prediction):
        if actual:
            if predicted:
                tp += 1
            else:
                fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def dice_score(reference: Iterable[bool | int], prediction: Iterable[bool | int]) -> float:
    """Dice overlap for binary masks."""
    left, right = _paired_binary(reference, prediction, "masks")
    tp, _, fp, fn = _binary_counts(left, right)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    precision: float
    accuracy: float


def binary_metrics(reference: Iterable[bool | int], prediction: Iterable[bool | int]) -> BinaryMetrics:
    left, right = _paired_binary(reference, prediction, "labels")
    tp, tn, fp, fn = _binary_counts(left, right)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return BinaryMetrics(sensitivity, specificity, precision, (tp + tn) / len(left))


Point3D = tuple[float, float, float]


def _nearest_distances(surface_a: Sequence[Point3D], surface_b: Sequence[Point3D]) -> list[float]:
    nearest_a = []
    nearest_b = [inf] * len(surface_b)
    for point in surface_a:
        nearest = inf
        for index, candidate in enumerate(surface_b):
            separation = dist(point, candidate)
            if separation < nearest:
                nearest = separation
            if separation < nearest_b[index]:
                nearest_b[index] = separation
        nearest_a.append(nearest)
    return nearest_a + nearest_b


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
    values = sorted(_nearest_distances(surface_a, surface_b))
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
