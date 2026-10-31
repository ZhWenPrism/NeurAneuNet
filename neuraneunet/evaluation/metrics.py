"""Dependency-light metrics for neurovascular perception and planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence


def _paired_finite(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left, right = tuple(y_true), tuple(y_pred)
    if not left or len(left) != len(right):
        raise ValueError("inputs must be non-empty and have equal length")
    if any(not isfinite(value) for value in (*left, *right)):
        raise ValueError("inputs must contain finite values")
    return left, right


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


def dice_score(reference: Iterable[bool | int], prediction: Iterable[bool | int]) -> float:
    reference_values, prediction_values = tuple(reference), tuple(prediction)
    if not reference_values or len(reference_values) != len(prediction_values):
        raise ValueError("masks must be non-empty and have equal length")
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
    left, right = tuple(reference), tuple(prediction)
    if not left or len(left) != len(right):
        raise ValueError("labels must be non-empty and have equal length")
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


def _distance(left: Point3D, right: Point3D) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _directed_distances(source: Sequence[Point3D], target: Sequence[Point3D]) -> list[float]:
    return [min(_distance(point, candidate) for candidate in target) for point in source]


def percentile_hausdorff(
    surface_a: Sequence[Point3D],
    surface_b: Sequence[Point3D],
    percentile: float = 95.0,
) -> float:
    """Symmetric percentile Hausdorff distance for physical-space surface points."""

    if not surface_a or not surface_b:
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
