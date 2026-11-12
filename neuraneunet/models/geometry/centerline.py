"""Deterministic polyline operations in physical millimetre coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

Point3D = tuple[float, float, float]


def _point(point: Point3D) -> Point3D:
    if len(point) != 3 or any(not isfinite(value) for value in point):
        raise ValueError("points must contain three finite coordinates")
    x, y, z = point
    return float(x), float(y), float(z)


def _centerline(points: Sequence[Point3D]) -> tuple[Point3D, ...]:
    if len(points) < 2:
        raise ValueError("a centerline requires at least two points")
    converted = tuple(_point(point) for point in points)
    if all(left == right for left, right in zip(converted, converted[1:])):
        raise ValueError("centerline must contain a non-zero segment")
    return converted


def _distance(left: Point3D, right: Point3D) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _arc_lengths(points: Sequence[Point3D]) -> tuple[float, ...]:
    result = [0.0]
    for left, right in zip(points, points[1:]):
        result.append(result[-1] + _distance(left, right))
    return tuple(result)


def cumulative_arc_length(points: Sequence[Point3D]) -> tuple[float, ...]:
    return _arc_lengths(_centerline(points))


def interpolate_along_centerline(points: Sequence[Point3D], arc_length_mm: float) -> Point3D:
    points = _centerline(points)
    distances = _arc_lengths(points)
    if not isfinite(arc_length_mm) or not 0 <= arc_length_mm <= distances[-1]:
        raise ValueError("arc_length_mm must lie on the centerline")
    for index, (start, end) in enumerate(zip(distances, distances[1:])):
        if arc_length_mm <= end and end > start:
            fraction = (arc_length_mm - start) / (end - start)
            return tuple(
                left + fraction * (right - left)
                for left, right in zip(points[index], points[index + 1])
            )  # type: ignore[return-value]
    return points[-1]


@dataclass(frozen=True, slots=True)
class Projection:
    point: Point3D
    segment_index: int
    segment_fraction: float
    arc_length_mm: float
    distance_mm: float


def project_to_centerline(point: Point3D, centerline: Sequence[Point3D]) -> Projection:
    centerline = _centerline(centerline)
    return _project(_point(point), centerline, _arc_lengths(centerline))


def _project(point: Point3D, centerline: Sequence[Point3D], distances: Sequence[float]) -> Projection:
    best: Projection | None = None
    for index, (start, end) in enumerate(zip(centerline, centerline[1:])):
        vector = tuple(right - left for left, right in zip(start, end))
        denominator = sum(value * value for value in vector)
        if denominator == 0:
            continue
        offset = tuple(value - origin for value, origin in zip(point, start))
        fraction = max(0.0, min(1.0, sum(a * b for a, b in zip(offset, vector)) / denominator))
        projected = tuple(origin + fraction * step for origin, step in zip(start, vector))
        distance = _distance(point, projected)  # type: ignore[arg-type]
        candidate = Projection(
            point=projected,  # type: ignore[arg-type]
            segment_index=index,
            segment_fraction=fraction,
            arc_length_mm=distances[index] + fraction * sqrt(denominator),
            distance_mm=distance,
        )
        if best is None or candidate.distance_mm < best.distance_mm:
            best = candidate
    if best is None:
        raise ValueError("centerline contains no projectable segment")
    return best


def landing_zone_offset(
    reference: Point3D, prediction: Point3D, centerline: Sequence[Point3D]
) -> float:
    """Absolute along-centerline offset between two landing-zone points."""

    centerline = _centerline(centerline)
    distances = _arc_lengths(centerline)
    reference_projection = _project(_point(reference), centerline, distances)
    prediction_projection = _project(_point(prediction), centerline, distances)
    return abs(reference_projection.arc_length_mm - prediction_projection.arc_length_mm)
