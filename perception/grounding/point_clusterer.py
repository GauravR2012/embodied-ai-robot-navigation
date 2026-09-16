from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .models import ProjectedScanRay


@dataclass(frozen=True)
class ScanCluster:
    """
    A contiguous group of LaserScan rays associated with a visual target.

    The cluster is represented by the original projected rays rather
    than by a single point so that downstream grounding can apply
    additional robust statistics.
    """

    rays: tuple[ProjectedScanRay, ...]

    @property
    def size(self) -> int:
        """Number of supporting LaserScan rays."""

        return len(self.rays)

    @property
    def min_range_m(self) -> float:
        """Minimum measured range in the cluster."""

        return min(ray.range_m for ray in self.rays)

    @property
    def max_range_m(self) -> float:
        """Maximum measured range in the cluster."""

        return max(ray.range_m for ray in self.rays)

    @property
    def median_range_m(self) -> float:
        """Median measured range in the cluster."""

        values = sorted(ray.range_m for ray in self.rays)
        middle = len(values) // 2

        if len(values) % 2 == 1:
            return values[middle]

        return 0.5 * (values[middle - 1] + values[middle])


@dataclass(frozen=True)
class ClustererConfig:
    """
    Configuration for contiguous LaserScan clustering.

    `max_index_gap` controls how many missing scan indices may occur
    before two rays are considered part of different objects.

    `max_range_jump_m` prevents a sudden depth discontinuity from
    being merged into one physical cluster.
    """

    max_index_gap: int = 1
    max_range_jump_m: float = 0.35
    min_cluster_size: int = 2
    max_cluster_size: int = 100

    def __post_init__(self) -> None:
        if self.max_index_gap < 1:
            raise ValueError("max_index_gap must be at least 1")

        if self.max_range_jump_m <= 0.0:
            raise ValueError("max_range_jump_m must be positive")

        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")

        if self.max_cluster_size < self.min_cluster_size:
            raise ValueError(
                "max_cluster_size must be >= min_cluster_size"
            )


class ScanPointClusterer:
    """
    Groups projected LaserScan rays into spatially consistent clusters.

    Input rays are expected to have already been filtered against a VLM
    bounding box.

    Clustering uses two constraints:

    1. Scan-index continuity.
    2. Range continuity.

    This deliberately avoids assuming that the object occupies every
    pixel or every LaserScan ray inside its visual bounding box.
    """

    def __init__(
        self,
        config: ClustererConfig | None = None,
    ) -> None:
        self._config = config or ClustererConfig()

    @staticmethod
    def _range_is_consistent(
        previous: ProjectedScanRay,
        current: ProjectedScanRay,
        max_range_jump_m: float,
    ) -> bool:
        """
        Determine whether two consecutive rays have compatible ranges.

        A range discontinuity is useful because the boundary between an
        object and a wall/background often produces a sudden jump.
        """

        return (
            abs(current.range_m - previous.range_m)
            <= max_range_jump_m
        )

    def cluster(
        self,
        rays: Iterable[ProjectedScanRay],
    ) -> tuple[ScanCluster, ...]:
        """
        Group projected rays into contiguous scan clusters.
        """

        ordered = sorted(
            rays,
            key=lambda ray: ray.scan_index,
        )

        if not ordered:
            return ()

        clusters: list[ScanCluster] = []
        current: list[ProjectedScanRay] = [ordered[0]]

        for ray in ordered[1:]:
            previous = current[-1]

            index_gap = ray.scan_index - previous.scan_index

            contiguous = (
                0 < index_gap <= self._config.max_index_gap
            )

            range_consistent = self._range_is_consistent(
                previous,
                ray,
                self._config.max_range_jump_m,
            )

            if contiguous and range_consistent:
                current.append(ray)
                continue

            self._append_if_valid(
                clusters,
                current,
            )

            current = [ray]

        self._append_if_valid(
            clusters,
            current,
        )

        return tuple(clusters)

    def _append_if_valid(
        self,
        clusters: list[ScanCluster],
        rays: list[ProjectedScanRay],
    ) -> None:
        """Add a cluster when its size satisfies configuration limits."""

        size = len(rays)

        if size < self._config.min_cluster_size:
            return

        if size > self._config.max_cluster_size:
            # Large contiguous regions are usually background surfaces
            # such as walls. Split them deterministically into chunks
            # rather than silently accepting an unbounded cluster.
            for start in range(
                0,
                size,
                self._config.max_cluster_size,
            ):
                chunk = rays[
                    start : start + self._config.max_cluster_size
                ]

                if len(chunk) >= self._config.min_cluster_size:
                    clusters.append(
                        ScanCluster(tuple(chunk))
                    )

            return

        clusters.append(
            ScanCluster(tuple(rays))
        )

    @staticmethod
    def select_best_cluster(
        clusters: Iterable[ScanCluster],
        *,
        expected_range_m: float | None = None,
    ) -> ScanCluster | None:
        """
        Select the most plausible cluster.

        Without an expected range, the strongest cluster is the one
        with the most supporting rays.

        When an expected range is supplied, range agreement is used
        as a secondary criterion.
        """

        candidates = tuple(clusters)

        if not candidates:
            return None

        if expected_range_m is not None:
            if not math.isfinite(expected_range_m):
                raise ValueError(
                    "expected_range_m must be finite"
                )

            return min(
                candidates,
                key=lambda cluster: (
                    abs(
                        cluster.median_range_m
                        - expected_range_m
                    ),
                    -cluster.size,
                ),
            )

        return max(
            candidates,
            key=lambda cluster: (
                cluster.size,
                -cluster.median_range_m,
            ),
        )