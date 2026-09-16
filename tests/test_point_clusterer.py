from __future__ import annotations

import pytest

from perception.grounding.models import ProjectedScanRay
from perception.grounding.point_clusterer import (
    ClustererConfig,
    ScanPointClusterer,
)


def make_ray(
    scan_index: int,
    range_m: float,
    *,
    pixel_u: float = 320.0,
    pixel_v: float = 240.0,
) -> ProjectedScanRay:
    return ProjectedScanRay(
        scan_index=scan_index,
        range_m=range_m,
        angle_rad=0.1 * scan_index,
        camera_x_m=0.0,
        camera_y_m=0.0,
        camera_z_m=range_m,
        pixel_u=pixel_u,
        pixel_v=pixel_v,
    )


def test_contiguous_rays_form_one_cluster() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            min_cluster_size=2,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.05),
        make_ray(12, 2.02),
        make_ray(13, 2.08),
    ]

    clusters = clusterer.cluster(rays)

    assert len(clusters) == 1
    assert clusters[0].size == 4
    assert clusters[0].median_range_m == pytest.approx(2.035)


def test_scan_index_gap_splits_clusters() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            max_index_gap=1,
            min_cluster_size=2,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.05),
        make_ray(13, 2.02),
        make_ray(14, 2.08),
    ]

    clusters = clusterer.cluster(rays)

    assert len(clusters) == 2
    assert clusters[0].size == 2
    assert clusters[1].size == 2


def test_range_discontinuity_splits_clusters() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            max_range_jump_m=0.35,
            min_cluster_size=2,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.05),
        make_ray(12, 4.5),
        make_ray(13, 4.55),
    ]

    clusters = clusterer.cluster(rays)

    assert len(clusters) == 2
    assert clusters[0].size == 2
    assert clusters[1].size == 2


def test_single_ray_cluster_is_rejected_by_default() -> None:
    clusterer = ScanPointClusterer()

    rays = [
        make_ray(10, 2.0),
    ]

    clusters = clusterer.cluster(rays)

    assert clusters == ()


def test_cluster_statistics() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            min_cluster_size=1,
            max_range_jump_m=0.35,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.10),
        make_ray(12, 2.20),
    ]

    clusters = clusterer.cluster(rays)

    assert len(clusters) == 1

    cluster = clusters[0]

    assert cluster.min_range_m == pytest.approx(2.0)
    assert cluster.max_range_m == pytest.approx(2.20)
    assert cluster.median_range_m == pytest.approx(2.10)

    
def test_best_cluster_prefers_supporting_rays() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            min_cluster_size=1,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.0),
        make_ray(12, 2.0),
        make_ray(20, 5.0),
    ]

    clusters = clusterer.cluster(rays)

    best = clusterer.select_best_cluster(clusters)

    assert best is not None
    assert best.size == 3
    assert best.median_range_m == pytest.approx(2.0)


def test_best_cluster_can_use_expected_range() -> None:
    clusterer = ScanPointClusterer(
        ClustererConfig(
            min_cluster_size=1,
        )
    )

    rays = [
        make_ray(10, 2.0),
        make_ray(11, 2.0),
        make_ray(20, 5.0),
        make_ray(21, 5.0),
    ]

    clusters = clusterer.cluster(rays)

    best = clusterer.select_best_cluster(
        clusters,
        expected_range_m=4.8,
    )

    assert best is not None
    assert best.median_range_m == pytest.approx(5.0)


def test_empty_input_returns_no_clusters() -> None:
    clusterer = ScanPointClusterer()

    assert clusterer.cluster([]) == ()
    assert clusterer.select_best_cluster([]) is None
