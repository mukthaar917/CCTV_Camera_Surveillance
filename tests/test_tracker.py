from src.tracking.trajectory import TrajectoryManager


def test_trajectory_summary() -> None:
    manager = TrajectoryManager(stationary_distance_threshold=5.0, stationary_duration_threshold=2.0)
    manager.update(1, 10, 10, 0, 0.0)
    manager.update(1, 13, 14, 10, 2.0)
    summary = manager.get_summary(1)
    assert summary is not None
    assert summary.displacement_pixels == 5.0
    assert summary.is_stationary is True


def test_remove_inactive_track() -> None:
    manager = TrajectoryManager(inactive_track_timeout=5.0)
    manager.update(7, 20, 20, 1, 1.0)
    assert manager.remove_inactive_tracks(7.0) == [7]
