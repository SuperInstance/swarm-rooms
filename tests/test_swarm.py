"""Tests for swarm-rooms."""

import numpy as np
import pytest

from swarm_rooms.eisenstein import eisenstein_snap, eisenstein_delta

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TestEisensteinSnap:
    def test_origin_snaps_to_self(self):
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        snapped = eisenstein_snap(points)
        np.testing.assert_allclose(snapped, points, atol=1e-5)

    def test_lattice_point(self):
        # Origin is always on the lattice
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        snapped = eisenstein_snap(points)
        np.testing.assert_allclose(snapped, points, atol=1e-5)

    def test_delta_nonzero_for_off_lattice(self):
        points = np.array([[0.3, 0.3]], dtype=np.float32)
        delta = eisenstein_delta(points)
        assert delta[0] > 0.01

    def test_delta_zero_on_lattice(self):
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        delta = eisenstein_delta(points)
        np.testing.assert_allclose(delta, [0.0], atol=1e-5)

    def test_batch(self):
        points = np.random.randn(100, 2).astype(np.float32)
        snapped = eisenstein_snap(points)
        assert snapped.shape == (100, 2)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestSwarmRoomNetwork:
    def test_single_step(self):
        from swarm_rooms import SwarmRoomNetwork
        net = SwarmRoomNetwork(n_agents=16, device="cpu")
        metrics = net.step()
        assert "propagation_rate" in metrics
        assert "context_diversity" in metrics

    def test_multiple_steps(self):
        from swarm_rooms import SwarmRoomNetwork
        net = SwarmRoomNetwork(n_agents=16, device="cpu")
        for _ in range(5):
            net.step()
        assert net.step_count == 5

    def test_benchmark(self):
        from swarm_rooms import SwarmRoomNetwork
        net = SwarmRoomNetwork(n_agents=16, device="cpu")
        bench = net.benchmark(n_steps=10)
        assert bench["steps_per_sec"] > 0
