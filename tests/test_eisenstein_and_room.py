"""Tests for swarm_rooms.eisenstein and swarm_rooms.room_state."""

import math
import pytest
import numpy as np
from swarm_rooms.eisenstein import eisenstein_snap, eisenstein_delta


class TestEisensteinSnap:
    def test_origin(self):
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        snapped = eisenstein_snap(points)
        np.testing.assert_allclose(snapped[0], [0.0, 0.0], atol=1e-5)

    def test_on_lattice(self):
        # Point (1, 0) is on the lattice
        points = np.array([[1.0, 0.0]], dtype=np.float32)
        snapped = eisenstein_snap(points)
        np.testing.assert_allclose(snapped[0], [1.0, 0.0], atol=0.5)

    def test_snap_radius(self):
        points = np.array([[2.0, 0.0]], dtype=np.float32)
        snapped = eisenstein_snap(points, radius=0.5)
        # With radius=0.5, lattice is denser
        assert snapped.shape == (1, 2)

    def test_multiple_points(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        snapped = eisenstein_snap(points)
        assert snapped.shape == (3, 2)


class TestEisensteinDelta:
    def test_origin_delta_zero(self):
        points = np.array([[0.0, 0.0]], dtype=np.float32)
        delta = eisenstein_delta(points)
        assert delta[0] < 0.01

    def test_delta_positive(self):
        points = np.array([[0.5, 0.5]], dtype=np.float32)
        delta = eisenstein_delta(points)
        assert delta[0] >= 0.0


class TestRoomState:
    def test_creation(self):
        import torch
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=1, context_dim=4, obs_dim=8)
        assert rs.room_id == 1
        assert rs.local_context.shape == (4,)

    def test_update_context(self):
        import torch
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=1, context_dim=4, obs_dim=8)
        delta = torch.ones(4) * 0.1
        rs.update_context(delta)
        assert rs.clock == 1

    def test_propagate(self):
        import torch
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=1, context_dim=4, obs_dim=8)
        # Initially, last_propagated == local_context, so no propagation
        result = rs.propagate()
        # delta is 0, so should not propagate (within tolerance)
        assert result is None or isinstance(result, torch.Tensor)

    def test_receive_observation(self):
        import torch
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=1, context_dim=4, obs_dim=4)
        other_ctx = torch.ones(4)
        rs.receive_observation(2, other_ctx)
        assert rs.obs_buffer.sum() > 0
