"""Tests for RoomState CRDT."""

import pytest

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestRoomState:
    def test_init(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=0, context_dim=32, obs_dim=64, device="cpu")
        assert rs.room_id == 0
        assert rs.context_dim == 32
        assert rs.obs_dim == 64
        assert rs.clock == 0
        assert rs.local_context.shape == (32,)
        assert rs.obs_buffer.shape == (64,)

    def test_update_context(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=1, context_dim=16, obs_dim=32, device="cpu")
        delta = torch.ones(16) * 0.5
        rs.update_context(delta)
        assert rs.clock == 1
        # local_context should have changed
        assert torch.norm(rs.local_context).item() > 0

    def test_should_propagate_false_initially(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=2, context_dim=16, obs_dim=32, device="cpu")
        # Small initial context, tolerance=0.1
        # last_propagated starts equal to local_context
        assert not rs.should_propagate()

    def test_should_propagate_true_after_large_update(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=3, context_dim=16, obs_dim=32, device="cpu")
        delta = torch.ones(16) * 5.0
        rs.update_context(delta)
        assert rs.should_propagate()

    def test_propagate_returns_context_when_changed(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=4, context_dim=16, obs_dim=32, device="cpu")
        delta = torch.ones(16) * 5.0
        rs.update_context(delta)
        result = rs.propagate()
        assert result is not None
        assert result.shape == (16,)

    def test_propagate_returns_none_when_unchanged(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=5, context_dim=16, obs_dim=32, device="cpu")
        result = rs.propagate()
        assert result is None

    def test_receive_observation(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=6, context_dim=16, obs_dim=32, device="cpu")
        other_ctx = torch.ones(32)
        rs.receive_observation(other_id=99, other_context=other_ctx)
        # obs_buffer should have changed (0.9*zeros + 0.1*ones = 0.1*ones)
        assert torch.norm(rs.obs_buffer).item() > 0

    def test_multiple_updates(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=7, context_dim=8, obs_dim=16, device="cpu")
        for _ in range(10):
            rs.update_context(torch.randn(8) * 0.5)
        assert rs.clock == 10

    def test_custom_tolerance(self):
        from swarm_rooms.room_state import RoomState
        rs = RoomState(room_id=8, context_dim=16, obs_dim=32, device="cpu")
        rs.tolerance = 100.0  # very high tolerance
        rs.update_context(torch.ones(16) * 5.0)
        assert not rs.should_propagate()  # should not exceed tolerance
