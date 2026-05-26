import numpy as np
"""Tests for swarm_rooms.network — SwarmRoomNetwork."""

import pytest
import torch
from swarm_rooms.network import SwarmRoomNetwork


class TestSwarmRoomNetwork:
    def test_creation(self):
        net = SwarmRoomNetwork(n_agents=8, context_dim=16, obs_dim=16)
        assert net.n_agents == 8
        assert repr(net)

    def test_step(self):
        net = SwarmRoomNetwork(n_agents=8, context_dim=16, obs_dim=16)
        metrics = net.step()
        assert isinstance(metrics, dict)
        assert "step" in metrics
        assert "propagation_rate" in metrics
        assert metrics["step"] == 1

    def test_step_with_signal(self):
        net = SwarmRoomNetwork(n_agents=8, context_dim=16, obs_dim=16)
        signal = torch.randn(16)
        metrics = net.step(task_signal=signal)
        assert metrics["step"] == 1

    def test_multiple_steps(self):
        net = SwarmRoomNetwork(n_agents=8, context_dim=16, obs_dim=16)
        for i in range(5):
            metrics = net.step()
        assert net.step_count == 5

    def test_metrics_range(self):
        net = SwarmRoomNetwork(n_agents=8, context_dim=16, obs_dim=16)
        metrics = net.step()
        assert 0.0 <= metrics["propagation_rate"] <= 1.0
        assert metrics["snap_hit_rate"] >= 0.0
        assert metrics["context_diversity"] >= 0.0 or np.isnan(metrics["context_diversity"])
