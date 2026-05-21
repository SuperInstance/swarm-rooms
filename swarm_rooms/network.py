"""Swarm Room Network — GPU-accelerated swarm of interconnected rooms."""

import math
import time
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .eisenstein import eisenstein_snap_gpu, eisenstein_delta_gpu


class SwarmRoomNetwork:
    """
    GPU-accelerated swarm of interconnected rooms.

    Each room has small local context + large observation of others.
    CRDT merge for conflict-free convergence. Deadband for attention allocation.
    """

    def __init__(
        self,
        n_agents: int = 256,
        context_dim: int = 64,
        obs_dim: int = 128,
        device: str = "cpu",
        interconnection_density: float = 0.3,
        deadband_tolerance: float = 0.1,
        snap_radius: float = 1.0,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch is required for SwarmRoomNetwork")

        self.n_agents = n_agents
        self.context_dim = context_dim
        self.obs_dim = obs_dim
        self.device = device
        self.snap_radius = snap_radius

        self.contexts = torch.randn(n_agents, context_dim, device=device) * 0.1
        self.observations = torch.zeros(n_agents, obs_dim, device=device)
        self.connections = self._build_connections(interconnection_density)
        self.attention = torch.zeros(n_agents, n_agents, device=device)
        self.last_propagated = self.contexts.clone()
        self.tolerance = deadband_tolerance
        self.clocks = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.step_count = 0
        self.propagation_counts: List[float] = []
        self.attention_entropies: List[float] = []

    def __repr__(self) -> str:
        return f"SwarmRoomNetwork(agents={self.n_agents}, ctx_dim={self.context_dim}, obs_dim={self.obs_dim}, device={self.device})"

    def _build_connections(self, density: float):
        n = self.n_agents
        mask = torch.rand(n, n, device=self.device) < density
        mask.fill_diagonal_(False)
        return mask.float()

    def step(self, task_signal: Optional[object] = None) -> Dict:
        self.step_count += 1
        N = self.n_agents

        if task_signal is not None:
            self.contexts = self.contexts + task_signal.unsqueeze(0) * 0.01

        scores = self.contexts @ self.contexts.T / math.sqrt(self.context_dim)
        scores = scores.masked_fill(self.connections < 1, float('-inf'))
        self.attention = F.softmax(scores, dim=1)
        aggregated = self.attention @ self.contexts

        if self.context_dim != self.obs_dim:
            proj = torch.randn(self.context_dim, self.obs_dim, device=self.device) * 0.05
            new_observations = aggregated @ proj
        else:
            new_observations = aggregated

        proj_snap = torch.randn(self.obs_dim, 2, device=self.device) * 0.1
        obs_2d = new_observations @ proj_snap
        snapped_2d = eisenstein_snap_gpu(obs_2d, radius=self.snap_radius)
        deltas = eisenstein_delta_gpu(obs_2d, radius=self.snap_radius)

        context_deltas = torch.norm(self.contexts - self.last_propagated, dim=1)
        active_mask = context_deltas > self.tolerance
        n_propagating = active_mask.sum().item()

        alpha = 0.1
        self.observations = (1 - alpha) * self.observations + alpha * new_observations
        context_update = self.observations @ torch.randn(self.obs_dim, self.context_dim, device=self.device) * 0.01
        self.contexts = self.contexts + context_update
        self.last_propagated[active_mask] = self.contexts[active_mask].clone()
        self.clocks += 1

        with torch.no_grad():
            attn_entropy = -(self.attention * (self.attention + 1e-8).log()).sum(dim=1)
            centroid = self.contexts.mean(dim=0)
            diversity = torch.norm(self.contexts - centroid, dim=1).mean().item()

        metrics = {
            "step": self.step_count,
            "propagating": n_propagating,
            "propagation_rate": n_propagating / N,
            "attention_entropy": attn_entropy.mean().item(),
            "snap_hit_rate": (deltas < 0.01).float().mean().item(),
            "context_diversity": diversity,
            "mean_delta": deltas.mean().item(),
            "max_delta": deltas.max().item(),
        }
        self.propagation_counts.append(n_propagating)
        self.attention_entropies.append(metrics["attention_entropy"])
        return metrics

    def benchmark(self, n_steps: int = 100) -> Dict:
        for _ in range(5):
            self.step()
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        metrics_list = [self.step() for _ in range(n_steps)]
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        return {
            "n_agents": self.n_agents,
            "steps_per_sec": round(n_steps / elapsed, 1),
            "elapsed_sec": round(elapsed, 3),
            "final_metrics": metrics_list[-1],
            "mean_propagation_rate": np.mean([m["propagation_rate"] for m in metrics_list]),
            "mean_context_diversity": np.mean([m["context_diversity"] for m in metrics_list]),
        }
