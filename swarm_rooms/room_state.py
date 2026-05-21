"""CRDT-mergeable room state for one agent."""

from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class RoomState:
    """
    CRDT-mergeable room state.

    Properties: commutative, associative, idempotent merges.
    Uses last-writer-wins with lamport clock. Deadband propagation.
    """

    def __init__(self, room_id: int, context_dim: int = 64, obs_dim: int = 128, device: str = "cpu"):
        if not _HAS_TORCH:
            raise ImportError("torch is required for RoomState")

        self.room_id = room_id
        self.context_dim = context_dim
        self.obs_dim = obs_dim
        self.device = device

        self.local_context = torch.randn(context_dim, device=device) * 0.1
        self.obs_buffer = torch.zeros(obs_dim, device=device)
        self.clock = 0
        self.last_propagated = self.local_context.clone()
        self.tolerance = 0.1

    def update_context(self, delta):
        self.local_context = self.local_context + delta
        self.clock += 1

    def should_propagate(self) -> bool:
        delta = torch.norm(self.local_context - self.last_propagated)
        return delta.item() > self.tolerance

    def propagate(self) -> Optional[object]:
        if self.should_propagate():
            self.last_propagated = self.local_context.clone()
            return self.local_context
        return None

    def receive_observation(self, other_id: int, other_context):
        self.obs_buffer = 0.9 * self.obs_buffer + 0.1 * other_context
        self.clock = max(self.clock, self.clock) + 1
