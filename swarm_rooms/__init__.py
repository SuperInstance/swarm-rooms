"""Swarm Rooms — GPU-accelerated multi-agent room simulation."""

from .eisenstein import eisenstein_snap, eisenstein_delta
from .room_state import RoomState
from .network import SwarmRoomNetwork

__all__ = [
    "eisenstein_snap", "eisenstein_delta",
    "RoomState",
    "SwarmRoomNetwork",
]
