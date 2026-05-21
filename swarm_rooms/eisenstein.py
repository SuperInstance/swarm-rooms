"""Eisenstein lattice snap — CPU fallback with optional GPU."""

import math
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import numpy as np


def eisenstein_snap(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Snap 2D points to Eisenstein lattice Z[ω]. Works on CPU (numpy)."""
    basis = np.array([[1.0, 0.0], [-0.5, math.sqrt(3) / 2]], dtype=np.float32)
    basis_inv = np.linalg.inv(basis)
    coords = points @ basis_inv.T
    snapped_coords = np.round(coords)
    snapped_points = snapped_coords @ basis.T * radius
    return snapped_points.astype(np.float32)


def eisenstein_delta(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Compute distance from nearest Eisenstein lattice point."""
    snapped = eisenstein_snap(points, radius)
    delta = np.linalg.norm(points - snapped, axis=-1)
    return delta.astype(np.float32)


def eisenstein_snap_gpu(points, radius: float = 1.0):
    """GPU version of eisenstein_snap (requires torch)."""
    if not _HAS_TORCH:
        raise ImportError("torch is required for GPU operations")
    import torch
    import torch.nn.functional as F
    basis = torch.tensor([[1.0, 0.0], [-0.5, math.sqrt(3) / 2]],
                          device=points.device, dtype=torch.float32)
    basis_inv = torch.linalg.inv(basis)
    coords = points @ basis_inv.T
    snapped_coords = torch.round(coords)
    snapped_points = snapped_coords @ basis.T * radius
    return snapped_points


def eisenstein_delta_gpu(points, radius: float = 1.0):
    """GPU version of eisenstein_delta (requires torch)."""
    snapped = eisenstein_snap_gpu(points, radius)
    return torch.norm(points - snapped, dim=-1)
