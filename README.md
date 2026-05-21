# swarm-rooms

**GPU-accelerated multi-agent room simulation with CRDT merge and Eisenstein lattice compression.**

Tests the hypothesis that swarm intelligence emerges from interconnection density, not individual capacity.

## Key Concepts

- **Rooms**: N agents, each with small local context and large observation of others
- **CRDT merge**: observation updates commute (order-independent convergence)
- **Deadband**: only propagate when delta exceeds tolerance
- **Eisenstein snap**: compress observations to Z[ω] lattice points

## Installation

```bash
pip install swarm-rooms[gpu]
```

Requires PyTorch >= 2.0.

## Quick Start

```python
from swarm_rooms import SwarmRoomNetwork

net = SwarmRoomNetwork(n_agents=256, device="cpu")
metrics = net.step()
print(metrics)  # propagation_rate, context_diversity, attention_entropy, ...
```

## License

MIT
