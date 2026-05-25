# swarm-rooms

**GPU-accelerated multi-agent room simulation with CRDT merge and Eisenstein lattice compression.**

Tests the hypothesis that swarm intelligence emerges from interconnection density, not individual capacity. Extracted from [plato-training](https://github.com/SuperInstance/plato-training).

## The Swarm Room Concept

A **swarm room** is a PLATO room where N agents share observations:

1. Each agent maintains a **small local context** (private state)
2. Each agent has a **large observation of others** (shared state)
3. Observations update via **CRDT merge** — order-independent, eventually consistent
4. Only propagate when delta exceeds **deadband tolerance**
5. Observations are compressed to **Z[ω] Eisenstein lattice** points

The hypothesis: intelligence scales with *interconnection density* (how much each agent observes about others), not with individual agent capacity.

```
Agent 0 ←→ Agent 1 ←→ Agent 2 ←→ ... ←→ Agent N
   ↕          ↕          ↕                    ↕
 CRDT merge propagates observations across the mesh
 Eisenstein snap compresses values to lattice points
 Deadband filters prevent trivial updates
```

## Key Concepts

| Concept | What It Does |
|---------|-------------|
| **Rooms** | N agents, each with local context + observation of others |
| **CRDT merge** | Observation updates commute — order-independent convergence |
| **Deadband** | Only propagate when delta exceeds tolerance |
| **Eisenstein snap** | Compress observations to Z[ω] lattice points |
| **Propagation rate** | Fraction of agents that converge per step |
| **Attention entropy** | Measures how uniformly agents distribute attention |

## Installation

```bash
pip install swarm-rooms[gpu]
```

Requires PyTorch >= 2.0. Falls back to CPU if no GPU available.

## Quick Start

```python
from swarm_rooms import SwarmRoomNetwork

# Create a network of 256 agents
net = SwarmRoomNetwork(n_agents=256, device="cpu")

# Run simulation steps
for step in range(100):
    metrics = net.step()
    print(f"Step {step}: "
          f"propagation={metrics['propagation_rate']:.3f}, "
          f"diversity={metrics['context_diversity']:.3f}, "
          f"entropy={metrics['attention_entropy']:.3f}")
```

## Architecture

```
SwarmRoomNetwork
├── Agent states (local context + observation buffer)
├── CRDT merge engine (commutative update resolution)
├── Deadband filter (hysteresis-based propagation)
├── Eisenstein compressor (lattice snap + quantization)
└── Metrics collector (propagation, diversity, entropy)
```

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| `propagation_rate` | Fraction of agents updated per step |
| `context_diversity` | Variance of local contexts across agents |
| `attention_entropy` | Shannon entropy of attention distribution |
| `convergence_speed` | Steps to reach deadband threshold |

## Relationship to PLATO

Swarm rooms are a specific room type within the PLATO ecosystem. They emerged from research on how agents coordinate in shared spaces:

- **[plato-core](https://github.com/SuperInstance/plato-core)** — Base room/tile types
- **[plato-training](https://github.com/SuperInstance/plato-training)** — Training framework (swarm rooms extracted from here)
- **[eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed)** — Eisenstein lattice embedding
- **[constraint-substrate](https://github.com/SuperInstance/constraint-substrate)** — Core math primitives

## License

MIT
