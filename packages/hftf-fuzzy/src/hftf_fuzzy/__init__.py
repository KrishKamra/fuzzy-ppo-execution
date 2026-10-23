"""hftf_fuzzy: evolving fuzzy neural network in torch (ARCHITECTURE §5). No dependency on hftf_sim.

Modules: layer (FuzzyTSK: Gaussian antecedents, softmax-normalised firing, per-rule linear consequents,
structure edits with optimizer-state remapping), evolution (Evolver: add / merge / prune, function-preserving
insertion; k-means initialisation), interpret (linguistic rule report).
"""

from hftf_fuzzy.evolution import Event, EvolutionConfig, Evolver, kmeans
from hftf_fuzzy.layer import FuzzyTSK, bhattacharyya, remap_optimizer_state

__all__ = [
    "EvolutionConfig",
    "Event",
    "Evolver",
    "FuzzyTSK",
    "bhattacharyya",
    "kmeans",
    "remap_optimizer_state",
]
