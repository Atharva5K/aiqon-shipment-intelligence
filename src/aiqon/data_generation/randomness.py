"""Stable, namespace-separated random seed derivation."""

import hashlib

import numpy as np


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed without Python's process-randomized hash()."""
    material = "|".join([str(master_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big", signed=False)


def rng_for(master_seed: int, *parts: object) -> np.random.Generator:
    """Create an independent PCG64 generator for a seed namespace."""
    return np.random.default_rng(derive_seed(master_seed, *parts))
