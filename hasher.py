from abc import ABC, abstractmethod

import numpy as np


class Hasher(ABC):
    def __init__(self, nbits: int, dim: int, *args, **kwargs):
        self.nbits: int = nbits
        self.dim: int = dim

    @abstractmethod
    def hash(self, vec: np.array) -> int:
        raise NotImplementedError


class RandomProjectionHasher(Hasher):
    def __init__(self, nbits: int, dim: int):
        super().__init__(nbits, dim)
        # Random plane normals
        self._plane_normals = np.random.rand(nbits, dim) - 0.5
        # Precompute powers of 2 to speed up hashing
        self._powers_of_two = 1 << np.arange(nbits)[::-1]

    def hash(self, vec: np.array) -> int:
        return np.dot((np.dot(self._plane_normals, vec) > 0).astype(np.uint8), self._powers_of_two)
