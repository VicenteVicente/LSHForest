from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class Hasher(ABC):
    def __init__(self, nbits: int, dim: int, *args, **kwargs):
        self.nbits: int = nbits
        self.dim: int = dim

    @abstractmethod
    def hash(self, vec: np.array) -> int:
        raise NotImplementedError


# Hamming distance
# "Similarity Estimation Techniques from RoundingAlgorithms"
#
# ref:
# [1] https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf
class RandomProjectionHasher(Hasher):
    def __init__(self, nbits: int, dim: int):
        super().__init__(nbits, dim)

        # Random plane normals
        self._plane_normals = np.random.rand(nbits, dim) - 0.5
        # Precompute powers of 2 to speed up hashing
        self._powers_of_two = 1 << np.arange(nbits)[::-1]

    def hash(self, vec: np.array) -> int:
        return np.dot((np.dot(self._plane_normals, vec) > 0).astype(np.uint8), self._powers_of_two)


# Minkowski distance (for p=1 and p=2)
# "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions"
#
# ref:
# [1] https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf
# [2] https://github.com/guoziqingbupt/Locality-sensitive-hashing/blob/master/e2LSH.py
# [3] https://www.mit.edu/~andoni/LSH/manual.pdf
class PStableHasher(Hasher):
    def __init__(self, nbits: int, dim: int, distribution: Literal["cauchy", "levy", "normal"]):
        super().__init__(nbits, dim)

        # Constant proposed in [1]
        self.r = 4
        # Constant proposed in [3]
        self.C = pow(2, 32) - 5
        self.b = np.random.randint(0, self.r)
        if distribution == "cauchy":
            self.a = [np.random.standard_cauchy() for _ in range(dim)]
        elif distribution == "levy":
            self.a = [np.random.standard_levy() for _ in range(dim)]
        elif distribution == "normal":
            self.a = [np.random.standard_normal() for _ in range(dim)]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def hash(self, vec: np.array) -> int:
        # h_1(a), defined in [3] at 3.5.2.
        return (int(np.dot(self.a, vec) + self.b) / self.r) % pow(2, self.nbits)
