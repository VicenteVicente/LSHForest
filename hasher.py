# Implementation of different hashing schemes mentioned in "A Survey on Locality Sensitive Hashing Algorithms
# and their Applications"
#
# src: https://arxiv.org/pdf/2102.08942.pdf

import random
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


# Also known as "SimHash". Used for cosine distance
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


# Used for Minkowski distance
#
# ref:
# [1] https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf
# [2] https://github.com/guoziqingbupt/Locality-sensitive-hashing/blob/master/e2LSH.py
# [3] https://www.mit.edu/~andoni/LSH/manual.pdf
class PStableHasher(Hasher):
    def __init__(self, nbits: int, dim: int, distribution: Literal["cauchy", "levy", "normal"] = "normal"):
        super().__init__(nbits, dim)

        # Constant proposed in [1]
        self.r = 4
        # Prime constant proposed in [3]
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


# Used for Hamming distance
#
# ref:
# [1] https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
class HammingHasher(Hasher):
    def __init__(self, nbits: int, dim: int):
        if nbits >= dim:
            raise ValueError(f"nbits ({nbits}) must be less than dim ({dim})")
        super().__init__(nbits, dim)
        # Precompute powers of 2 to speed up hashing
        self._powers_of_two = 1 << np.arange(dim)[::-1]
        # Keep just nbits set
        self._powers_of_two[np.random.choice(dim, dim - nbits)] = 0

    def hash(self, vec: np.array) -> int:
        return np.dot(vec, self._powers_of_two)
