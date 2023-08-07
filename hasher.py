# Implementation of different binary hashing schemes mentioned in "A Survey on Locality Sensitive Hashing Algorithms
# and their Applications". hash method must return a binary numpy array of fixed size (nbits).
#
# src: https://arxiv.org/pdf/2102.08942.pdf

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt

BinaryNumpyArray = npt.NDArray[np.bool_]


class Hasher(ABC):
    def __init__(self, nbits: int, dim: int, *args, **kwargs):
        self.nbits: int = nbits
        self.dim: int = dim

    @abstractmethod
    def hash(self, vec: np.array) -> BinaryNumpyArray:
        raise NotImplementedError


# Also known as "SimHash". Used for cosine distance
#
# ref:
# [1] https://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf
# [2] https://ieeexplore.ieee.org/document/6783789
class RandomProjectionHasher(Hasher):
    def __init__(self, nbits: int, dim: int):
        super().__init__(nbits, dim)

        # Random plane normals
        self._plane_normals = np.random.standard_normal((nbits, dim))

    def hash(self, vec: np.array) -> BinaryNumpyArray:
        return (np.dot(self._plane_normals, vec) > 0).astype(bool)
