from collections import defaultdict
from typing import Dict, Generator, Set, Type

import numpy as np

from hasher import Hasher


# Could be optimized using LSH Trees, implemented as a Trie
#
# See http://infolab.stanford.edu/~bawa/Pub/similarity.pdf at 5.2
class HashTable:
    def __init__(self, nbits: int, dim: int, hasher_class: Type[Hasher]):
        self.nbits: int = nbits
        self.dim: int = dim
        self.hasher: Hasher = hasher_class(nbits, dim)
        self.table: Dict[int, Set[int]] = defaultdict(set)

    def insert(self, vec: np.array, vec_id: int):
        hashed = self.hasher.hash(vec)
        self.table[hashed].add(vec_id)

    def bucket_iter(self, vec: np.array) -> Generator[Set[int], None, None]:
        hashed = self.hasher.hash(vec)
        for k in self._prefix_iter(hashed):
            if k in self.table:
                yield self.table[k]

    def _prefix_iter(self, num: int) -> int:
        for i in range(self.nbits**2):
            yield num ^ i
