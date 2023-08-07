from typing import Any, Generator, Set

import numpy as np
import numpy.typing as npt

from hasher import Hasher, RandomProjectionHasher
from patricia_trie import PATRICIATrie


class HashTable:
    def __init__(self, hasher: Hasher):
        self._hasher: Hasher = hasher
        self._trie: PATRICIATrie = PATRICIATrie()
        self._table = dict()

    def __setitem__(self, vec: npt.ArrayLike, value: Any) -> None:
        self.insert(vec, value)

    # Insert a vector into the hash table
    def insert(self, vec: npt.ArrayLike, value: Any) -> None:
        hashed = bytes(self._hasher.hash(vec))
        if not hashed in self._table:
            # Create a reference of the bucket in the trie
            self._table[hashed] = set()
            self._trie[hashed] = self._table[hashed]
        self._table[hashed].add(value)

    # Returns an iterator that yields each bucket sorted by the lenght of the prefix match
    def get_prefix_bucket_iter(self, vec: npt.ArrayLike) -> Generator[Set[Any], None, None]:
        hashed = bytes(self._hasher.hash(vec))
        for node in self._trie.get_prefix_iter(hashed):
            yield node.value

    # Clear the hash table
    def clear(self) -> None:
        self._table = dict()
        self._trie = PATRICIATrie()
