from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from hash_table import HashTable
from hasher import RandomProjectionHasher


# Return tuples of (vec_id, distance) "approximately" sorted by distance from vec
class QueryIter:
    def __init__(self, vec: npt.ArrayLike, lshforest_ref: "LSHForest"):
        self.vec = vec
        self.lshforest_ref = lshforest_ref
        # Bucket iterators for each hash table
        self.bucket_iters = [ht.get_prefix_bucket_iter(self.vec) for ht in self.lshforest_ref.hash_tables]
        # Current vec_ids generated from each hash table
        self.current_buckets = [next(bucket_iter) for bucket_iter in self.bucket_iters]
        # Candidates
        self.candidates: List[Tuple[int, float]] = list()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, float]:
        if self.candidates:
            # Return the next candidate
            return self.candidates.pop()
        else:
            # Populate candidates
            while True:
                intersection = set.intersection(*self.current_buckets)
                if intersection:
                    # There are intersections between current buckets
                    # Remove intersection
                    for bucket in self.current_buckets:
                        bucket.difference_update(intersection)
                    # Compute distance
                    for vec_id in intersection:
                        distance = self.lshforest_ref.distance_func(self.lshforest_ref.data_ref[vec_id], self.vec)
                        self.candidates.append((vec_id, distance))
                    # Sort candidates
                    self.candidates.sort(key=lambda x: x[1], reverse=False)
                    # Return the next candidate
                    return self.candidates.pop()
                else:
                    try:
                        # Extend the smallest bucket to increase the intersection probability
                        idx = self._smallest_bucket_index()
                        self.current_buckets[idx].update(next(self.bucket_iters[idx]))
                    except StopIteration:
                        # All bucket iters are exhausted
                        raise StopIteration

    # Return the index of the smallest bucket
    def _smallest_bucket_index(self):
        min_idx, min_val = 0, len(self.current_buckets[0])
        for idx, bucket in enumerate(self.current_buckets):
            if len(bucket) < min_val:
                min_idx, min_val = idx, len(bucket)
        return min_idx


class LSHForest:
    def __init__(
        self,
        nbits: int,
        dim: int,
        num_hash_tables: int,
        distance_metric: Literal["cosine", "jaccard"],
        data_ref: npt.ArrayLike,
    ):
        self.nbits: int = nbits
        self.dim: int = dim
        self.hash_tables: List[HashTable] = [
            HashTable(hasher=RandomProjectionHasher(nbits, dim)) for _ in range(num_hash_tables)
        ]
        self.data_ref = data_ref

        if distance_metric == "cosine":
            self.distance_func = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif distance_metric == "jaccard":
            self.distance_func = lambda v1, v2: np.bitwise_and(v1, v2).sum().astype(np.double) / np.bitwise_or(
                v1, v2
            ).sum().astype(np.double)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Clear all the hash tables and index the data
    def index_data(self):
        for table_idx in range(len(self.hash_tables)):
            self._handle_index_data(table_idx)

    # Get an iterator that yields each indexed vector sorted by the distance from input vec
    def query_iter(self, vec: npt.ArrayLike) -> QueryIter:
        return QueryIter(vec, self)

    # Clear and index the data on the hash table at index
    def _handle_index_data(self, table_idx: int):
        self.hash_tables[table_idx].clear()
        for vec_idx in tqdm(
            range(self.data_ref.shape[0]),
            desc=f"Indexing table {table_idx+1}/{len(self.hash_tables)}",
        ):
            self.hash_tables[table_idx].insert(self.data_ref[vec_idx], vec_idx)
