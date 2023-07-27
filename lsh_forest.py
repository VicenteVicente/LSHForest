from typing import Generator, List, Literal, Tuple

import numpy as np

from hash_table import HashTable
from hasher import RandomProjectionHasher


# Return tuples of (vec_id, distance) "approximately" sorted by distance from vec
class QueryIter:
    def __init__(self, vec: np.array, lshforest_ref: "LSHForest"):
        self.vec = vec
        self.lshforest_ref = lshforest_ref
        # Bucket iterators for each hash table
        self.bucket_iters = [ht.bucket_iter(self.vec) for ht in self.lshforest_ref.hash_tables]
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
                    self.candidates.sort(key=lambda x: x[1], reverse=self.lshforest_ref.sorting_reverse)
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
        distance_metric: Literal["euclidean", "cosine"],
        data_ref: np.array,
    ):
        self.nbits: int = nbits
        self.dim: int = dim
        self.hash_tables: List[HashTable] = [
            HashTable(nbits, dim, RandomProjectionHasher) for _ in range(num_hash_tables)
        ]
        self.data_ref = data_ref
        if distance_metric == "euclidean":
            self.distance_func = lambda v1, v2: np.linalg.norm(v1 - v2)
            self.sorting_reverse = True
        elif distance_metric == "cosine":
            self.distance_func = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            self.sorting_reverse = False
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

    def index_data(self):
        for ht in self.hash_tables:
            for i in range(self.data_ref.shape[0]):
                ht.insert(self.data_ref[i], i)

    def query_iter(self, vec: np.array) -> QueryIter:
        return QueryIter(vec, self)
