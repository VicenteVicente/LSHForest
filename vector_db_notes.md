# Locality Sensitive Hashing (LSH)

Create a hash table with buckets that contains vectors that are likely close.

The **hash function** depends on the distance metric. Some examples for a given vector $x=[x_0, x_1, \ldots, x_n]$ are:

1. Hamming distance
    - $H(x) = x_i$
2. Minkowski distance
    - $H(x) = \lfloor\frac{ax + b}{w}\rfloor$
3. Cosine similarity
    - $H(x) = sign(ax)$
4. Jaccard distance
    - $H(x) = min\{{\pi(x_i)}\}$

The **storage and querying methods** also depends on the distance metric. Some examples are:

1. Hamming distance:
    - Use multiple hash functions and tables to increase the probability of collisions.
2. Minkowski distance:
    - Use multiple hash functions and tables to increase the probability of collisions. Usually at the final step use a hash function for reducing the number of keys. Visit the closest buckets to the hashed query point.
3. Cosine similarity:
    - Random hyperplane projections. (Maybe this could be used for Minkowski if the points has a similar magnitude).
4. Jaccard distance:
    - LSH Forest with prefix tree for iterating over the largest prefix match. (Maybe this could be used for Hamming and Cosine too)