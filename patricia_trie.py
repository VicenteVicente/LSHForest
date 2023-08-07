from typing import Any, Dict, Generator

import numpy as np


class PrefixIter:
    def __init__(self, patricia_trie_ref: "PATRICIATrie", prefix: bytes):
        self.patricia_trie_ref: "PATRICIATrie" = patricia_trie_ref
        self.prefix: bytes = prefix
        self.trie_iter: Generator["PATRICIATrie", None, None] = self._traverse_up(self.patricia_trie_ref.descend(self.prefix))

    def __iter__(self):
        return self

    def __next__(self) -> Generator["PATRICIATrie", None, None]:
        return next(self.trie_iter)

    def _traverse_up(self, node: "PATRICIATrie") -> Generator["PATRICIATrie", None, None]:
        if node.is_root:
            return
        elif node.is_leaf:
            yield node
        for child in node.parent.children.values():
            if child != node:
                yield from self._post_order_traversal(child)
        yield from self._traverse_up(node.parent)

    def _post_order_traversal(self, node: "PATRICIATrie") -> Generator["PATRICIATrie", None, None]:
        if node.is_leaf:
            yield node
        else:
            for child in node.children.values():
                yield from self._post_order_traversal(child)


class PATRICIATrie:
    def __init__(self):
        self.children: Dict[bytes, PATRICIATrie] = dict()
        self.parent: PATRICIATrie = None
        self.value: PATRICIATrie = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

    # Returns the node with the longest prefix match
    def descend(self, key: bytes) -> "PATRICIATrie":
        found, prefix_index = self._prefix_match(key)
        if found is None:
            # No prefix found, return a randomly chosen leaf
            curr = self
            while not curr.is_leaf:
                curr = next(iter(curr.children.values()))
            return curr
        elif prefix_index == len(key):
            # Leaf found
            return self.children[found]
        else:
            # key[:prefix_index] is a prefix of found
            return self.children[found].descend(key[prefix_index:])

    # Gets the value for a key if it exists
    def __getitem__(self, key: bytes) -> Any:
        try:
            return self.get(key)
        except KeyError:
            raise KeyError(f'Key "{key}" not found')

    # Sets the value for a key
    def __setitem__(self, key: bytes, value: Any) -> None:
        self.insert(key, value)

    def __repr__(self, ident: int = 0) -> str:
        ret = ""
        for key in self.children.keys():
            curr = self.children[key]
            ret += " " * ident + str(key)
            if curr.is_leaf:
                ret += f" ({curr.value})"
            ret += "\n" + curr.__repr__(ident + 4)
        return ret

    def get(self, key: bytes) -> Any:
        found, prefix_index = self._prefix_match(key)
        if prefix_index == len(found):
            # found is a prefix of key
            if prefix_index == len(key):
                # found equals key
                return self.children[found].value
            else:
                # key[:prefix_index] is a prefix of found
                return self.children[found].get(key[prefix_index:])
        else:
            # No key prefix found
            raise KeyError

    def insert(self, key: bytes, value: Any) -> None:
        found, prefix_index = self._prefix_match(key)
        if found is None:
            # No key prefix found, must create
            t = PATRICIATrie()
            t.parent = self
            t.value = value
            self.children[key] = t
        elif prefix_index == len(found):
            # found is a prefix of key, must extend
            if prefix_index == len(key):
                # found equals key
                self.children[found].value = value
            else:
                # Continue inserting the suffix
                self.children[found].insert(key[prefix_index:], value)
        else:
            # found[:prefix_index] is a prefix of key, must split
            t = PATRICIATrie()
            t.children[found[prefix_index:]] = self.children.pop(found)
            t.children[found[prefix_index:]].parent = t
            self.children[found[:prefix_index]] = t
            self.children[found[:prefix_index]].parent = self
            if prefix_index == len(key):
                # Reached the insertion node
                t.value = value
            else:
                # Continue inserting the suffix
                t.insert(key[prefix_index:], value)

    # Returns a key and its prefix index for key
    def _prefix_match(self, key: bytes):
        def get_prefix_index_np(b1: bytes, b2: bytes) -> int:
            max_len = min(len(b1), len(b2))
            if b1[:max_len] == b2[:max_len]:
                return max_len
            else:
                arr1 = np.frombuffer(b1, dtype=bool, count=max_len)
                arr2 = np.frombuffer(b2, dtype=bool, count=max_len)
                return np.where(arr1 != arr2)[0][0]

        for _key in self.children:
            prefix_index = get_prefix_index_np(_key, key)
            if prefix_index:
                return _key, prefix_index
        return None, None

    # Returns an iterator that yields each leaf sorted by the lenght of the prefix match
    def get_prefix_iter(self, prefix: bytes) -> PrefixIter:
        return PrefixIter(self, prefix)
