from typing import Any, Dict, Generator


class PrefixIter:
    def __init__(self, patricia_trie_ref: "PATRICIATrie", prefix: str):
        self.patricia_trie_ref = patricia_trie_ref
        self.prefix = prefix
        self.trie_iter = self._traverse_up(patricia_trie_ref.descend(prefix))

    def __iter__(self):
        return self

    def __next__(self) -> Generator["PATRICIATrie", None, None]:
        return next(self.trie_iter)

    def _traverse_up(self, node: "PATRICIATrie") -> Generator["PATRICIATrie", None, None]:
        if node.is_root:
            return
        elif node.is_leaf:
            yield node
        for child in node.parent._storage.values():
            if child != node:
                yield from self._post_order_traversal(child)
        yield from self._traverse_up(node.parent)

    def _post_order_traversal(self, node: "PATRICIATrie") -> Generator["PATRICIATrie", None, None]:
        if node.is_leaf:
            yield node
        else:
            for child in node._storage.values():
                yield from self._post_order_traversal(child)


class PATRICIATrie:
    def __init__(self):
        self._storage: Dict[str, PATRICIATrie] = dict()
        self.parent = None
        self.value = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

    def clear(self) -> None:
        self._storage = dict()

    # Returns the node with the longest prefix match
    def descend(self, key: str) -> "PATRICIATrie":
        found, prefix_index = self._prefix_match(key)
        if found is None:
            # No prefix found, return a randomly chosen leaf
            curr = self
            while not curr.is_leaf:
                curr = next(iter(curr._storage.values()))
            return curr
        elif prefix_index == len(key):
            # Leaf found
            return self._storage[found]
        else:
            # key[:prefix_index] is a prefix of found
            return self._storage[found].descend(key[prefix_index:])

    # Gets the value for a key if it exists
    def __getitem__(self, key: str) -> Any:
        try:
            return self._get(key)
        except KeyError:
            raise KeyError(f'Key "{key}" not found')

    # Sets the value for a key
    def __setitem__(self, key: str, value: Any) -> None:
        self._insert(key, value)

    def __repr__(self, ident: int = 0) -> str:
        ret = ""
        for key in self._storage.keys():
            curr = self._storage[key]
            ret += " " * ident + str(key)
            if curr.is_leaf:
                ret += f" ({curr.value})"
            ret += "\n" + curr.__repr__(ident + 4)
        return ret

    def _get(self, key: str) -> Any:
        found, prefix_index = self._prefix_match(key)
        if prefix_index == len(found):
            # found is a prefix of key
            if prefix_index == len(key):
                # found equals key
                return self._storage[found].value
            else:
                # key[:prefix_index] is a prefix of found
                return self._storage[found]._get(key[prefix_index:])
        else:
            # No key prefix found
            raise KeyError

    def _insert(self, key: str, value: Any) -> None:
        found, prefix_index = self._prefix_match(key)
        if found is None:
            # No key prefix found, must create
            t = PATRICIATrie()
            t.parent = self
            t.value = value
            self._storage[key] = t
        elif prefix_index == len(found):
            # found is a prefix of key, must extend
            if prefix_index == len(key):
                # found equals key
                self._storage[found].value = value
            else:
                # Continue inserting the suffix
                self._storage[found]._insert(key[prefix_index:], value)
        else:
            # found[:prefix_index] is a prefix of key, must split
            t = PATRICIATrie()
            t._storage[found[prefix_index:]] = self._storage.pop(found)
            t._storage[found[prefix_index:]].parent = t
            self._storage[found[:prefix_index]] = t
            self._storage[found[:prefix_index]].parent = self
            if prefix_index == len(key):
                # Reached the insertion node
                t.value = value
            else:
                # Continue inserting the suffix
                t._insert(key[prefix_index:], value)

    # Returns a key and its prefix index for key
    def _prefix_match(self, key: str):
        def get_prefix_index(s1: str, s2: str) -> int:
            max_len = min(len(s1), len(s2))
            for i in range(max_len):
                if s1[i] != s2[i]:
                    return i
            return max_len

        for _key in self._storage:
            prefix_index = get_prefix_index(_key, key)
            if prefix_index:
                return _key, prefix_index
        return None, None

    # Returns an iterator that yields each leaf sorted by the lenght of the prefix match
    def get_prefix_iter(self, prefix: str) -> PrefixIter:
        return PrefixIter(self, prefix)
