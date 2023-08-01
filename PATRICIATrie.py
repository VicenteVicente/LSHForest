from typing import Any, Dict


class PATRICIATrie:
    def __init__(self):
        self._storage: Dict[str, PATRICIATrie] = dict()
        self.value = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

    def clear(self) -> None:
        self._storage = dict()

    def __getitem__(self, key: str) -> Any:
        try:
            return self._get(key)
        except KeyError:
            raise KeyError(f'Key "{key}" not found')

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
            self._storage[found[:prefix_index]] = t
            if prefix_index == len(key):
                # Reached the insertion node
                t.value = value
            else:
                # Continue inserting the suffix
                t._insert(key[prefix_index:], value)

    # Returns a key and its prefix index for b if any
    def _prefix_match(self, b: str):
        def get_prefix_index(b1: str, b2: str) -> int:
            max_len = min(len(b1), len(b2))
            for i in range(max_len):
                if b1[i] != b2[i]:
                    return i
            return max_len

        for key in self._storage:
            prefix_index = get_prefix_index(b, key)
            if prefix_index:
                return key, prefix_index
        return None, None


t = PATRICIATrie()
t["0000"] = 0
t["0010"] = 1
t["1000"] = 2
t["1010"] = 3
t["10"] = 4
t["0"] = 5
t["00001"] = 6
print(t)