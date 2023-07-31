from typing import Dict, Tuple


def get_prefix_index(b1: bytes, b2: bytes) -> int:
    max_len = min(len(b1), len(b2))
    for i in range(max_len):
        if b1[i] != b2[i]:
            return i
    return max_len


class PATRICIATrie:
    def __init__(self):
        self._storage: Dict[bytes, PATRICIATrie] = dict()
        self.is_leaf: bool = False

    def insert(self, b: bytes):
        key, prefix_index = self._find_prefix_key(b)
        if prefix_index == len(b):
            # Key already inserted
            return
        elif key is None:
            # No key is prefix of b (create)
            t = PATRICIATrie()
            t.is_leaf = True
            self._storage[b] = t
        elif prefix_index == len(key):
            # A key is a complete prefix of b
            if len(self._storage) == 1:
                # Extend key if it is the only entry
                self._storage[b] = self._storage[key]
                del self._storage[key]
            else:
                # Recurse
                self._storage[key].is_leaf = False
                self._storage[key].insert(b[len(key) :])
        else:
            # A key is a partial prefix of b (split)
            t = PATRICIATrie()
            t._storage[key[prefix_index:]] = self._storage.pop(key)
            self._storage[key[:prefix_index]] = t
            t.insert(b[prefix_index:])

    def _find_prefix_key(self, b: bytes) -> Tuple[bytes, int]:
        for key in self._storage:
            prefix_index = get_prefix_index(b, key)
            if prefix_index > 0:
                return key, prefix_index
        return None, None

    def __repr__(self, ident: int = 0) -> str:
        ret = ""
        for key in self._storage.keys():
            curr = self._storage[key]
            ret += " " * ident + str(key)
            if curr.is_leaf:
                ret += "*"
            ret += "\n" + curr.__repr__(ident + 4)
        return ret


pt = PATRICIATrie()
pt.insert(b"\x00\x00\x00")
pt.insert(b"\x00\x00\x01")
pt.insert(b"\x01\x00")
print(pt)