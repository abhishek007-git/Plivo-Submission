import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_BPE_PATH = os.path.join(_HERE, "bpe.json")
_PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        self.token_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache = {}

    def _encode_chunk(self, b):
        cached = self._cache.get(b)
        if cached is not None:
            return cached
        ids = list(b)
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i:best_i + 2] = [256 + best_rank]
        self._cache[b] = ids
        return ids

    def encode(self, text):
        out = []
        for m in _PAT.findall(text):
            out.extend(self._encode_chunk(m.encode("utf-8")))
        return out

    def decode(self, ids):
        buf = b"".join(self.token_bytes[i] for i in ids)
        return buf.decode("utf-8", errors="replace")

    def save(self, path=None):
        path = path or _BPE_PATH
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": [list(m) for m in self.merges]}, f)


def load(path=None):
    p = path or _BPE_PATH
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        if d.get("type") == "bpe":
            return BPETokenizer(d["merges"])
    return BPETokenizer([]) 
