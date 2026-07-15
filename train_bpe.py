import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+", re.UNICODE)


def _pairs(sym):
    return [(sym[i], sym[i + 1]) for i in range(len(sym) - 1)]


def train(text, vocab_size):
    num_merges = vocab_size - 256
    wf = Counter(PAT.findall(text))
    words = [list(w.encode("utf-8")) for w in wf]
    freqs = list(wf.values())

    pair_counts = Counter()
    pair_words = defaultdict(set)
    for wi, sym in enumerate(words):
        for p in _pairs(sym):
            pair_counts[p] += freqs[wi]
            pair_words[p].add(wi)

    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=lambda k: (pair_counts[k], k))
        if pair_counts[best] <= 0:
            break
        new_id = 256 + len(merges)
        merges.append(best)
        a, b = best
        for wi in list(pair_words[best]):
            sym = words[wi]
            f = freqs[wi]
            for p in _pairs(sym):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            for p in set(_pairs(sym)):
                pair_words[p].discard(wi)
            new_sym = []
            i = 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    new_sym.append(new_id)
                    i += 2
                else:
                    new_sym.append(sym[i])
                    i += 1
            words[wi] = new_sym
            for p in _pairs(new_sym):
                pair_counts[p] += f
            for p in set(_pairs(new_sym)):
                pair_words[p].add(wi)
        pair_counts.pop(best, None)
        pair_words.pop(best, None)
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "bpe.json")

    text = open(args.data, encoding="utf-8").read()
    t0 = time.time()
    merges = train(text, args.vocab)
    with open(out, "w") as f:
        json.dump({"type": "bpe", "merges": [list(m) for m in merges]}, f)
    print(f"trained {len(merges)} merges (vocab {256 + len(merges)}) "
          f"in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
