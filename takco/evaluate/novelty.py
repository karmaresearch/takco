import logging as log
import time
from rdflib import URIRef, Literal
from collections import defaultdict


def get_cell_noveltyhashes(triples, kb):
    kind_novelty_hashes = defaultdict(lambda: defaultdict(set))
    for t in triples:
        novelty_hashes = kind_novelty_hashes[t.get("kind")]

        s, p, v = t.get("s"), t.get("p"), t.get("o")
        triplehash = t.get("hash")

        # Track correctness of triple hashes
        if t.get("gold"):
            novelty_hashes["gold"].add(triplehash)
        if t.get("pred"):
            novelty_hashes["pred"].add(triplehash)

        os = set(o for _, _, o in kb.triples([URIRef(s), URIRef(p), None]))
        if not os:
            novelty_hashes["attnovel"].add(triplehash)
        else:
            if isinstance(v, dict):
                val = v.get("lexical_or_value")
                if val:
                    # TODO: datatype match?
                    literal_match = lambda: any(
                        m
                        for o in os
                        for m in kb.cellType.literal_match(o, val, kb.stringmatch)
                    )
                    label_match = lambda: any(
                        m for o in os for m in kb.label_match(o, val)
                    )
                    if literal_match() or label_match():
                        novelty_hashes["existing"].add(triplehash)
                    else:
                        novelty_hashes["valnovel"].add(triplehash)
            elif URIRef(v) not in os:
                novelty_hashes["valnovel"].add(triplehash)
            else:
                novelty_hashes["existing"].add(triplehash)

    for k, nhs in kind_novelty_hashes.items():
        kind_novelty_hashes[k] = {n: list(hs) for n, hs in nhs.items()}

    return kind_novelty_hashes


def count_noveltyhashes(kind_novelty_hashes):

    kind_counts = {}

    # Count intersections of correct with others
    for kind, nhs in kind_novelty_hashes.items():
        counts = kind_counts.setdefault(kind, {})
        for n, hs in nhs.items():
            hs = set(hs)
            if n not in ["gold", "pred"]:
                g, p = set(nhs.get("gold", [])), set(nhs.get("pred", []))
                counts[f"tp_{n}"] = len(hs & g & p)
                counts[f"fn_{n}"] = len(hs & g - p)
                counts[f"fp_{n}"] = len(hs & p - g)

        for n in ["attnovel", "valnovel"]:
            try:
                tp = counts[f"tp_{n}"]
                fp = counts[f"fp_{n}"]
                fn = counts[f"fn_{n}"]
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * ((p * r) / (p + r))
                counts.update(
                    {
                        f"{n}_precision": p,
                        f"{n}_recall": r,
                        f"{n}_f1": f1,
                    }
                )
            except:
                pass

    return kind_counts
