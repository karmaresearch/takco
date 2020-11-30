import logging as log
import time
from rdflib import URIRef, Literal
from collections import defaultdict
import urllib


def get_cell_noveltyhashes(triples, kb):

    log.warn(kb)


    kind_novelty_hashes = defaultdict(lambda: defaultdict(set))
    for t in triples:
        novelty_hashes = kind_novelty_hashes[t.get("kind")]

        s, p, v = t.get("s"), t.get("p"), t.get("o")

        spr = urllib.parse.urlparse(s)
        s = spr._replace(path=urllib.parse.quote(spr.path)).geturl()

        ppr = urllib.parse.urlparse(p)
        p = ppr._replace(path=urllib.parse.quote(ppr.path)).geturl()

        triplehash = t.get("hash")

        # Track correctness of triple hashes
        if t.get("gold"):
            novelty_hashes["gold"].add(triplehash)
        if t.get("pred"):
            novelty_hashes["pred"].add(triplehash)

        os = kb.get_prop_values(s,p)
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
                        for m in kb.typer.literal_match(o, val, kb.stringmatch)
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
                counts[n] = {
                    "tp": len(hs & g & p),
                    "fn": len(hs & g - p),
                    "fp": len(hs & p - g),
                }

        for n in counts:
            try:
                tp = counts[n]["tp"]
                fp = counts[n]["fp"]
                fn = counts[n]["fn"]
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * ((p * r) / (p + r))
                counts[n].update(
                    {"precision": p, "recall": r, "f1": f1,}
                )
            except:
                pass

    return kind_counts
