import logging as log
import rdflib
from rdflib import URIRef, Literal
import time


def get_kbgold(kb, table):
    kblinks = {}
    kbinfo = get_kbinfo(kb, table)
    return kblinks, kbinfo


def get_kbinfo(kb, table):
    start = time.time()
    kbinfo = {}
    props = [
        p for cps in table["properties"].values() for ps in cps.values() for p in ps
    ]
    classes = [c for cs in table["classes"].values() for c in cs]
    for e in set(props) | set(classes):
        if e:
            kbinfo[str(e)] = {
                "uri": e,
                "props": [
                    {
                        "uri": lp,
                        "vals": [
                            str(o)
                            for (s, p, o) in kb.triples([URIRef(e), lp, None])
                            if not hasattr(o, "language")
                            or (o.language in [None, "en"])
                        ],
                    }
                    for lp in kb.labelProperties
                ],
            }
            if e in props:
                kbinfo[str(e)]["size"] = kb.count((None, URIRef(e), None))
            else:
                kbinfo[str(e)]["inlinks"] = kb.count((None, None, URIRef(e)))
                kbinfo[str(e)]["outlinks"] = kb.count((URIRef(e), None, None))

    def all_ents(table):
        ents = table.get("entities", {})
        return set(e for row_es in ents.values() for es in row_es.values() for e in es)

    for e in all_ents(table) | all_ents(table.get("gold", {})):
        all_p = set(kb.labelProperties) | set(kb.typeProperties) | set(props)
        kbinfo[str(e)] = {
            "uri": e,
            "props": [],
            "inlinks": kb.count((None, None, URIRef(e))),
            "outlinks": kb.count((URIRef(e), None, None)),
        }
        for p in sorted(set([str(p) for p in all_p])):
            vals = set(
                ("> " if d < 0 else "") + str(o)
                for d, spos in [
                    (1, kb.triples([URIRef(e), URIRef(p), None])),
                    (
                        -1,
                        [
                            (o, p, s)
                            for s, p, o in kb.triples([None, URIRef(p), URIRef(e)])
                        ],
                    ),
                ]
                for s, p, o in spos
                if e and o
                if not hasattr(o, "language") or (o.language in [None, "en"])
            )
            if vals:
                kbinfo[str(e)]["props"].append({"uri": p, "vals": list(vals)})

    log.debug(f"KBinfo took {time.time() - start:.1f} seconds")
    return kbinfo
