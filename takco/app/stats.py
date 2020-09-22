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

    def get_links(e, p):
        if kb.count([URIRef(e), URIRef(p), None]) < 10:
            yield 1, (o for _, _, o in kb.triples([URIRef(e), URIRef(p), None]))
        else:
            yield 1, None

        if kb.count([None, URIRef(p), URIRef(e)]) < 10:
            yield -1, (s for s, _, _ in kb.triples([None, URIRef(p), URIRef(e)]))
        else:
            yield -1, None

    all_e = all_ents(table) | all_ents(table.get("gold", {}))

    all_p = set(kb.labelProperties) | set(kb.typeProperties) | set(props)
    all_p = sorted(str(p) for p in all_p)

    log.info(
        f"Getting data for {len(all_e)} entities and {len(all_p)} properties = {len(all_e) * len(all_p)}"
    )

    for e in all_e:
        kbinfo[str(e)] = {
            "uri": e,
            "props": [],
            "inlinks": kb.count((None, None, URIRef(e))),
            "outlinks": kb.count((URIRef(e), None, None)),
        }
        for p in all_p:
            vals = set(
                ("> " if d < 0 else "") + str(o)
                for d, os in get_links(e, p)
                for o in (os if os is not None else ["(many)"])
                if e and o
                if not hasattr(o, "language") or (o.language in [None, "en"])
            )
            if vals:
                kbinfo[str(e)]["props"].append({"uri": p, "vals": list(vals)})

    log.debug(f"KBinfo took {time.time() - start:.1f} seconds")
    return kbinfo
