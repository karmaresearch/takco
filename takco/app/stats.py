import rdflib
from rdflib import URIRef, Literal


def get_kbgold(kb, table):
    kblinks = get_table_kblinks(kb, table)
    kblinks["novelty"] = get_novelty(kb, table, kblinks)
    kbinfo = get_kbinfo(kb, table)
    return kblinks, kbinfo


def get_kbinfo(kb, table):
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

    for e in set(
        e for row_es in table["entities"].values() for es in row_es.values() for e in es
    ):
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
    return kbinfo


def get_table_kblinks(kb, table):
    kblinks = {}
    kblinks["entity_hasclass"] = {}
    kblinks["entity_matchclass"] = {}
    if set([c for cs in table["classes"].values() for c in cs]):
        kblinks["entity_hasclass"] = {
            str(e): any(
                list(kb.triples([URIRef(e), tp, None])) for tp in kb.typeProperties
            )
            for eci, row_es in table["entities"].items()
            for es in row_es.values()
            for e in es
        }

        kblinks["entity_matchclass"] = {
            str(e): any(
                list(kb.triples((URIRef(e), tp, URIRef(c))))
                for tp in kb.typeProperties
                for cci, cs in table["classes"].items()
                for c in cs
                if eci == cci  # entity col == class col
            )
            for eci, row_es in table["entities"].items()
            for es in row_es.values()
            for e in es
        }

    kblinks["entity_prop_exists"] = {
        str(e): {
            str(p): list(kb.triples([URIRef(e), URIRef(p), None]))
            for fromci, cps in table["properties"].items()
            for ps in cps.values()
            for p in ps
            if fromci == eci  # entity col == prop from col
        }
        for eci, row_es in table["entities"].items()
        for es in row_es.values()
        for e in es
    }

    kblinks["rownr_colnr_matches"] = {}
    for rownr, row in enumerate(table["rows"]):
        for fromci, row_es in table["entities"].items():
            for e in row_es.get(str(rownr), []):
                colnr_matches = {}
                for toci, cell in enumerate(row):
                    for p in (
                        table["properties"].get(str(fromci), {}).get(str(toci), [])
                    ):
                        toents = (
                            table["entities"].get(str(toci), {}).get(str(rownr), {})
                        )
                        if toents:
                            for o in toents:
                                t = (URIRef(e), URIRef(p), URIRef(o))
                                for _ in kb.triples(t):
                                    colnr_matches[str(toci)] = {
                                        "score": 1,
                                    }
                        else:
                            score, literal, dtype = kb.match(URIRef(e), URIRef(p), cell)
                            if score:
                                colnr_matches[str(toci)] = {
                                    "score": score,
                                    "lit": literal,
                                    "dtype": dtype,
                                }
                kblinks["rownr_colnr_matches"][str(rownr)] = colnr_matches

    return kblinks


def pct(a, b):
    return a / b if b else 0


def novelty_add_pct(novelty):
    novelty.update(
        {
            "lbl_pct": pct(novelty["lbl"], novelty["lbl_total"]),
            "cls_pct": pct(novelty["cls"], novelty["cls_total"]),
            "prop_pct": pct(novelty["prop"], novelty["prop_total"]),
        }
    )
    novelty.update(
        {
            "lbl_val_pct": pct(novelty["lbl_nomatch"], novelty["lbl_total"]),
            "cls_val_pct": pct(novelty["cls_nomatch"], novelty["cls_total"]),
            "prop_val_pct": pct(novelty["prop_nomatch"], novelty["prop_total"]),
        }
    )


def get_novelty(kb, table, kblinks):
    lbl_ci = None  # TODO: multiple label columns?
    for cps in table["properties"].values():
        for colnr, ps in cps.items():
            for p in ps:
                if URIRef(p) in kb.labelProperties:
                    lbl_ci = colnr

    novelty = {
        "lbl": sum(
            (not any(ppe.get(str(lp)) for lp in kb.labelProperties))
            for e, ppe in kblinks["entity_prop_exists"].items()
        ),
        "lbl_nomatch": sum(
            not bool(cm.get(str(lbl_ci), {}))
            for cm in kblinks["rownr_colnr_matches"].values()
        ),
        "lbl_total": sum(1 for e, ppe in kblinks["entity_prop_exists"].items()),
        "cls": sum((not hc) for e, hc in kblinks["entity_hasclass"].items()),
        "cls_nomatch": sum((not hc) for e, hc in kblinks["entity_matchclass"].items()),
        "cls_total": len(kblinks["entity_hasclass"]),
        "prop": sum(
            (not pe)
            for e, ppe in kblinks["entity_prop_exists"].items()
            for p, pe in ppe.items()
            if URIRef(p) not in kb.labelProperties
        ),
        "prop_nomatch": sum(
            not cm.get(str(toci))
            for cm in kblinks["rownr_colnr_matches"].values()
            for fromci, toci_ps in table["properties"].items()
            for toci, ps in toci_ps.items()
            if toci != lbl_ci
        ),
        "prop_total": sum(
            1
            for e, ppe in kblinks["entity_prop_exists"].items()
            for fromci, toci_ps in table["properties"].items()
            for toci, ps in toci_ps.items()
            for p in ps
            if URIRef(p) not in kb.labelProperties
        ),
    }
    novelty["lbl_nomatch"] = novelty["lbl_nomatch"] - novelty["lbl"]
    novelty["cls_nomatch"] = novelty["cls_nomatch"] - novelty["cls"]
    novelty["prop_nomatch"] = novelty["prop_nomatch"] - novelty["prop"]

    novelty.update(
        {
            "lbl_redundant": novelty["lbl_total"]
            - novelty["lbl_nomatch"]
            - novelty["lbl"],
            "cls_redundant": novelty["cls_total"]
            - novelty["cls_nomatch"]
            - novelty["cls"],
            "prop_redundant": novelty["prop_total"]
            - novelty["prop_nomatch"]
            - novelty["prop"],
        }
    )
    novelty_add_pct(novelty)

    return novelty
