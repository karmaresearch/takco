from rdflib import URIRef, Literal


TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def triple_hash(s, p, v):
    vnode = Literal(**v) if isinstance(v, dict) else URIRef(v)
    return hash((URIRef(s), URIRef(p), vnode))


def yield_triples(table, include_type=True, include_label=True):
    gold = table.get("gold", {})

    # if fields are set but empty, the table is annotated but has no correct prediction
    hasgold = bool(gold)

    if "rows" in table:
        rows = table["rows"]
    else:
        rows = [[c.get("text", "") for c in r] for r in table.get("tableData", [])]

    # Yield gold and predicted triples
    if include_label:
        for ri, cells in enumerate(rows):
            for ci, cell in enumerate(cells):
                predents = table.get("entities", {}).get(str(ci), {}).get(str(ri), {})
                goldents = gold.get("entities", {}).get(str(ci), {}).get(str(ri), {})
                for e, escore in list(predents.items()) + list(goldents.items()):
                    v = {"lexical_or_value": str(cell)}
                    isgold = e in goldents
                    ispred = e in predents
                    yield {
                        "kind": "label",
                        "s": str(e),
                        "p": str(LABEL),
                        "o": v,
                        "s_score": escore,
                        "col": int(ci),
                        "row": int(ri),
                        "gold": isgold if hasgold else None,
                        "pred": ispred,
                        "hash": triple_hash(e, LABEL, v),
                    }

    if include_type:
        for ci, predclss in table.get("classes", {}).items():
            goldclss = gold.get("classes", {}).get(ci, {})
            for cls, cls_score in list(predclss.items()) + list(goldclss.items()):
                for ri, predents in table.get("entities", {}).get(ci, {}).items():
                    goldents = gold.get("entities", {}).get(ci, {}).get(ri, {})

                    for e, escore in list(predents.items()) + list(goldents.items()):
                        isgold = (e in goldents) and (cls in goldclss)
                        ispred = (e in predents) and (cls in predclss)
                        if isgold or ispred:
                            yield {
                                "kind": "class",
                                "s": str(e),
                                "p": str(TYPE),
                                "o": str(cls),
                                "s_score": escore,
                                "o_score": cls_score,
                                "col": int(ci),
                                "row": int(ri),
                                "gold": isgold if hasgold else None,
                                "pred": ispred,
                                "hash": triple_hash(e, TYPE, cls),
                            }

    # TODO: structure qualifiers
    for fromci, toci_props in table.get("properties", {}).items():
        for toci, predprops in toci_props.items():
            goldprops = gold.get("properties", {}).get(fromci, {}).get(toci, {})

            ri_toents = table.get("entities", {}).get(toci, {})
            isentityprop = bool(ri_toents)

            for p, p_score in list(predprops.items()) + list(goldprops.items()):
                for ri, predents in table.get("entities", {}).get(fromci, {}).items():
                    goldents = gold.get("entities", {}).get(fromci, {}).get(ri, {})

                    for s, s_score in list(predents.items()) + list(goldents.items()):
                        isgold = (s in goldents) and (p in goldprops)
                        ispred = (s in predents) and (p in predprops)
                        if isgold or ispred:
                            sp = {
                                "kind": "property",
                                "s": str(s),
                                "p": str(p),
                                "s_score": s_score,
                                "p_score": p_score,
                                "col": int(toci),
                                "row": int(ri),
                                "gold": isgold if hasgold else None,
                                "pred": ispred,
                            }

                            if isentityprop:
                                for o, o_score in ri_toents.get(ri, {}).items():
                                    yield {
                                        **sp,
                                        "o": str(o),
                                        "o_score": o_score,
                                        "hash": triple_hash(s, p, o),
                                    }
                            else:
                                try:
                                    cell = rows[int(ri)][int(toci)]
                                except:
                                    cell = ""
                                if cell:
                                    tocls = table.get("classes", {}).get(toci, {})
                                    if tocls:
                                        for cls, cls_score in tocls.items():
                                            v = {
                                                "lexical_or_value": str(cell),
                                                "datatype": str(cls),
                                            }
                                            yield {
                                                **sp,
                                                "o": v,
                                                "o_score": cls_score,
                                                "hash": triple_hash(s, p, v),
                                            }
                                    else:
                                        v = {"lexical_or_value": str(cell)}
                                        yield {
                                            **sp,
                                            "o": v,
                                            "hash": triple_hash(s, p, v),
                                        }
