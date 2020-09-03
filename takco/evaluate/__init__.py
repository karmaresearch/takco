from . import score
from . import dataset

TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def triples(tables, include_type=True):
    """Make triples based on table predictions"""
    for table in tables:
        entities = table.get("entities", {})

        if include_type:
            for ci, clss in table.get("classes", {}).items():
                for cls, cls_score in clss.items():
                    for ri, ents in entities.get(ci, {}).items():
                        for e, escore in ents.items():
                            yield {
                                "s": str(e),
                                "p": str(TYPE),
                                "o": str(cls),
                                "s_score": escore,
                                "o_score": cls_score,
                            }

        # TODO: structure qualifiers
        # TODO: literals with datatypes

        for fromci, toci_props in table.get("properties", {}).items():
            for toci, props in toci_props.items():
                for p, p_score in props.items():
                    for ri, froments in entities.get(fromci, {}).items():
                        toents = entities.get(toci, {}).get(ri, {})

                        for s, s_score in froments.items():
                            for o, o_score in toents.items():
                                yield {
                                    "s": str(s),
                                    "p": str(p),
                                    "o": str(o),
                                    "s_score": s_score,
                                    "p_score": p_score,
                                    "o_score": o_score,
                                }
