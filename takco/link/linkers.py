from typing import Container, List, Dict
import logging as log

from .base import Linker, Searcher, SearchResult
from .rdf import GraphDB

URI = str


class First(Linker):
    """Returns the first match for entity linking, under possible constraints

    Args:
        searcher: The Searcher to find KB entities
        limit: Number of candidates to search for
        majority_class: Only use entities that have the majority value for this property
        exclude_about: Exclude entities that have these property-values

    """

    def __init__(
        self,
        searcher: Searcher,
        limit: int = 1,
        contextual: bool = False,
        search_limit: int = 10,
        majority_class: URI = None,
        exclude_about: Dict[URI, URI] = None,
        normalize: bool = False,
        majority_class_search: bool = False,
    ):
        self.searcher = searcher
        self.limit = limit
        self.contextual = contextual
        self.search_limit = search_limit
        self.majority_class = majority_class
        self.majority_class_search = bool(majority_class_search)
        self.exclude_about = exclude_about
        self.add_about = bool(majority_class or exclude_about)
        self.normalize = normalize

    def __enter__(self):
        self.searcher.__enter__()
        return self

    def __exit__(self, *args):
        self.searcher.__exit__(*args)

    def link(
        self,
        rows: List[List[str]],
        usecols: Container[int] = None,
        skiprows: Container[int] = None,
        existing: Dict = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:

        existing_entities = (existing or {}).get("entities", {})
        existing_entities = {
            int(ci): {int(ri): es for ri, es in res.items()}
            for ci, res in existing_entities.items()
        }
        existing_classes = (existing or {}).get("classes", {})
        existing_classes = {int(ci): cs for ci, cs in existing_classes.items()}

        def get_rowcol_searchresults(col_classes):
            return self.get_rowcol_searchresults(
                rows,
                limit=max(self.search_limit, self.limit),
                contextual=self.contextual,
                usecols=usecols,
                skiprows=skiprows,
                existing_entities=existing_entities,
                add_about=self.add_about,
                col_classes=col_classes,
            )

        rowcol_searchresults = get_rowcol_searchresults(existing_classes)

        # TODO: lookup facts about existing entities

        if self.majority_class:
            # Keep track of the most frequent attribute of a certain predicate
            ci_att_count = {}
            for (_, ci), results in rowcol_searchresults.items():
                for result in results[:1]:
                    for att in result.get(self.majority_class, []):
                        ci_att_count.setdefault(ci, {}).setdefault(att, 0)
                        ci_att_count[ci][att] += 1

            ci_majoratt = {}
            ci_classes = existing_classes
            for ci, att_count in ci_att_count.items():
                ci_majoratt[ci] = max(att_count, key=att_count.get)
                ci_classes[ci] = {ci_majoratt[ci]: att_count[ci_majoratt[ci]]}

            if self.majority_class_search:
                log.debug(f"Re-searching {self} with classes {ci_classes}")
                rowcol_searchresults = get_rowcol_searchresults(ci_classes)
            else:
                log.debug(f"Filtering {self} with classes {ci_classes}")
                atts = lambda r: r.get(self.majority_class, [])
                for (ri, ci), results in rowcol_searchresults.items():
                    rowcol_searchresults[(ri, ci)] = [
                        r for r in results if ci_majoratt.get(ci) in atts(r)
                    ]

        if self.exclude_about:
            for k, results in rowcol_searchresults.items():
                for p, os in self.exclude_about.items():

                    def isbad(r):
                        return any(o in r.get(p, []) for o in os)

                    results = [r for r in results if not isbad(r)]
                rowcol_searchresults[k] = results

        entities = existing_entities
        for (ri, ci), results in rowcol_searchresults.items():
            results = results[: self.limit]
            if self.normalize:
                total_score = sum(r.score for r in results)
                results = [
                    SearchResult(r.uri, r, r.score / total_score) for r in results
                ]

            for r in results:
                ents = entities.setdefault(str(ci), {}).setdefault(str(ri), {})
                ents[r.uri] = r.score

        if self.majority_class:
            classes = {str(ci): c for ci, c in ci_classes.items()}
            return {"entities": entities, "classes": classes}
        else:
            return {"entities": entities}


class Salient(Linker):
    """Filters on the most salient class and property per column.

    When ``expand`` is set, expands the candidate entity set with values of salient
    properties, if they match according to
    :meth:`~takco.link.base.Searcher.label_match`. Salience is defined as follows:

    .. math::

        \\frac{\\text{number of matches}}{\\text{count}}

    Args:
        searcher: The Searcher to find KB entities
        limit: Number of candidates to search for
        replace_class: Map from bad class URIs to good ones
        class_cover: Minimum class cover fraction
        prop_cover: Minimum property cover fraction
        expand: Whether to expand candidates with values of salient properties
        graph: Graph to use when expanding, if ``searcher`` is not a Graph
        max_backlink: When expanding, only use properties that have fewer backlinks

    """

    def __init__(
        self,
        searcher: Searcher,
        limit: int = 10,
        contextual: bool = False,
        replace_class: Dict[URI, URI] = None,
        class_cover: float = 0.5,
        prop_cover: float = 0.5,
        expand: bool = False,
        graph: GraphDB = None,
        max_backlink: int = 100,
    ):
        self.searcher = searcher
        self.limit = limit
        self.replace_class = replace_class or {}
        self.class_cover = class_cover
        self.prop_cover = prop_cover
        self.contextual = contextual

        self.expand = expand
        if (graph is None) and isinstance(searcher, GraphDB):
            self.graph = searcher
        else:
            self.graph = graph
        self.max_backlink = max_backlink

    def __enter__(self):
        self.searcher.__enter__()
        if self.graph != self.searcher:
            self.graph.__enter__()
        return self

    def __exit__(self, *args):
        self.searcher.__exit__(*args)
        if self.graph != self.searcher:
            self.graph.__exit__(*args)

    def link(
        self,
        rows: List[List[str]],
        usecols: Container[int] = None,
        skiprows: Container[int] = None,
        existing: Dict = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:

        existing_entities = (existing or {}).get("entities", {})
        rowcol_searchresults = self.get_rowcol_searchresults(
            rows,
            limit=self.limit,
            contextual=self.contextual,
            usecols=usecols,
            skiprows=skiprows,
            existing_entities=existing_entities,
            add_about=True,
        )
        # TODO: lookup facts about existing entities

        ci_ri_searchresults = {}
        for (ri, ci), results in rowcol_searchresults.items():
            results = sorted(results, key=len)[::-1][: self.limit]
            ci_ri_searchresults.setdefault(ci, {})[ri] = results

        from collections import Counter, defaultdict

        # Most salient property per column pair
        fromci_toci_props = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for toci, tocol in enumerate(zip(*rows)):

            ri_toresults = ci_ri_searchresults.get(toci, {})
            for fromci, fromri_searchresults in list(ci_ri_searchresults.items()):
                if fromci == toci:
                    continue

                prop_count = Counter()
                for ri, celltext in enumerate(tocol):
                    for fr in fromri_searchresults.get(ri, {}):
                        # Loop over 'from' attributes
                        for p, os in fr.items():

                            # Check for overlap with the 'to' entities
                            cellresults = ri_toresults.get(ri, {})
                            touris = [tr.uri for tr in cellresults]
                            if set(map(str, os)) & set(touris):
                                prop_count[p] += 1

                            if not cellresults:
                                # Check for overlap with literals
                                for o in os:
                                    matches = self.graph.cellType.literal_match(
                                        o, celltext
                                    )
                                    mscore = sum(1 for m in matches)
                                    if mscore:
                                        prop_count[p] += mscore
                if prop_count:
                    log.debug(f"Prop count for col {fromci}->{toci}: {prop_count}")

                ntotal = len(ri_toresults)
                from rdflib import URIRef

                prop_salience = {
                    p: n / (self.graph.count((None, URIRef(p), None)) + 1)
                    for p, n in prop_count.items()
                    if n >= ntotal * self.prop_cover
                }
                if prop_salience:
                    log.debug(f"Salience for col {fromci}->{toci}: {prop_salience}")

                for p, s in Counter(prop_salience).most_common(1):
                    fromci_toci_props[str(fromci)][str(toci)][p] = s

                    # Add matching values from salient properties to candidates
                    if not self.expand:
                        continue
                    for ri, row in enumerate(rows):
                        if not ci_ri_searchresults.get(toci, {}).get(ri):
                            rs = []
                            celltext = row[toci]
                            for fr in fromri_searchresults.get(ri, {}):
                                for o in fr.get(p, []):
                                    matches = self.graph.label_match(o, celltext)
                                    for m in matches:
                                        log.debug(f"Matched {o} to {celltext}")
                                        rs.append(SearchResult(str(o), {}, m.score))
                            rs = sorted(rs, key=lambda m: -m.score)
                            ci_ri_searchresults.setdefault(toci, {})[ri] = rs

                        # add backward links if there's a graph
                        if self.graph is not None:
                            from rdflib import URIRef

                            if not ci_ri_searchresults.get(fromci, {}).get(ri):
                                rs = []
                                celltext = row[fromci]
                                for tr in ci_ri_searchresults.get(toci, {}).get(ri, {}):
                                    o = URIRef(tr.uri)
                                    if (
                                        self.graph.count([None, URIRef(p), o])
                                        > self.max_backlink
                                    ):
                                        continue
                                    for s, _, _ in self.graph.triples([None, p, o]):
                                        matches = self.graph.label_match(s, celltext)
                                        for m in matches:
                                            log.debug(f"Matched back {s} to {celltext}")
                                            rs.append(SearchResult(str(s), {}, m.score))
                                rs = sorted(rs, key=lambda m: -m.score)
                                ci_ri_searchresults.setdefault(fromci, {})[ri] = rs

        # Select only candidate entities that are salient properties
        for ci, ri_searchresults in ci_ri_searchresults.items():
            for ri, results in ri_searchresults.items():
                for r in results:

                    ok = not fromci_toci_props
                    # Check if r.uri is the salient-prop object of another cell
                    for fromci, toci_props in fromci_toci_props.items():
                        for prop in toci_props.get(str(ci), {}):
                            ros = ci_ri_searchresults.get(int(fromci), {}).get(
                                int(ri), {}
                            )
                            other_vals = set(
                                str(v) for ro in ros for v in ro.get(prop, [])
                            )
                            # Check if any salient-prop val of r matches other cell
                            if r.uri in other_vals:
                                ok = True

                    # Check if another cell is the salient-prop of r
                    for toci, props in fromci_toci_props.get(str(ci), {}).items():
                        ros = ci_ri_searchresults.get(int(toci), {}).get(int(ri), {})
                        vals = set(str(v) for p in props for v in r.get(p, []))
                        if any(str(ro.uri) in vals for ro in ros):
                            ok = True

                    if ok:
                        ri_searchresults[ri] = [r]
                        break

        # Most salient class per column
        ci_classes = {}
        for ci, ri_searchresults in ci_ri_searchresults.items():
            ent_result = {r.uri: r for rs in ri_searchresults.values() for r in rs}
            ntotal = len(ent_result)

            cls_count = Counter()
            for uri, r in ent_result.items():
                for t in self.graph.typeProperties:
                    for cls in r.get(t, []):
                        cls_count[cls] += 1
            cls_salience = {
                cls: n / (self.graph.count((None, None, cls)) + 1)
                for cls, n in cls_count.items()
                if n >= ntotal * self.class_cover
            }
            for cls, s in Counter(cls_salience).most_common(1):
                cls = self.replace_class.get(str(cls), cls)
                ci_classes[str(ci)] = {str(cls): s}

        ci_ri_ents = existing_entities
        for ci, ri_searchresults in ci_ri_searchresults.items():
            for ri, results in ri_searchresults.items():
                for r in results:
                    ci_ri_ents.setdefault(str(ci), {})[str(ri)] = {r.uri: 1}
                    break

        return {
            "entities": ci_ri_ents,
            "properties": fromci_toci_props,
            "classes": ci_classes,
        }
