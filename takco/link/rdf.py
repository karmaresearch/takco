import rdflib
import logging as log
from rdflib import URIRef, Literal
from typing import List, Dict, Tuple, Collection, Iterator

import datetime
import re

from .base import (
    Searcher,
    Database,
    SearchResult,
    Triple,
    QualifierMatchResult,
    MatchResult,
    LiteralMatchResult,
)


class GraphDB(Database, rdflib.Graph):
    def __init__(self, *args, **kwargs):
        rdflib.Graph.__init__(self, *args, **kwargs)

    def triples(self, triplepattern, **kwargs):
        return rdflib.Graph.triples(self, triplepattern, **kwargs)

    def count(self, triplepattern):
        if hasattr(self.store, "count"):
            return self.store.count(triplepattern)
        elif hasattr(self.store, "hdt_document"):
            _, n = self.store.hdt_document.search(triplepattern)
            return n
        else:
            ts = self.triples(triplepattern)
            return len(ts) if hasattr(t, "__len__") else sum(1 for _ in ts)

    def __len__(self):
        return self.store.__len__()

    def pages_about(self, triplepattern=None):
        if hasattr(self.store, "pages_about"):
            return self.store.pages_about(triplepattern)
        else:
            s_pages = {}
            for (s, _, _) in self.triples(triplepattern):
                s_pages.setdefault(s, []).append(str(s))
            return s_pages


class RDFSearcher(Searcher, GraphDB):
    """Entity Searcher model based on an RDF graph database.
    
    Args:
        store: RDF store object
        language: Language code for label lookups (default: English)
        encoding: Encoding of labels (use "wikidata" for ``\\\\Uxx`` escaped strings)
        labelProperties: Additional property URIs for labels (default: ``rdfs:label`` & 
            ``skos:prefLabel``)
        typeProperties: Additional property URIs for types (default: ``rdf:type``)
        qualifierIDProperty: Property URI for (``<qualifier> <id> <entity>``) triples
        statementURIprefix: URI prefix for (``<entity> prefix:foo <qualifier>``) triples    
    """

    YEAR = re.compile("^(\d{4})(?:[-–—]\d{2,4})?$")

    def __init__(
        self,
        store=None,
        language="en",
        stringmatch="jaccard",
        encoding=None,
        labelProperties=[],
        typeProperties=[],
        qualifierIDProperty=None,
        statementURIprefix=None,
        **kwargs,
    ):

        self.language = language
        self.labelProperties = [URIRef(p) for p in labelProperties] + [
            URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        ]
        self.typeProperties = [URIRef(p) for p in typeProperties] + [
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        ]
        self.encoding = encoding
        self.stringmatch = stringmatch

        self.qualifierIDProperty = None
        if qualifierIDProperty:
            self.qualifierIDProperty = URIRef(qualifierIDProperty)

        self.statementURIprefix = statementURIprefix

        GraphDB.__init__(self, store=store)

    def get_about(self, uri):
        about = {}
        for _, p, o in self.triples([URIRef(uri), None, None]):
            about.setdefault(p, []).append(o)
        return SearchResult(uri, about)

    def search_entities(self, query: str, limit=1, add_about=False):
        if self.encoding and (query != query.encode("ascii", errors="ignore").decode()):
            if self.encoding == "wikidata":
                import json

                chars = [json.dumps(c)[1:-1] for c in query]
                chars = [
                    c[:2] + c[2:].upper() if c.startswith("\\") else c for c in chars
                ]
                newquery = "".join(chars)
                log.debug(f"Transformed {query} to {newquery}")
                query = newquery
            else:
                query = query.encode(self.encoding)
        results = [
            e
            for l in self.labelProperties
            for lang in [None, self.language]
            for e, _, _ in self.triples((None, l, Literal(query, lang=lang)))
        ][:limit]
        if not results:
            ls = [Literal(query, lang=lang).n3() for lang in [None, self.language]]
            ls = " or ".join(ls)
            log.debug(f"No {self.__class__.__name__} results for {query} ({ls})")
        else:
            log.debug(
                f"{len(results):2d} {self.__class__.__name__} results for {query}"
            )
        return [
            SearchResult(str(e), self.about(e) if add_about else {}) for e in results
        ]

    def label_match(self, uri, surface):
        if isinstance(uri, URIRef):
            s = uri
            for lp in self.labelProperties:
                for _, _, o in self.triples([s, lp, None]):
                    for match in self.literal_match(o, surface):
                        yield match

    def literal_match(self, literal, surface):

        dtype = literal.datatype if hasattr(literal, "datatype") else None
        literal, surface = str(literal).strip(), str(surface).strip()

        # TODO: literal distance function module
        score = 0
        if dtype:

            # Typed literals should match well
            try:
                l = datetime.datetime.fromisoformat(literal).timestamp()
                yearmatch = self.YEAR.match(surface)
                if yearmatch:
                    s = datetime.datetime(int(yearmatch.groups()[0]), 1, 1).timestamp()
                else:
                    s = datetime.datetime.fromisoformat(surface).timestamp()

                score = max(0, 1 - (abs(s - l) / max(abs(s), abs(l))))

                if score > 0.95:
                    yield LiteralMatchResult(score, literal, dtype)
                    return
            except Exception as e:
                pass

            try:
                s, l = float(surface.replace(",", "")), float(literal.replace(",", ""))
                score = max(0, 1 - (abs(s - l) / max(abs(s), abs(l))))
                if score > 0.95:
                    yield LiteralMatchResult(score, literal, dtype)
                    return
            except Exception as e:
                pass

            s, l = surface.lower(), literal.lower()
            score = bool(s == l)

        elif surface and literal:
            # Strings may match approximately
            if self.stringmatch == "jaccard":
                s, l = set(surface.lower().split()), set(literal.lower().split())
                if s and l:
                    score = len(s & l) / len(s | l)
            elif self.stringmatch == "levenshtein":
                import Levenshtein

                s, l = surface.lower(), literal.lower()
                if s and l:
                    m = min(len(s), len(l))
                    score = max(0, (m - Levenshtein.distance(s, l)) / m)

        if score:
            yield LiteralMatchResult(score, literal, dtype)

    def match(self, e, p, surface):
        matches = []
        for _, _, o in self.triples((e, p, None)):

            if isinstance(o, URIRef):
                for lp in self.labelProperties:
                    for _, _, label in self.triples((o, lp, None)):
                        for s, label, dt in self.literal_match(label, surface):
                            matches.append((s, label, o))
            else:
                for s, _, dt in self.literal_match(o, surface):
                    matches.append((s, o, dt))

        if matches:
            return max(matches, key=lambda x: x[0])

        return 0, None, None

    def _yield_qualified_statements_about(self, e):
        if self.qualifierIDProperty:
            # Qualifier model with ID property
            for q, _, _ in self.triples([None, self.qualifierIDProperty, e]):
                yield q

        elif self.statementURIprefix:
            # Qualifier model with statement URI prefix
            for _, _, q in self.triples([e, None, None]):
                if str(q).startswith(self.statementURIprefix):
                    yield q

    def get_rowfacts(
        self, celltexts: List[str], entsets: List[Collection[str]]
    ) -> Iterator[MatchResult]:
        """Get matched facts for a row
        
        Args:
            celltexts: Cell text
            entsets: Set of URIs per cell
        """
        for ci1, ents1 in enumerate(entsets):
            for ci2, ents2 in enumerate(entsets):
                if ci1 == ci2:
                    pass

                for e1, e2 in ((e1, e2) for e1 in ents1 for e2 in ents2 if e1 != e2):

                    # Find simple matches
                    match = False
                    for s, p, o in self.triples([e1, None, e2]):
                        yield MatchResult((ci1, ci2), (e1, p, e2), [])
                        match = True
                    if not match:
                        continue

                    for q in self._yield_qualified_statements_about(e1):
                        _, ps, os = zip(*self.triples([q, None, None]))
                        if e2 in set(os):
                            mainprop, qmatches = None, []
                            for p, o in zip(ps, os):
                                if o == e2:
                                    mainprop = p
                                    continue

                                if hasattr(o, "datatype"):
                                    for ci, txt in enumerate(celltexts):
                                        for lm in self.literal_match(o, txt):
                                            qm = QualifierMatchResult(ci, (q, p, o), lm)
                                            qmatches.append(qm)
                                else:
                                    for ci, es in enumerate(entsets):
                                        if o in es:
                                            qm = QualifierMatchResult(
                                                ci, (q, p, o), None
                                            )
                                            qmatches.append(qm)

                            yield MatchResult((ci1, ci2), (e1, mainprop, e2), qmatches)
