import rdflib
import logging as log
from rdflib import URIRef, Literal
from rdflib.store import Store as RDFStore
from typing import List, Dict, Tuple, Collection, Iterable
from dataclasses import dataclass
import datetime
import re

from ..base import (
    Searcher,
    Database,
    SearchResult,
    Triple,
    Node,
    LiteralMatchResult,
    Typer,
)
from ..types import SimpleTyper
from ..integrate import NaryDB, NaryMatchResult, QualifierMatchResult


def encode_wikidata(query):
    import json

    chars = [json.dumps(c)[1:-1] for c in query]
    chars = [c[:2] + c[2:].upper() if c.startswith("\\") else c for c in chars]
    newquery = "".join(chars)
    return newquery

@dataclass
class GraphDB(Database):
    store: RDFStore
    labelProperties: Collection[str] = ()
    typeProperties: Collection[str] = ()

    def __enter__(self):
        try:
            self.store.open(self.store.endpoint)
        except Exception as e:
            log.warn(f"Could not open graph due to {e}")
        return self

    def __exit__(self, *args):
        self.store.close()

    def triples(self, triplepattern, **kwargs):
        return (t for t, _ in self.store.triples(triplepattern, **kwargs))

    def get_prop_values(self, e, p):
        return set(o for _, _, o in self.triples([URIRef(e), URIRef(p), None]))

    def about(self, uri, att_uris=None):
        about = {}
        if att_uris and hasattr(att_uris, "__iter__"):
            for att in att_uris:
                for _, p, o in self.triples([URIRef(uri), URIRef(att), None]):
                    about.setdefault(p, []).append(o)
        else:
            for _, p, o in self.triples([URIRef(uri), None, None]):
                about.setdefault(p, []).append(o)
        return about

    def count(self, triplepattern):
        if hasattr(self.store, "count"):
            return self.store.count(triplepattern)
        elif hasattr(self.store, "hdt_document"):
            _, n = self.store.hdt_document.search(triplepattern)
            return n
        else:
            ts = self.triples(triplepattern)
            return len(ts) if hasattr(ts, "__len__") else sum(1 for _ in ts)

    def __len__(self):
        return self.store.__len__()


class RDFSearcher(Searcher, GraphDB, NaryDB):
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

    def __init__(
        self,
        store=None,
        language="en",
        refsort=True,
        typer: Typer = SimpleTyper(),
        stringmatch="jaccard",
        encoding=None,
        labelProperties=[],
        typeProperties=[],
        qualifierIDProperty=None,
        statementURIprefix=None,
        **kwargs,
    ):
        GraphDB.__init__(self, store=store)
        self.language = language
        self.labelProperties = [URIRef(p) for p in labelProperties] + [
            URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
            URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
        ]
        self.typeProperties = [URIRef(p) for p in typeProperties] + [
            URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        ]

        self.encoding = encoding
        self.refsort = refsort
        self.typer = typer
        self.stringmatch = stringmatch

        self.qualifierIDProperty = None
        if qualifierIDProperty:
            self.qualifierIDProperty = URIRef(qualifierIDProperty)

        self.statementURIprefix = statementURIprefix

    def search_entities(self, query_params, limit=1, add_about=False):
        if log.getLogger().level == log.INFO:
            try:
                import tqdm

                query_params = tqdm.tqdm(query_params)
            except:
                pass

        for query, _ in query_params:
            is_ascii = query == query.encode("ascii", errors="ignore").decode()
            if self.encoding and not is_ascii:
                if self.encoding == "wikidata":
                    query = encode_wikidata(query)
                else:
                    query = query.encode(self.encoding)
            result_uris = [
                e
                for l in self.labelProperties
                for lang in [None, self.language]
                for e, _, _ in self.triples((None, l, Literal(query, lang=lang)))
            ]

            if not result_uris:
                ls = [Literal(query, lang=lang).n3() for lang in [None, self.language]]
                ls = " or ".join(ls)
                log.debug(f"No {self.__class__.__name__} results for {query} ({ls})")
            else:
                log.debug(
                    f"{len(result_uris):2d} {self.__class__.__name__} results for {query}"
                )

            e_score = {}
            if self.refsort:
                # sort by inverse refCount score
                for e in result_uris:
                    e_score[e] = 1 - 1 / (1 + self.count([None, None, URIRef(e)]))

                result_uris = sorted(result_uris, key=lambda e: -e_score[e])

            results = [
                SearchResult(
                    str(e),
                    self.about(e, add_about) if add_about else {},
                    score=e_score.get(e, 1),
                )
                for e in result_uris[:limit]
            ]

            yield results

    def label_match(self, uri, surface):
        if isinstance(uri, URIRef):
            s = uri
            for lp in self.labelProperties:
                for _, _, o in self.triples([s, lp, None]):
                    for match in self.typer.literal_match(o, surface, self.stringmatch):
                        yield match
        else:
            for match in self.typer.literal_match(uri, surface, self.stringmatch):
                yield match

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
    ) -> Iterable[NaryMatchResult]:
        """Get matched (n-ary) facts for a row

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
                        yield NaryMatchResult((ci1, ci2), (e1, p, e2), [])
                        match = True
                    if not match:
                        continue

                    for q in self._yield_qualified_statements_about(e1):
                        _, ps, os = zip(*self.triples([q, None, None]))
                        if e2 in set(os):
                            mainprop, qmatches = None, []
                            for p, o in zip(ps, os):
                                if o == e1:
                                    continue
                                if o == e2:
                                    mainprop = p
                                    continue

                                if hasattr(o, "datatype"):
                                    for ci, txt in enumerate(celltexts):
                                        for lm in self.typer.literal_match(o, txt):
                                            qm = QualifierMatchResult(ci, (q, p, o), lm)
                                            qmatches.append(qm)
                                else:
                                    for ci, es in enumerate(entsets):
                                        if o in es:
                                            qm = QualifierMatchResult(
                                                ci, (q, p, o), None
                                            )
                                            qmatches.append(qm)

                            yield NaryMatchResult(
                                (ci1, ci2), (e1, mainprop, e2), qmatches
                            )
