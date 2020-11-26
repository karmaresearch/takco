from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Collection,
    NamedTuple,
    Any,
    Iterator,
)
import enum

URI = str
Literal = str
Node = Union[URI, Literal]


class Asset:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Annotation(dict):
    """A table annotation object. """


class Triple(NamedTuple):
    """A (subject, predicate, object) triple."""

    s: URI  #:
    p: URI  #:
    o: Node  #:


class TriplePattern(NamedTuple):
    """A (subject, predicate, object) triple pattern where any can be None."""

    s: Optional[URI]  #:
    p: Optional[URI]  #:
    o: Optional[Node]  #:


class LiteralMatchResult(NamedTuple):
    """The result of matching a value to a KB literal"""

    score: float  #:
    literal: Literal  #:
    datatype: Optional[Node]  #:


class SearchResult(dict):
    def __init__(
        self,
        uri: URI,
        about: Dict[URI, List[URI]] = None,
        context_matches: Dict[int, Dict[URI, Collection[LiteralMatchResult]]] = None,
        score: int = 1,
    ):
        """An entity search result with optional score"""
        self.uri = uri
        self.update(about or {})
        self.score = score
        self.context_matches = context_matches or {}

    def get(self, k, default=None):
        if k in self:
            return self[k]
        else:
            for c in set(x.__class__ for x in self.keys()):
                try:
                    if c(k) in self:
                        return self[c(k)]
                except:
                    pass
        return default

    def __repr__(self):
        return f"SearchResult('{self.uri}', {dict(self)}, {self.context_matches}, score={self.score})"


class Typer(Asset):
    def coltype(
        self, cell_ents: Iterator[Tuple[str, Collection[URI]]],
    ) -> Dict[str, int]:
        """Find column type for cells and their entities"""
        return {}

    @classmethod
    def literal_match(literal: Literal, surface: str) -> Iterator[LiteralMatchResult]:
        """Match a cell value to a KB literal"""
        score = float(bool(str(literal) == surface))
        if score:
            yield LiteralMatchResult(score, literal, None)

    def is_literal_type(self) -> bool:
        """Return whether this is a literal type"""
        return None


class Searcher(Asset):
    """For searching and matching for Knowledge Base entities."""

    def search_entities(
        self,
        query_contexts: Collection[Tuple[str, Collection[str]]],
        limit: int = 1,
        add_about: bool = False,
    ) -> Collection[Collection[SearchResult]]:
        """Search for entities using label queries.

        Args:
            query_contexts: Label queries and contexts
            limit: Maximum number of results to return
            add_about: Include facts about the result entity
        
        Returns:
            Search results per query
        """
        return


class Lookup(Asset):
    def lookup_title(self, title: str) -> str:
        """Lookup (Wikipedia) page title in KB, return URI

        Args:
            title: The title to look up
        """
        return

    def lookup_cells(self, hrefs, **kwargs):
        href_rowcols = {}
        for ri, row in enumerate(hrefs):
            for ci, hs in enumerate(row):
                for href in hs:
                    if href:
                        href_rowcols.setdefault(href, set()).add((ri, ci))

        ci_ri_ents = {}
        for href, rowcols in href_rowcols.items():
            uri = self.lookup_title(href, **kwargs)
            if uri:
                for ri, ci in rowcols:
                    ci_ri_ents.setdefault(str(ci), {}).setdefault(str(ri), {})[uri] = 1
        return ci_ri_ents


class Database(Asset):
    """For querying a Knowledge Base."""

    def get_prop_values(self, e: URI, prop: URI):
        return self.about(e).get(prop, [])

    def about(self, e: URI, att_uris=None) -> Dict[URI, List[Node]]:
        """Look up facts about an entity

        Args:
            e: Entity URI to query
        """
        about = {}
        for (_, p, o) in self.triples((e, None, None)):
            about.setdefault(p, []).append(o)
        return about

    def triples(self, pattern):
        return []


class Linker(Asset):
    """For linking tables to a Knowledge Base

    Args:
        searcher: The Knowledge Base searcher to use
    """

    def __init__(self, searcher: Searcher, limit=1):
        self.searcher = searcher
        self.limit = limit

    def get_rowcol_searchresults(
        self,
        rows,
        contextual=False,
        usecols=None,
        skiprows=None,
        existing_entities=None,
        col_classes=None,
        **kwargs,
    ) -> Dict[Tuple[int, int], Collection[SearchResult]]:

        existing_entities = existing_entities or {}
        col_classes = col_classes or {}

        if contextual:
            rowcol_searchresults = {}
            query_rowcols = []
            for ri, row in enumerate(rows):
                if (not skiprows) or (ri not in skiprows):
                    for ci, cell in enumerate(row):
                        cis = tuple(range(0, ci)) + tuple(range(ci + 1, len(row)))
                        context = {row[i]: i for i in cis}
                        classes = col_classes.get(ci, [])
                        params = {"context": context, "classes": classes}

                        query = (cell, params) if contextual else (cell, ())

                        if (not usecols) or (ci in usecols):
                            existing = existing_entities.get(ci, {}).get(ri, {})
                            if not existing:
                                if not (len(cell) < 2 or cell.isnumeric()):
                                    query_rowcols.append((query, (ri, ci)))
                            else:
                                rowcol_searchresults[(ri, ci)] = [
                                    SearchResult(e, score=score)
                                    for e, score in existing.items()
                                ]
            if query_rowcols:
                queries, _ = zip(*query_rowcols)
                allresults = self.searcher.search_entities(queries, **kwargs)
                for (query, rowcol), results in zip(query_rowcols, allresults):
                    rowcol_searchresults[rowcol] = results
        else:
            rowcol_searchresults = {}
            query_rowcols = {}
            for ri, row in enumerate(rows):
                if (not skiprows) or (ri not in skiprows):
                    for ci, cell in enumerate(row):
                        query = (cell, tuple(col_classes.get(ci, [])))

                        if (not usecols) or (ci in usecols):
                            existing = existing_entities.get(ci, {}).get(ri, {})
                            if not existing:
                                if not (len(cell) < 2 or cell.isnumeric()):
                                    query_rowcols.setdefault(query, set()).add((ri, ci))
                            else:
                                rowcol_searchresults[(ri, ci)] = [
                                    SearchResult(e, score=score)
                                    for e, score in existing.items()
                                ]
            queries = [(q, {"classes": cs}) for q, cs in query_rowcols]
            allresults = self.searcher.search_entities(queries, **kwargs)
            for ((query, clss), rowcols), results in zip(
                query_rowcols.items(), allresults
            ):
                for rowcol in rowcols:
                    rowcol_searchresults[rowcol] = results

        return rowcol_searchresults

    def link(self, rows, usecols=None, skiprows=None, existing=None):
        existing = existing or {}

        existing_entities = (existing or {}).get("entities", {})
        rowcol_searchresults = self.get_rowcol_searchresults(
            rows,
            usecols=usecols,
            skiprows=skiprows,
            limit=self.limit,
            existing_entities=existing_entities,
        )

        entities = {}
        for (ri, ci), results in rowcol_searchresults.items():
            entities.setdefault(str(ci), {})[str(ri)] = {r.uri: 1 for r in results}

        return {"entities": entities}
