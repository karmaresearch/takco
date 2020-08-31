from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Container,
    NamedTuple,
    Any,
    Iterator,
)
import enum

URI = str
Literal = str
Node = Union[URI, Literal]


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


class QualifierMatchResult(NamedTuple):
    """The result of matching a qualified statement in the Knowledge Base"""

    column: int  #: Table column where this match was found
    triple: Triple  #: Statement qualifier triple
    match: Optional[LiteralMatchResult]  #: Match metadata


class MatchResult(NamedTuple):
    """The result of matching a Knowledge Base statement"""

    columns: Tuple[int, int]  #: Pair of (head, tail) table columns
    triple: Triple  #: Statement triple
    qualifiers: Container[QualifierMatchResult]  #: Statement qualifier matches


class SearchResult(dict):
    def __init__(self, uri: URI, about: Dict[URI, List[URI]] = None, score: int = 1):
        """An entity search result with optional score"""
        self.uri = uri
        self.update(about or {})
        self.score = score

    def __repr__(self):
        return f"SearchResult('{self.uri}', {dict(self)}, score={self.score})"


class Searcher:
    """For searching and matching for Knowledge Base entities."""

    def get_about(self, uri: URI) -> Optional[SearchResult]:
        """Get facts about an entity.
        
        Args:
            uri: Entity URI
        """
        return None

    def search_entities(
        self, query: str, limit: int = 1, add_about: bool = False
    ) -> Iterator[SearchResult]:
        """Search for entities using a label query.
        
        Args:
            query: Label query
            limit: Maximum number of results to return
            add_about: Include facts about the result entity
        """
        return

    def label_match(self, uri: URI, surface: str) -> Iterator[LiteralMatchResult]:
        """Match a cell value to a KB entity"""
        return

    def literal_match(
        self, literal: Literal, surface: str
    ) -> Iterator[LiteralMatchResult]:
        """Match a cell value to a KB literal"""
        score = float(bool(str(literal) == surface))
        if score:
            yield LiteralMatchResult(score, literal, None)


class WikiLookup:
    def lookup_wikititle(self, title: str) -> str:
        """Lookup Wikipedia title in KB
        
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
            uri = self.lookup_wikititle(href, **kwargs)
            if uri:
                for ri, ci in rowcols:
                    ci_ri_ents.setdefault(str(ci), {}).setdefault(str(ri), {})[uri] = 1
        return ci_ri_ents


class Database:
    """For querying a Knowledge Base."""

    def about(self, e: URI) -> Dict[URI, List[Node]]:
        """Look up facts about an entity
        
        Args:
            e: Entity URI to query
        """
        about = {}
        for (_, p, o) in self.triples((e, None, None)):
            about.setdefault(p, []).append(o)
        return about

    def count(self, triplepattern: TriplePattern) -> int:
        """Count the number of triples that match a pattern
        
        Args:
            triplepattern: A 3-tuple of None or URI
        """
        return

    def __len__(self):
        return self.count([None, None, None])

    def triples(self, triplepattern: TriplePattern) -> Iterator[Triple]:
        """Yield triples that match a pattern
        
        Args:
            triplepattern: A 3-tuple of None or URI
        """
        return

    def pages_about(self, triplepattern=None) -> Iterator[Tuple[URI, str]]:
        """Yield URLs of webpages about subjects of a triple pattern
        
        Args:
            triplepattern: A 3-tuple of None or URI
        """
        return


class Linker:
    """For linking tables to a Knowledge Base
        
    Args:
        searcher: The Knowledge Base searcher to use
    """

    def __init__(self, searcher: Searcher, limit=1):
        self.searcher = searcher
        self.limit = limit

    def _rowcol_results(
        self, rows, usecols=None, skiprows=None, existing_entities=None, **kwargs
    ) -> Dict[Tuple[int, int], Container[SearchResult]]:

        existing_entities = existing_entities or {}
        cell_rowcols = {}
        for ri, row in enumerate(rows):
            if (not skiprows) or (ri not in skiprows):
                for ci, cell in enumerate(row):
                    if (not usecols) or (ci in usecols):
                        if not existing_entities.get(ci, {}).get(ri, {}):
                            cell_rowcols.setdefault(cell, set()).add((ri, ci))

        rowcol_results = {}
        for cell, rowcols in cell_rowcols.items():
            results = self.searcher.search_entities(cell, **kwargs)
            for rowcol in rowcols:
                rowcol_results[rowcol] = results
        return rowcol_results

    def link(self, rows, usecols=None, skiprows=None, existing=None):
        existing = existing or {}

        existing_entities = (existing or {}).get("entities", {})
        rowcol_results = self._rowcol_results(
            rows,
            usecols=usecols,
            skiprows=skiprows,
            limit=self.limit,
            existing_entities=existing_entities,
        )

        entities = {}
        for (ri, ci), results in rowcol_results.items():
            entities.setdefault(str(ci), {})[str(ri)] = {r.uri: 1 for r in results}

        return {"entities": entities}
