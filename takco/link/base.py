from typing import (
    Union,
    List,
    Dict,
    Mapping,
    Set,
    Tuple,
    Optional,
    Collection,
    NamedTuple,
    Any,
    Iterable,
)
import enum
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

URI = str
Literal = str
Node = Union[URI, Literal]


class Asset(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return


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
        about: Mapping[URI, List[URI]] = None,
        context_matches: Mapping[int, Mapping[URI, Collection[LiteralMatchResult]]] = None,
        score: float = 1.0,
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


class Searcher(Asset, ABC):
    """For searching and matching for Knowledge Base entities."""

    @abstractmethod
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
        pass


class Lookup(Asset, ABC):
    @abstractmethod
    def lookup_title(self, title: str) -> Optional[str]:
        """Lookup (Wikipedia) page title in KB, return URI

        Args:
            title: The title to look up
        """
        pass

    def lookup_cells(self, hrefs):
        href_rowcols: Dict[str, Set[Tuple[int, int]]] = {}
        for ri, row in enumerate(hrefs):
            for ci, hs in enumerate(row):
                for href in hs:
                    if href:
                        href_rowcols.setdefault(href, set()).add((ri, ci))

        ci_ri_ents: Dict[str, Dict[str, Dict[str, float]]] = {}
        for href, rowcols in href_rowcols.items():
            uri = self.lookup_title(href)
            if uri:
                for ri, ci in rowcols:
                    ci_ri_ents.setdefault(str(ci), {}).setdefault(str(ri), {})[uri] = 1
        return ci_ri_ents

    def flush(self):
        pass


class Database(Asset, ABC):
    """For querying a Knowledge Base."""

    def get_prop_values(self, e: URI, prop: URI):
        return self.about(e).get(prop, [])

    @abstractmethod
    def about(self, e: URI, att_uris=None) -> Mapping[URI, List[Node]]:
        """Look up facts about an entity

        Args:
            e: Entity URI to query
        """
        pass


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
    ) -> Mapping[Tuple[int, int], Collection[SearchResult]]:

        existing_entities = existing_entities or {}
        col_classes = col_classes or {}

        # TODO: clean this up
        if contextual:
            rowcol_searchresults: Dict[Tuple[int, int], Collection[SearchResult]] = {}
            query_rowcols_list = []
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
                                    query_rowcols_list.append((query, (ri, ci)))
                            else:
                                rowcol_searchresults[(ri, ci)] = [
                                    SearchResult(e, score=score)
                                    for e, score in existing.items()
                                ]
            if query_rowcols_list:
                queries, _ = zip(*query_rowcols_list)
                allresults = self.searcher.search_entities(queries, **kwargs)
                for (query, rowcol), results in zip(query_rowcols_list, allresults):
                    rowcol_searchresults[rowcol] = results
        else:
            rowcol_searchresults = {}
            query_rowcols_dict: Dict[Tuple[str, Tuple], Set[Tuple[int, int]]] = {}
            for ri, row in enumerate(rows):
                if (not skiprows) or (ri not in skiprows):
                    for ci, cell in enumerate(row):
                        query = (cell, tuple(col_classes.get(ci, [])))

                        if (not usecols) or (ci in usecols):
                            existing = existing_entities.get(ci, {}).get(ri, {})
                            if not existing:
                                if not (len(cell) < 2 or cell.isnumeric()):
                                    query_rowcols_dict.setdefault(query, set()).add(
                                        (ri, ci)
                                    )
                            else:
                                rowcol_searchresults[(ri, ci)] = [
                                    SearchResult(e, score=score)
                                    for e, score in existing.items()
                                ]
            queries = [(q, {"classes": cs}) for q, cs in query_rowcols_dict]
            allresults = self.searcher.search_entities(queries, **kwargs)
            for (_, rowcols), results in zip(query_rowcols_dict.items(), allresults):
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

        entities: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (ri, ci), results in rowcol_searchresults.items():
            entities.setdefault(str(ci), {})[str(ri)] = {r.uri: 1.0 for r in results}

        return {"entities": entities}

    def flush(self):
        pass
