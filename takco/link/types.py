"""
This module is executable. Run ``python -m takco.link.types -h`` for help.
"""
__all__ = ["Typer", "SimpleTyper", "EntityTyper", "EntityBloom"]

import logging as log
import time
import re
import datetime
import collections
import itertools
import math
import decimal
import enum
from typing import Dict, List, Tuple, Collection, Optional, Mapping, Iterable
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from rdflib import URIRef, Literal

from .base import Asset, LiteralMatchResult, Database

URI = str
CellValue = str

YEAR_PATTERN = re.compile("^(\d{4})(?:[-–—]\d{2,4})?$")

RDFTYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFSUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


class Typer(Asset, ABC):
    @abstractmethod
    def coltype(
        self, cell_ents: Iterable[Tuple[str, Collection[URI]]],
    ) -> Mapping[str, int]:
        """Find column type for cells and their entities"""
        pass

    def literal_match(
        cls, literal: Literal, surface: str
    ) -> Iterable[LiteralMatchResult]:
        """Match a cell value to a KB literal"""
        score = float(bool(str(literal) == surface))
        if score:
            yield LiteralMatchResult(score, literal, None)

    @abstractmethod
    def is_literal_type(self) -> bool:
        """Return whether this is a literal type"""
        pass

@dataclass
class SimpleTyper(Typer):
    use_dateparser: str = "dateutil"
    stringmatch: str = "jaccard"
    NUMBER = "http://www.w3.org/2001/XMLSchema#decimal"
    ENTITY = "http://www.w3.org/2002/07/owl#Thing"
    TEXT = "http://www.w3.org/2001/XMLSchema#string"
    DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"

    def nodes(self, cellvalue, links=[]):
        if self == self.NUMBER:
            try:
                d = decimal.Decimal(cellvalue.replace(",", ""))
                yield f'"+{d}"^^<http://www.w3.org/2001/XMLSchema#decimal>'
            except:
                pass

        if self == self.ENTITY:
            for link in links:
                uri = link.get("uri")
                if uri:
                    yield f"<{uri}>"

        if self == self.TEXT:
            cellvalue = cellvalue.replace('"', '"')
            vals = set()
            for pat in ['"%s"', '"%s"@en', '"%s"@en ']:
                for v in [cellvalue, cellvalue.lower()]:
                    vals.add(pat % v)
            yield from vals

    @staticmethod
    def _try_cast_float(s):
        try:
            s = s.lower().replace(",", "").replace(".", "").replace("−", "-")
            return float(s)
        except:
            pass

    def _dateparse(self, x):
        if self.use_dateparser == "dateparser":
            import dateparser

            try:
                return dateparser.parse(x)
            except:
                pass
        elif self.use_dateparser == "dateutil":
            import dateutil

            try:
                return dateutil.parser.parse(x)
            except:
                pass
        else:
            import datetime

            try:
                return datetime.datetime.fromisoformat(x)
            except:
                pass

    def is_literal_type(self, typ):
        return str(typ).startswith("http://www.w3.org/2001/XMLSchema#")

    def coltype(self, cell_ents):
        n = len(cell_ents)
        counts: collections.Counter = collections.Counter()
        for c, l in cell_ents:
            # This is a simple cell typing hierarchy
            # The first pattern to match gives the cell a type
            if YEAR_PATTERN.match(c):
                counts[self.DATETIME] += 1
            elif self._try_cast_float(c) != None:
                counts[self.NUMBER] += 1
            elif l and not hasattr(l, "datatype"):
                counts[self.ENTITY] += 1
            elif self._dateparse(c):
                counts[self.DATETIME] += 1
            else:
                counts[self.TEXT] += 1

        # Return a type if it covers half of the cells in the column
        for t in counts:
            if counts[t] > n / 2:
                return {t: counts[t] / n}
        return {}

    def literal_match(self, literal: Literal, surface: str):

        dtype = literal.datatype if hasattr(literal, "datatype") else None
        literal, surface = str(literal).strip(), str(surface).strip()

        score = 0.0
        if dtype:
            # Typed literals should match well

            if str(dtype) == str(self.DATETIME):
                try:
                    l = datetime.datetime.fromisoformat(literal).timestamp()

                    yearmatch = YEAR_PATTERN.match(surface)
                    if yearmatch:
                        year = int(yearmatch.groups()[0])
                        s = datetime.datetime(year, 1, 1).timestamp()
                    else:
                        try:
                            s = datetime.datetime.fromisoformat(surface).timestamp()
                        except:
                            s = self._dateparse(surface).timestamp()
                    if s:
                        score = max(0, 1 - (abs(s - l) / (60 * 60 * 24 * 365)))

                        if score:
                            yield LiteralMatchResult(score, literal, dtype)
                            return
                #                         else:
                #                             log.debug(f"No date match ({l},{s}) = {score}")
                except Exception as e:
                    pass
            else:
                try:
                    s = float(surface.replace(",", ""))
                    l = float(literal.replace(",", ""))
                    score = max(0, 1 - (abs(s - l) / max(abs(s), abs(l))))
                    if score > 0.95:
                        yield LiteralMatchResult(score, literal, dtype)
                        return
                except Exception as e:
                    pass

            score = bool(surface.lower() == literal.lower())

        elif surface and literal:
            # Strings may match approximately
            if self.stringmatch == "jaccard":
                stok, ltok = set(surface.lower().split()), set(literal.lower().split())
                if stok and ltok:
                    score = len(stok & ltok) / len(stok | ltok)
            elif self.stringmatch == "levenshtein":
                import Levenshtein

                slow, llow = surface.lower(), literal.lower()
                if slow and llow:
                    m = min(len(slow), len(llow))
                    score = max(0, (m - Levenshtein.distance(slow, llow)) / m)

        if score:
            yield LiteralMatchResult(score, literal, dtype)


@dataclass
class EntityTyper(SimpleTyper):
    """Select ``topn`` KB types with highest cover
    
    Looks up entities in DB, and looks up types of those entities.
    The types ``topn`` that occur more than ``cover_threshold`` of the column are kept.

    """

    db: Optional[Database] = None
    type_prop: str = RDFTYPE
    supertype_prop: str = RDFSUBCLASS
    cover_threshold: float = 0.5
    topn: int = 1
    ignore_types: Collection[str] = ()

    def __post_init__(self):
        if self.db is None:
            raise TypeError(f"Error creating {self}, param `db` is required!")

    def __enter__(self):
        self.db.__enter__()
        return self

    def __exit__(self, *args):
        self.db.__exit__(*args)

    def supertypes(self, e):
        # Don't recurse, because broad types are never useful anyway
        for t in self.db.get_prop_values(e, self.type_prop):
            yield t
            if self.supertype_prop:
                for st in self.db.get_prop_values(t, self.supertype_prop):
                    yield st

    def coltype(self, cell_ents):
        types = super().coltype(cell_ents)
        if SimpleTyper.ENTITY in types:
            n = sum(1 for _, ents in cell_ents if ents)
            counts: Dict = collections.Counter()
            for c, es in cell_ents:
                ts = [set(self.supertypes(e)) for e in es]
                if ts:
                    counts.update(set.union(*ts))

            # Select topn types with highest cover
            scores = [
                (t, c / n)
                for t, c in counts.most_common()
                if c > (n * self.cover_threshold)
            ][: self.topn]
            if scores:
                return dict(scores)
        return types


@dataclass
class EntityBloom(SimpleTyper):
    """Find entity columns using bloom filter"""

    bloomfile: Optional[Path] = None
    threshold: float = 0.5

    def __post_init__(self):
        if self.bloomfile is None:
            raise TypeError(f"Error creating {self}, param `bloomfile` is required!")

    def __enter__(self):
        if not hasattr(self, "bloom"):
            from pybloomfilter import BloomFilter

            self.bloom = BloomFilter.open(self.bloomfile)
        return self

    def __exit__(self, *args):
        if hasattr(self, "bloom"):
            del self.bloom

    @staticmethod
    def create(infile, outfile, capacity: int, error_rate: float = 0.05):
        import tqdm
        import urllib
        from pybloomfilter import BloomFilter

        bf = BloomFilter(capacity, error_rate, outfile)
        with open(infile) as f:
            for _, word in enumerate(tqdm.tqdm(f, total=capacity)):
                if "%" in word:
                    word = urllib.parse.unquote(word).lower()
                word = word.rstrip()
                bf.add(word)

        bf.close()

    def coltype(self, cell_ents):
        types = super().coltype(cell_ents)
        if SimpleTyper.TEXT in types and cell_ents:
            cells = (c.lower().strip() for c, _ in cell_ents)
            cells = (c for c in cells if self._try_cast_float(c) == None and len(c) > 1)
            n = sum(int(c in self.bloom) for c in cells)
            score = n / len(cell_ents)
            if score > self.threshold:
                return {SimpleTyper.ENTITY: score}
        return types


if __name__ == "__main__":
    import defopt, json, os, typing

    r = defopt.run(
        {"entitybloom": EntityBloom.create},
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
