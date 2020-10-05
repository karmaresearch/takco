__all__ = ["SimpleCellType", "EntityType"]

import logging as log
import time
import re
import datetime
import collections
import itertools
import math
import decimal
import enum
from typing import Dict, List, Tuple

from rdflib import URIRef, Literal

from .base import CellType, LiteralMatchResult, Database

URI = str
CellValue = str

YEAR_PATTERN = re.compile("^(\d{4})(?:[-–—]\d{2,4})?$")

RDFTYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFSUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


class SimpleCellType(CellType):
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

    @staticmethod
    def _dateparse(x):
        import dateparser

        try:
            return dateparser.parse(x)
        except:
            return None

    def is_literal_type(self, typ):
        return str(typ).startswith("http://www.w3.org/2001/XMLSchema#")

    @classmethod
    def coltype(
        cls, cell_ents: List[Tuple[CellValue, List[URI]]],
    ) -> Dict[CellType, int]:

        n = len(cell_ents)
        counts = collections.Counter()
        for c, l in cell_ents:
            # This is a simple cell typing hierarchy
            # The first pattern to match gives the cell a type
            if YEAR_PATTERN.match(c):
                counts[cls.DATETIME] += 1
            elif cls._try_cast_float(c):
                counts[cls.NUMBER] += 1
            elif l and not hasattr(l, "datatype"):
                counts[cls.ENTITY] += 1
            elif cls._dateparse(c):
                counts[cls.DATETIME] += 1
            else:
                counts[cls.TEXT] += 1

        # Return a type if it covers half of the cells in the column
        for t in counts:
            if counts[t] > n / 2:
                return {t: counts[t] / n}
        return {}

    @classmethod
    def literal_match(cls, literal: Literal, surface: str, stringmatch="jaccard"):

        dtype = literal.datatype if hasattr(literal, "datatype") else None
        literal, surface = str(literal).strip(), str(surface).strip()

        score = 0
        if dtype:
            # Typed literals should match well

            if str(dtype) == str(cls.DATETIME):
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
                            s = cls._dateparse(surface).timestamp()
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

            s, l = surface.lower(), literal.lower()
            score = bool(s == l)

        elif surface and literal:
            # Strings may match approximately
            if stringmatch == "jaccard":
                s, l = set(surface.lower().split()), set(literal.lower().split())
                if s and l:
                    score = len(s & l) / len(s | l)
            elif stringmatch == "levenshtein":
                import Levenshtein

                s, l = surface.lower(), literal.lower()
                if s and l:
                    m = min(len(s), len(l))
                    score = max(0, (m - Levenshtein.distance(s, l)) / m)

        if score:
            yield LiteralMatchResult(score, literal, dtype)


class EntityType(SimpleCellType):
    def __init__(
        self,
        db: Database,
        type_prop=RDFTYPE,
        supertype_prop=RDFSUBCLASS,
        cover_threshold=0.5,
        topn=None,
    ):
        self.db = db
        self.type_prop = str(type_prop)
        self.supertype_prop = str(supertype_prop) if supertype_prop else None
        self.cover_threshold = float(cover_threshold)
        self.topn = int(topn) if topn else None

    def supertypes(self, e):
        # Don't recurse, because broad types are never useful anyway
        for t in self.db.get_prop_values(e, self.type_prop):
            yield t
            if self.supertype_prop:
                for st in self.db.get_prop_values(t, self.supertype_prop):
                    yield st

    def coltype(self, cell_ents):
        types = super().coltype(cell_ents)
        if SimpleCellType.ENTITY in types:
            n = sum(1 for _, ents in cell_ents if ents)
            counts = collections.Counter()
            for c, es in cell_ents:
                ts = [set(self.supertypes(e)) for e in es]
                if ts:
                    counts.update(set.union(*ts))

            scores = [
                (t, c / n)
                for t, c in counts.most_common()
                if c > (n * self.cover_threshold)
            ][: self.topn]
            if scores:
                return dict(scores)
        return types
