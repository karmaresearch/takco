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

from .base import CellType, LiteralMatchResult

URI = str
CellValue = str

YEAR_PATTERN = re.compile("^(\d{4})(?:[-–—]\d{2,4})?$")


def dateparse(x):
    import dateparser

    try:
        return dateparser.parse(x)
    except:
        return None


def is_literal_type(uri):
    return uri.startswith("http://www.w3.org/2001/XMLSchema#")


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

    @classmethod
    def coltype(
        cls, cell_ents: List[Tuple[CellValue, List[URI]]]
    ) -> Dict[CellType, int]:

        n = len(cell_ents)
        counts = collections.Counter()
        for c, l in cell_ents:
            if YEAR_PATTERN.match(c):
                counts[cls.DATETIME] += 1
            elif cls._try_cast_float(c):
                counts[cls.NUMBER] += 1
            elif l and not hasattr(l, "datatype"):
                counts[cls.ENTITY] += 1
            elif dateparse(c):
                counts[cls.DATETIME] += 1
            else:
                counts[cls.TEXT] += 1

        for t in [cls.DATETIME, cls.NUMBER, cls.ENTITY, cls.TEXT]:
            if counts[t] > n / 2:
                return {t: counts[t]}
        return {}

    @classmethod
    def literal_match(cls, literal: Literal, surface: str, stringmatch="jaccard"):

        dtype = literal.datatype if hasattr(literal, "datatype") else None
        literal, surface = str(literal).strip(), str(surface).strip()

        score = 0
        if dtype:
            # Typed literals should match well

            if str(dtype) == str(cls.DATETIME.value):
                try:
                    l = datetime.datetime.fromisoformat(literal).timestamp()

                    yearmatch = YEAR_PATTERN.match(surface)
                    if yearmatch:
                        s = datetime.datetime(
                            int(yearmatch.groups()[0]), 1, 1
                        ).timestamp()
                    else:
                        try:
                            s = datetime.datetime.fromisoformat(surface).timestamp()
                        except:
                            s = dateparse(surface).timestamp()
                    if s:
                        score = max(0, 1 - (abs(s - l) / max(abs(s), abs(l))))

                        if score > 0.95:
                            yield LiteralMatchResult(score, literal, dtype)
                            return
                except Exception as e:
                    pass
            else:
                try:
                    s, l = (
                        float(surface.replace(",", "")),
                        float(literal.replace(",", "")),
                    )
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
