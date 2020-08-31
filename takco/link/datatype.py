import time
import re
import datetime
import collections
import itertools
import math
import decimal
import enum
from typing import Dict, List, Tuple

# import datefinder

URI = str
CellValue = str


def is_literal_type(uri):
    return uri.startswith("http://www.w3.org/2001/XMLSchema#")


class CellType(enum.Enum):
    """Cell datatype enumeration"""

    pass


class SimpleCellType(CellType):
    NUMBER = "http://www.w3.org/2001/XMLSchema#decimal"
    ENTITY = "http://www.w3.org/2002/07/owl#Thing"
    TEXT = "http://www.w3.org/2001/XMLSchema#string"

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

        nfloat = sum(1 for s, _ in cell_ents if cls._try_cast_float(s) is not None)
        if nfloat > n / 2:
            return {cls.NUMBER: nfloat}

        nlink = sum(1 for _, l in cell_ents if l and not hasattr(l, "datatype"))
        if nlink > n / 2:
            return {cls.ENTITY: nlink}

        ntext = sum(1 for c, l in cell_ents if not l and cls._try_cast_float(c) is None)
        return {cls.TEXT: ntext}
