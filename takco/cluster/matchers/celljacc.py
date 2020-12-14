from pathlib import Path
from typing import Dict, Set, FrozenSet
import logging as log
import pickle
import shutil

from .matcher import Matcher, default_tokenize
from .. import cluster


Cell = FrozenSet[str]


class CellJaccMatcher(Matcher):
    """Jaccard similarity of table cells from header and/or body"""

    def __init__(
        self,
        fdir: Path = None,
        name=None,
        source="head",
        tokenize=default_tokenize,
        create=False,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.indexed = False
        self.set_storage(fdir)

        self.source = source
        self.tokenize = tokenize

        self.cell_tables: Dict[Cell, Set[int]] = {}
        self.table_cells: Dict[int, Dict[Cell, Set[int]]] = {}
        self.pickles = ["cell_tables", "table_cells"]

    def _get_cells(self, table):
        rows = []
        if self.source in table:
            r = table[self.source]
            rows = list(zip(*[r if isinstance(r, list) else [r]]))
            ci_range = [table["tableIndex"]]
        else:
            if self.source != "head":
                rows += table.get("tableData", [])
            if self.source != "body":
                rows += table.get("tableHeaders", [])

            rows = [[c.get("text", "") for c in r] for r in rows]
            ci_range = list(
                range(
                    table["columnIndexOffset"],
                    table["columnIndexOffset"] + table["numCols"],
                )
            )

        for row in rows:
            for ci, cell in zip(ci_range, row):
                txt = frozenset(self.tokenize(cell))
                if txt and not all(c.startswith("_") for c in txt):
                    yield ci, txt

    def add(self, table):
        if table:
            ti = table["tableIndex"]
            for ci, cell in self._get_cells(table):
                if not any(tok.startswith("_") for tok in cell):
                    self.cell_tables.setdefault(cell, set()).add(ti)
                self.table_cells.setdefault(ti, {}).setdefault(cell, set()).add(ci)

    def merge(self, matcher: "CellJaccMatcher"):
        if matcher is not None:
            for cell, tables in matcher.cell_tables.items():
                self.cell_tables.setdefault(cell, set()).update(tables)

            for ti, cells in matcher.table_cells.items():
                for cell, cis in cells.items():
                    self.table_cells.setdefault(ti, {}).setdefault(cell, set()).update(
                        cis
                    )
        return self

    def index(self):
        if not self.indexed:
            self.indexed = True
            if self.storage:
                for p in self.pickles:
                    self.storage.save_pickle(getattr(self, p), p)
                self.close()

    def __enter__(self):
        if self.indexed and self.storage:
            for p in self.pickles:
                setattr(self, p, self.storage.load_pickle(p))
        return self

    def close(self):
        if self.indexed and self.storage:
            for p in self.pickles:
                delattr(self, p)

    def block(self, ti: int, cis):
        """Block tables on having some cell in common."""
        for cell in self.table_cells.get(ti, []):
            yield from self.cell_tables.get(cell, [])

    def match(self, tableid_colids_pairs):
        """Match columns on token jaccard."""
        pairs = cluster.progress(tableid_colids_pairs, f"Looking up {self.name}")
        for (ti1, _), (ti2, _) in pairs:
            cell_columns1 = self.table_cells.get(ti1, {})
            cell_columns2 = self.table_cells.get(ti2, {})

            for cell1, cis1 in cell_columns1.items():
                for cell2, cis2 in cell_columns2.items():
                    u = len(cell1 | cell2)
                    cell_jacc = len(cell1 & cell2) / u if u else 0

                    for ci1 in cis1:
                        for ci2 in cis2:
                            yield (ti1, ti2, ci1, ci2), cell_jacc
