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
        self.mdir = (Path(fdir) / Path(self.name)).resolve() if fdir else None
        self.indexed = False

        self.source = source
        self.tokenize = tokenize

        self.cell_tables: Dict[Cell, Set[int]] = {}
        self.table_cells: Dict[int, Dict[Cell, Set[int]]] = {}
        self.pickles = ["cell_tables", "table_cells"]

    def _get_cells(self, table):
        rows = []
        if self.source in table:
            r = table[self.source]
            rows = zip(*[r if isinstance(r, list) else [r]])
            ci_range = [table['tableIndex']]
        else:
            if self.source != "head":
                rows += table.get("tableData", [])
            if self.source != "body":
                rows += table.get("tableHeaders", [])
            
            rows = [[c.get('text', '') for c in r] for r in rows]
            ci_range = range(
                table["columnIndexOffset"],
                table["columnIndexOffset"] + table["numCols"],
            )
        
        for row in rows:
            for ci, cell in zip(ci_range, row):
                txt = frozenset(self.tokenize(cell))
                if txt:
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
            if self.mdir:
                self.mdir.mkdir(parents=True, exist_ok=True)
                for p in self.pickles:
                    with (self.mdir / Path(f"{p}.pickle")).open("wb") as fw:
                        pickle.dump(getattr(self, p), fw)
                self.__exit__()

    def __enter__(self):
        super().__enter__()
        if self.indexed and self.mdir:
            for p in self.pickles:
                fpath = self.mdir / Path(f"{p}.pickle")
                if fpath.exists():
                    setattr(self, p, pickle.load(fpath.open("rb")))
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.indexed and self.mdir:
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
            cells1 = self.table_cells.get(ti1, {})
            cells2 = self.table_cells.get(ti2, {})

            for cell1, cis1 in cells1.items():
                for cell2, cis2 in cells2.items():

                    special1 = all(c.startswith("_") for c in cell1)
                    special2 = all(c.startswith("_") for c in cell2)
                    if special1 or special2:
                        # for special cells, ignore jaccard value
                        cell_jacc = float("nan")
                    else:
                        u = len(cell1 | cell2)
                        cell_jacc = len(cell1 & cell2) / u if u else 0

                    for ci1 in cis1:
                        for ci2 in cis2:
                            yield (ti1, ti2, ci1, ci2), cell_jacc
