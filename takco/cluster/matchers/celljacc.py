from pathlib import Path
import logging as log
import pickle
import shutil

from .matcher import Matcher


class CellJaccMatcher(Matcher):
    """Jaccard similarity of table cells from header and/or body"""

    def __init__(
        self,
        fdir: Path,
        name=None,
        source="head",
        tokenize=True,
        create=False,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.mdir = Path(fdir) / Path(self.name)
        self.indexed = False

        self.source = source
        self.tokenize = tokenize

        self.cell_tables = {}
        self.table_cells = {}
        self.pickles = ["cell_tables", "table_cells"]

        super().__init__(fdir)

    def _get_cells(self, table):
        rows = []
        if self.source != "head":
            rows += table.get("tableData", [])
        if self.source != "body":
            rows += table.get("tableHeaders", [])
        for row in rows:
            ci_range = range(
                table["columnIndexOffset"],
                table["columnIndexOffset"] + table["numCols"],
            )
            for ci, cell in zip(ci_range, row):
                txt = cell.get("text", "").lower()
                txt = (
                    frozenset(Matcher.tokenize(txt))
                    if self.tokenize
                    else frozenset([txt])
                )
                if txt:
                    yield ci, txt

    def add(self, table):
        if table:
            ti = table["tableIndex"]
            for ci, cell in self._get_cells(table):
                if not any(tok.startswith("_") for tok in cell):
                    self.cell_tables.setdefault(cell, set()).add(ti)
                self.table_cells.setdefault(ti, {}).setdefault(cell, set()).add(ci)

    def merge(self, matcher: Matcher):
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
            self.mdir.mkdir(parents=True, exist_ok=True)
            for p in self.pickles:
                with (self.mdir / Path(f"{p}.pickle")).open("wb") as fw:
                    pickle.dump(getattr(self, p), fw)
            self.__exit__()

    def __enter__(self):
        if self.indexed:
            for p in self.pickles:
                fpath = self.mdir / Path(f"{p}.pickle")
                if fpath.exists():
                    setattr(self, p, pickle.load(fpath.open("rb")))
        return self

    def __exit__(self, *args):
        if self.indexed:
            for p in self.pickles:
                delattr(self, p)

    def block(self, ti: int):
        """Block tables on having some cell in common."""
        for cell in self.table_cells.get(ti, []):
            yield from self.cell_tables.get(cell, [])

    def match(self, table_index_pairs):
        """Match columns on token jaccard."""
        for ti1, ti2 in table_index_pairs:
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
