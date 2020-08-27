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
        source="head",
        tokenize=True,
        cell_threshold=0,
        create=False,
        **kwargs,
    ):
        mdir = Path(fdir) / Path("CellJaccMatcher")
        if create:
            shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.tokenize = tokenize
        self.cell_threshold = cell_threshold
        self.config(Path(mdir) / Path("config.toml"))

        self.cell_tables_fname = Path(mdir) / Path("cell_tables.pickle")
        if self.cell_tables_fname.exists():
            self.cell_tables = pickle.load(self.cell_tables_fname.open("rb"))
        else:
            self.cell_tables = {}

        self.table_cells_fname = Path(mdir) / Path("table_cells.pickle")
        if self.table_cells_fname.exists():
            self.table_cells = pickle.load(self.table_cells_fname.open("rb"))
        else:
            self.table_cells = {}

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
                txt = cell.get("text").lower()
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
        for cell, tables in matcher.cell_tables.items():
            self.cell_tables.setdefault(cell, set()).update(tables)
        for table, cells in matcher.table_cells.items():
            for cell, ci in cells.items():
                self.table_cells.setdefault(table, {}).setdefault(cell, set()).add(ci)

    def index(self):
        with self.cell_tables_fname.open("wb") as fw:
            pickle.dump(self.cell_tables, fw)
        with self.table_cells_fname.open("wb") as fw:
            pickle.dump(self.table_cells, fw)

    def block(self, ti: int):
        """Block tables on having some cell in common."""
        for cell in self.table_cells.get(ti, []):
            yield from self.cell_tables.get(cell, [])

    def match(self, ti1: int, ti2: int):
        """Match columns on token jaccard."""
        cells1 = self.table_cells.get(ti1, {})
        cells2 = self.table_cells.get(ti2, {})

        #         log.debug(f"{list(cells1)}, {list(cells2)}")
        for cell1, cis1 in cells1.items():
            for cell2, cis2 in cells2.items():
                u = len(cell1 | cell2)
                cell_jacc = len(cell1 & cell2) / u if u else 0
                if cell_jacc > self.cell_threshold:
                    for ci1 in cis1:
                        for ci2 in cis2:
                            yield cell_jacc, ci1, ci2
