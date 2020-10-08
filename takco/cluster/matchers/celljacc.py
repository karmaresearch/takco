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
        mdir = Path(fdir) / Path(self.name)
        # if create:
        #     shutil.rmtree(mdir, ignore_errors=True)
        # mdir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.tokenize = tokenize
        # self.config(Path(mdir) / Path("config.toml"))

        self.cell_tables_fname = Path(mdir) / Path("cell_tables.pickle")
        if self.cell_tables_fname.exists():
            pass
            # self.cell_tables = pickle.load(self.cell_tables_fname.open("rb"))
        else:
            self.cell_tables = {}

        self.table_cells_fname = Path(mdir) / Path("table_cells.pickle")
        if self.table_cells_fname.exists():
            pass
            # self.table_cells = pickle.load(self.table_cells_fname.open("rb"))
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
                    self.table_cells.setdefault(ti, {}).setdefault(cell, set()).update(cis)
        return self

    def index(self):
        pass
        # with self.cell_tables_fname.open("wb") as fw:
        #     pickle.dump(self.cell_tables, fw)
        # with self.table_cells_fname.open("wb") as fw:
        #     pickle.dump(self.table_cells, fw)

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
