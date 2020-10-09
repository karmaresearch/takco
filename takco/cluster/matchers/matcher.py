from pathlib import Path
import logging as log
import sqlite3
import re

import toml


class Matcher:
    @staticmethod
    def tokenize(text):
        if text.startswith("_"):
            return [text]
        return re.split(r"\W+", text.lower())

    def __init__(self, fdir: Path, **kwargs):
        self.indices_fname = Path(fdir) / Path("indices.sqlite")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def __hash__(self):
        name = getattr(self, "name") if hasattr(self, "name") else ""
        return hash((self.__class__, name))

    def get_table(self, ci):
        with sqlite3.connect(self.indices_fname) as indices:
            r = indices.execute(
                """
                select i from indices
                where (columnIndexOffset <= :1) and (columnIndexOffset + numCols > :1)
            """,
                [int(ci)],
            ).fetchone()
            if r:
                return r[0]
            else:
                log.error(f"Could not find column {ci} in {self.indices_fname}")
                raise KeyError(ci)

    def get_columns(self, ti):
        with sqlite3.connect(self.indices_fname) as indices:
            r = indices.execute(
                """
                 select columnIndexOffset, numCols from indices
                 where (i == ?)
            """,
                [int(ti)],
            ).fetchone()
            if r:
                columnIndexOffset, numCols = r
                return range(columnIndexOffset, columnIndexOffset + numCols)
            else:
                log.error(f"Could not find table {ti} in {self.indices_fname}")
                raise KeyError(ti)

    def get_columns_multi(self, tis):
        tis = list(set(tis))
        size = 999
        with sqlite3.connect(self.indices_fname) as indices:
            for xi in range(0, len(tis), size):
                chunk = tis[xi:(xi+size)]
                rs = indices.execute(
                    f"""
                     select i, columnIndexOffset, numCols from indices
                     where (i in ({', '.join('?' for _ in tis)}))
                """,
                    [int(ti) for ti in tis],
                ).fetchall()
                if rs:
                    for ti, columnIndexOffset, numCols in rs:
                        yield ti, range(columnIndexOffset, columnIndexOffset + numCols)
                else:
                    log.error(f"Could not find table {tis} in {self.indices_fname}")
                    raise KeyError(tis)

    def add(self, table):
        pass

    def merge(self, matcher):
        """Merge this matcher with another"""
        return self

    def index(self):
        pass

    def close(self):
        pass

    def prepare_block(self, tableIds):
        pass

    def block(self, tableId):
        return set([tableId])

    def match(self, table_index_pairs):
        for ti1, ti2 in table_index_pairs:
            for ci1 in self.get_columns(ti1):
                for ci2 in self.get_columns(ti2):
                    yield (ti1, ti2, ci1, ci2), int(ci1 == ci2)
