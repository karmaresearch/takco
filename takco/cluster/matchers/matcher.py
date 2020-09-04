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

    def config(self, config_fname: Path):
        if config_fname.exists():
            c = toml.load(config_fname.open())
            for k, v in c.items():
                setattr(self, k, v)
        else:
            with config_fname.open("w") as fw:
                toml.dump(self.__dict__, fw)

    def __init__(self, fdir: Path, **kwargs):
        self.indices_fname = Path(fdir) / Path("indices.sqlite")

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

    def add(self, table):
        pass

    def merge(self, matcher):
        """Merge this matcher with another"""
        pass

    def index(self):
        pass

    def close(self):
        pass

    def prepare_block(self, tableIds):
        pass

    def block(self, tableId):
        return set([tableId])

    def match(self, ti1, ti2):
        return int(ti1 == ti2)
