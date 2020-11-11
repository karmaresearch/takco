from pathlib import Path
import typing
import logging as log
import shutil
import pickle

from .matcher import Matcher
from .. import cluster


class TypeCosMatcher(Matcher):
    def __init__(
        self,
        fdir: Path = None,
        name=None,
        create=False,
        exclude_types=["https://www.w3.org/2001/XMLSchema#string"],
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.mdir = (Path(fdir) / Path(self.name)) if fdir else None

        if self.mdir:

            self.coltypes_fname = Path(self.mdir) / Path("coltypes.pickle")
        self.coltypes: typing.Dict[int, typing.Any] = {}
        self.exclude_types = exclude_types

        self.indexed = False

    def add(self, table):
        if table:
            ti = table["tableIndex"]
            ci_range = range(
                table["columnIndexOffset"],
                table["columnIndexOffset"] + table["numCols"],
            )
            for ci, c in zip(ci_range, range(table["numCols"])):
                classes = table.get("classes", {}).get(str(c))
                if classes:
                    classes = {
                        k: v for k, v in classes.items() if k not in self.exclude_types
                    }
                    norm = sum(v ** 2 for v in classes.values()) ** 0.5
                    self.coltypes.setdefault(ti, {})[ci] = (classes, norm)

    def merge(self, matcher: "TypeCosMatcher"):
        if matcher is not None:
            log.debug(f"merging {self} with {matcher}")
            for ti, ci_classes in matcher.coltypes.items():
                self.coltypes.setdefault(ti, {}).update(ci_classes)
        return self

    def __enter__(self):
        super().__enter__()
        if self.indexed and self.mdir:
            self.coltypes = pickle.load(self.coltypes_fname.open("rb"))
        return self

    def close(self):
        if self.indexed and self.mdir:
            del self.coltypes

    def index(self):
        log.debug(f"TypeCos index is len {len(self.coltypes)}")
        if self.mdir:
            log.debug(f"Serializing {self} to {self.mdir}")
            self.mdir.mkdir(parents=True, exist_ok=True)
            with self.coltypes_fname.open("wb") as fw:
                pickle.dump(self.coltypes, fw)
            self.indexed = True
            self.close()

    def match(self, tableid_colids_pairs):
        """Match columns on token jaccard."""
        pairs = cluster.progress(tableid_colids_pairs, f"Looking up {self.name}")
        for (ti1, _), (ti2, _) in pairs:
            ci_classes1 = self.coltypes.get(ti1, {})
            ci_classes2 = self.coltypes.get(ti2, {})

            for ci1, (cls1, n1) in ci_classes1.items():
                for ci2, (cls2, n2) in ci_classes2.items():
                    dot = lambda a, b: sum((a[k] * b[k]) for k in set(a) & set(b))
                    cos = dot(cls1, cls2) / (n1 * n2)
                    yield (ti1, ti2, ci1, ci2), max(cos, 0)
