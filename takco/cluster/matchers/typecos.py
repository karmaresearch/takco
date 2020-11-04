from pathlib import Path
import typing
import logging as log
import shutil
import pickle

from .matcher import Matcher


class TypeCosMatcher(Matcher):
    def __init__(
        self, fdir: Path = None, name=None, create=False, **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.mdir = (Path(fdir) / Path(self.name)).resolve() if fdir else None

        if self.mdir:
            self.coltypes_fname = Path(self.mdir) / Path("coltypes.pickle")
        self.coltypes: typing.Dict[int, typing.Any] = {}

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

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.indexed and self.mdir:
            del self.coltypes

    def index(self):
        log.debug(f"TypeCos index is len {len(self.coltypes)}")
        if self.mdir:
            with self.coltypes_fname.open("wb") as fw:
                pickle.dump(self.coltypes, fw)
            del self.coltypes
            self.indexed = True

    def match(self, tableid_colids_pairs):
        """Match columns on token jaccard."""
        for (ti1, _), (ti2, _) in tableid_colids_pairs:
            ci_classes1 = self.coltypes.get(ti1, {})
            ci_classes2 = self.coltypes.get(ti2, {})

            for ci1, (cls1, n1) in ci_classes1.items():
                for ci2, (cls2, n2) in ci_classes2.items():
                    dot = lambda a, b: sum((a[k] * b[k]) for k in set(a) & set(b))
                    cos = dot(cls1, cls2) / (n1 * n2)
                    yield (ti1, ti2, ci1, ci2), max(cos, 0)
