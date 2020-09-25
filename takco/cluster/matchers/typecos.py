from pathlib import Path
import logging as log
import shutil
import pickle

from .matcher import Matcher


class TypeCosMatcher(Matcher):
    def __init__(
        self, fdir: Path, name=None, create=False, **kwargs,
    ):
        self.name = name or self.__class__.__name__
        mdir = Path(fdir) / Path(self.name)
        if create:
            shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir(parents=True, exist_ok=True)

        self.coltypes_fname = Path(mdir) / Path("coltypes.pickle")
        if self.coltypes_fname.exists():
            self.coltypes = pickle.load(self.coltypes_fname.open("rb"))
        else:
            self.coltypes = {}

        super().__init__(fdir)

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
                    self.coltypes.setdefault(ti, {})[ci] = classes

    def merge(self, matcher: Matcher):
        log.debug(f"merging {self} with {matcher}")
        for ti, ci_classes in matcher.coltypes:
            self.coltypes.setdefault(ti, {}).update(ci_classes)

    def index(self):
        log.debug(f"TypeCos index is len {len(self.coltypes)}")
        with self.coltypes_fname.open("wb") as fw:
            pickle.dump(self.coltypes, fw)

    def match(self, ti1: int, ti2: int):
        """Match columns on token jaccard."""
        ci_classes1 = self.coltypes.get(ti1, {})
        ci_classes2 = self.coltypes.get(ti2, {})

        for ci1, cls1 in ci_classes1.items():
            for ci2, cls2 in ci_classes2.items():
                if ci_classes1 and ci_classes2:
                    dot = lambda a, b: sum((a[k] * b[k]) for k in set(a) & set(b))
                    norm = lambda a: sum(v ** 2 for v in a.values()) ** 0.5
                    cos = dot(cls1, cls2) / (norm(cls1) * norm(cls2))
                    yield max(cos, 0), ci1, ci2
