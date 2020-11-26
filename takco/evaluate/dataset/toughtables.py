import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List
import logging as log
import csv
import html
import urllib
import tarfile, zipfile, io

from .dataset import Dataset


class ToughTables(Dataset):
    def __init__(
        self, datadir=None, resourcedir=None, path=None, part=None, **kwargs,
    ):
        kwargs = self.params(
            path=path, datadir=datadir, resourcedir=resourcedir, **kwargs
        )
        path = Path(kwargs.get("path", "."))
        if not part:
            raise Exception(f"You must supply `part`")
        assert part in ["2T", "2T_WD"]
        self.part = part
        self.root = path.joinpath(self.part)

    def iter_gt(self, fname):
        chunkname = None
        chunk: List[List[str]] = []
        with open(fname) as f:
            for row in csv.reader(f):
                try:
                    if row[0] != chunkname and chunk:
                        yield chunkname, chunk
                        chunkname, chunk = row[0], []
                    chunkname = row[0]
                    chunk.append(row[1:])
                except:
                    pass
        if chunk:
            yield chunkname, chunk

    @property
    def tables(self):
        classes_gt = dict(
            self.iter_gt(self.root.joinpath("gt", f"CTA_{self.part}_gt.csv"))
        )
        for name, ents_gt in self.iter_gt(
            self.root.joinpath("gt", f"CEA_{self.part}_gt.csv")
        ):
            rows = list(csv.reader(open(self.root.joinpath("tables", f"{name}.csv"))))

            entities = {}  # type: ignore
            for ci, ri, ents in ents_gt:
                if self.part == "2T_WD":
                    # in the Wikidata dataset, row and column indices are switched!
                    ci, ri = ri, ci

                ci, ri = str(ci), str(int(ri) - 1)
                entities.setdefault(ci, {})[ri] = {e: 1 for e in ents.split()}

            classes = {}  # type: ignore
            for ci, ents in classes_gt.get(name, []):
                ci = str(ci)
                classes[ci] = {e: 1 for e in ents.split()}

            yield {
                "name": name,
                "headers": rows[:1],
                "rows": rows[1:],
                "entities": entities,
                "classes": classes,
            }
