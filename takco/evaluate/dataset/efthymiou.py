import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import logging as log
import csv
import html
import urllib
import tarfile, zipfile, io

from .dataset import Dataset


class LimayeGS(Dataset):
    def __init__(
        self, datadir=None, resourcedir=None, path=None, **kwargs,
    ):
        kwargs = self.params(
            path=path, datadir=datadir, resourcedir=resourcedir, **kwargs
        )
        path = Path(kwargs.get("path", "."))
        self.root = path.joinpath("LimayeGS")

    @classmethod
    def fix(cls, s, depth=1):
        if depth > 0:
            s = s.encode("latin1", errors="ignore").decode("utf8", errors="ignore")
            return cls.fix(s, depth - 1)
        return s

    @classmethod
    def mapping_entities(cls, rows, mappings, fname):
        fix_uri = lambda x: urllib.parse.unquote_plus(x)

        entities = {}
        for row in mappings:
            if row:
                uri, celltext, rownum = row
                celltext = html.unescape(cls.fix(celltext, 3))
                ri = int(rownum)
                if int(rownum) >= len(rows):
                    continue
                c_ci = {c: ci for ci, c in enumerate(rows[ri])}
                if celltext not in c_ci:
                    log.warn(f"{celltext} not found in {rows[ri]} in {fname}")
                    log.warn(str(rows))
                    continue
                ci = c_ci[celltext]
                entities.setdefault(str(ci), {})[str(ri)] = {fix_uri(uri): 1.0}
        return entities

    @property
    def tables(self):
        for fname in self.root.joinpath("tables_instance").glob("*.csv"):
            name = fname.stem
            file_rows = csv.reader(open(fname))
            rows = [[html.unescape(self.fix(c, 3)) for c in r] for r in file_rows]
            mapping_path = self.root.joinpath("entities_instance", fname.name)
            if mapping_path.exists():
                mappings = list(csv.reader((open(mapping_path))))
                if any(mappings):
                    entities = self.mapping_entities(rows, mappings, fname) or {}

                    yield {
                        "name": name,
                        "headers": [],
                        "rows": rows,
                        "entities": entities,
                        "keycol": None,
                    }
