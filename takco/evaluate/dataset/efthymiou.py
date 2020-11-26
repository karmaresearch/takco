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

    @property
    def tables(self):
        for fname in self.root.joinpath("tables_instance").glob("*.csv"):
            name = fname.stem
            tablefile = html.unescape(self.fix(open(fname).read(), 3))
            rows = list(csv.reader(tablefile.splitlines()))
            mapping_file = self.root.joinpath("entities_instance", fname.name)
            if mapping_file.exists():
                mappings = list(csv.reader(open(mapping_file)))
                if any(mappings):

                    fix_uri = lambda x: urllib.parse.unquote_plus(x)

                    row_uris = {}
                    for row in mappings:
                        keycol = 0  # ??
                        if row:
                            uri, celltext, rownum = row
                            rownum = str(int(rownum))
                            row_uris[rownum] = {fix_uri(uri): 1.0}
                    entities = {str(keycol): row_uris} if row_uris else {}

                    yield {
                        "name": name,
                        "rows": rows,
                        "entities": entities,
                        "keycol": keycol,
                    }
