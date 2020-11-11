import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import logging as log
import json

from .dataset import Dataset


class WebDataCommons(Dataset):
    def __init__(self, workdir=None, fnames=None, executor=None, exkw=(), **kwargs):
        if executor:
            self.tables = executor.load(fnames, **dict(exkw))
        else:
            if not isinstance(fnames, list):
                fnames = [fnames]
            self.tables = (json.loads(l) for f in fnames for l in open(f))

    def get_unannotated_tables(self):
        if hasattr(self.tables, "pipe"):
            return self.tables.pipe(self.convert)
        else:
            return self.convert(self.tables)

    @staticmethod
    def convert(docs):
        for doc in docs:
            if doc.get("headerPosition") == "FIRST_ROW":
                header, *body = zip(*doc.pop("relation"))
                yield {
                    "_id": "wdc-" + str(abs(hash(str(doc)))),
                    "tbNr": doc.get("tableNum", 0),
                    "pgId": doc.get("url", ""),
                    "pgTitle": doc.get("pageTitle", "").strip() or doc.get("url", ""),
                    "tableCaption": doc.get("title", "").strip(),
                    "tableHeaders": [[{"text": c} for c in header]],
                    "tableData": [[{"text": c} for c in row] for row in body],
                    "numHeaderRows": 1,
                    "numCols": len(header),
                    "numDataRows": len(body),
                    **doc,
                }
