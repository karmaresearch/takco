import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import logging as log
import json
import urllib

from .dataset import Dataset
from takco import Table

class WebDataCommons(Dataset):
    """A collection of tables from the WebDataCommons project (http://webdatacommons.org/webtables/)

    Args:
        fnames: Filenames of json files.
    """
    def __init__(self, workdir=None, fnames=None, executor=None, exkw=(), **kwargs):
        if not isinstance(fnames, list):
            fnames = [fnames]
        self.fnames = fnames
        self.executor = executor
        self.exkw = exkw

    @property
    def tables(self):
        if self.executor:
            return self.executor.load(self.fnames, **dict(self.exkw))
        else:
            for f in self.fnames:
                for l in open(f):
                    d = json.loads(l)
                    d['fname'] = Path(str(f)).name
                    yield d

    def get_unannotated_tables(self):
        if hasattr(self.tables, "pipe"):
            return self.tables.pipe(self.convert)
        else:
            return self.convert(self.tables)

    @staticmethod
    def convert(docs):
        for doc in docs:
            if 'table' in doc:
                if 'fname' in doc:
                    doc['table']['fname'] = doc['fname']
                doc = doc['table']
            
            if doc.get("headerPosition") == "FIRST_ROW":
                header, *body = zip(*doc.pop("relation"))
                if "url" in doc:
                    doc['domain'] = urllib.parse.urlparse(doc["url"]).netloc

                if 'fname' in doc:
                    _id = doc['fname']
                else:
                    _id = "wdc-" + str(abs(hash(str(doc))))

                yield Table({
                    "_id": _id,
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
                }, linked=False)
    
    @staticmethod
    def convert_back(table, snow=False):
        doc = {
            'relation': [
                row
                for row in zip(*(table.head + table.body))
            ],
            'hasHeader': True,
            'headerPosition': 'FIRST_ROW',
            'tableType': 'RELATION',
            'tableNum': 0,
            'recordEndOffset': 0,
            'recordOffset': 0,
            'tableOrientation': 'HORIZONTAL',
        }
        if snow:
            doc.update({
                'functionalDependencies': [
                    # {
                    #     'determinant': [1],
                    #     'dependant': [0],
                    #     'probability': 1.0,
                    # }
                ],
                # 'candidateKeys': [[1]],
            })
            
            doc = {
                'table': doc,
                'mapping': {
                    # 'numHeaderRows': 0,
                    # 'mappedProperties': {},
                    # 'mappedInstances': {},
                    # 'keyIndex': 0,
                    # 'dataTypes': {
                    #     '0': 'numeric',
                    #     '1': 'string',
                    # },
                }
            }
        return doc