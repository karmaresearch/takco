from pathlib import Path
import json
import glob
import logging as log
import gzip


class Dataset:
    pass


class Annotation(Dataset):
    def __init__(self, root: Path, fname: str = None, name=None, **kwargs):
        self.fpath = Path(root) / Path(fname) if fname else Path(root)
        self.name = name or ""

    def get_unannotated_tables(self):
        return self.tables

    @property
    def tables(self):
        files = []

        if self.fpath.is_dir():
            files = self.fpath.glob("*")
        elif self.fpath.is_file():
            files = [self.fpath]
        else:
            log.error(f"Cannot load {self.fpath}")

        for file in files:
            gzipped = file.name.endswith("gz")
            with (gzip.open(file) if gzipped else file.open()) as f:
                for line in f:
                    table = json.loads(line)
                    table["name"] = table["_id"]
                    table["rows"] = [
                        [c.get("text", "") for c in row] for row in table["tableData"]
                    ]
                    if "tableHeaders" in table:
                        table["headers"] = [
                            [c.get("text", "") for c in row]
                            for row in table["tableHeaders"]
                        ]
                    table["numheaderrows"] = len(table["tableHeaders"])
                    table["keycol"] = 0
                    table.setdefault(
                        "entities",
                        {
                            str(ci): {
                                str(ri): {
                                    uri: 1
                                    for l in c.get("surfaceLinks", [])
                                    for uri in [l.get("target", {}).get("href")]
                                    if uri
                                }
                                for ri, c in enumerate(col)
                            }
                            for ci, col in enumerate(zip(*table["tableData"]))
                        },
                    )
                    table.setdefault("classes", {})
                    table.setdefault("properties", {})
                    yield table
