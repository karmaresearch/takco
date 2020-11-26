from pathlib import Path
import json
import glob
import logging as log
import gzip
import typing


class Dataset:
    tables = None

    def params(self, resourcedir: Path = None, datadir: Path = None, **params):
        params = dict(params)
        classname = self.__class__.__name__

        resourcedir = resourcedir or Path(".").resolve()
        Path(resourcedir).mkdir(exist_ok=True, parents=True)

        datadir = datadir or resourcedir

        if params.get("path"):
            path = Path(params["path"])
        elif datadir:
            path = Path(datadir) / Path(params.get("name", classname))
        else:
            raise Exception(f'No path to put {params.get("name", classname)}!')
        path.mkdir(exist_ok=True, parents=True)
        params["path"] = path

        if "download" in params:
            from urllib.request import urlretrieve
            from urllib.parse import urlparse

            urls = params.pop("download", [])
            if type(urls) != list:
                urls = [urls]
            for url in urls:
                fname = urlparse(url).path.split("/")[-1]
                fpath = path / Path(fname)
                if not fpath.exists():
                    log.info(f"Downloading {url} to {fpath}")
                    urlretrieve(url, fpath)

                    if len(urls) > 1:
                        dpath = fpath.parent / Path(fpath.name.split(".")[0])
                        dpath.mkdir(parents=True, exist_ok=True)
                    else:
                        dpath = path

                    if fpath.name.endswith(".tar.gz"):
                        import tarfile

                        log.info(f"Unpacking {fpath} to {dpath}")
                        tarfile.open(fpath, "r:gz").extractall(dpath)
                    elif fpath.name.endswith(".zip"):
                        import zipfile

                        log.info(f"Unpacking {fpath} to {dpath}")
                        with zipfile.ZipFile(fpath, "r") as zip_ref:
                            zip_ref.extractall(dpath)

        for k, v in params.items():
            if (type(v) == str) and v.endswith("csv"):
                import csv

                vpath = Path(resourcedir) / Path(v)
                if vpath.exists():
                    log.info(f"Loading data from {vpath}")
                    params[k] = list(csv.reader(vpath.open()))
                else:
                    log.warning(f"Could not load data from {vpath}")
                    params[k] = {}

        return params

    def get_unannotated_tables(self):
        for table in self.tables:
            rows = [[{"text": c} for c in row] for row in table.get("rows")]
            headers = [[{"text": c} for c in row] for row in table.get("headers", [])]
            yield {
                "_id": table.get("name", ""),
                "tableData": rows,
                "tableHeaders": headers,
                "keycol": table.get("keycol"),
                "gold": {
                    task: table.get(task, {})
                    for task in ["entities", "classes", "properties"]
                },
            }

    def get_annotated_tables(self):
        return {table["name"]: table for table in self.tables}

    def get_annotated_tables_as_predictions(self):
        for table in self.get_unannotated_tables():
            yield {**table, **table.get("gold", {})}


class Annotation(Dataset):
    def __init__(self, fname: str = None, name=None, **kwargs):
        assert fname
        self.fpath = Path(fname)
        self.name = name or ""

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
