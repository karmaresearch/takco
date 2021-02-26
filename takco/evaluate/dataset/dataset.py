from pathlib import Path
import json
import glob
import logging as log
import gzip
import typing

from takco import Table

class Dataset:
    tables: typing.Sequence[typing.Dict[str, typing.Any]] = ()

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

        return params

    def get_unannotated_tables(self) -> typing.Sequence[Table]:
        for table in self.tables:
            table = dict(table)
            rows = [[{"text": c} for c in row] for row in table.pop("rows", [])]
            headers = [[{"text": c} for c in row] for row in table.pop("headers", [])]
            yield Table(obj={
                "_id": table.pop("name", ""),
                "tableData": rows,
                "tableHeaders": headers,
                "keycol": table.pop("keycol", None),
                "gold": {
                    task: table.pop(task, {})
                    for task in ["entities", "classes", "properties"]
                },
                **table,
            })

    def get_annotated_tables(self) -> typing.Sequence[Table]:
        return Table({table["name"]: table for table in self.tables})

    def get_annotated_tables_as_predictions(self):
        for table in self.get_unannotated_tables():
            table.annotations = table.get("gold", {})
            yield table
