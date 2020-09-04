from .dataset import *
from .t2d import *
from .semtab import *

__all__ = ["Dataset", "Annotation", "T2D", "Semtab", "load"]


from pathlib import Path
import logging as log
import typing


def load(resourcedir: Path = None, datadir: Path = None, **params):
    """Load a dataset from specification"""
    assert "class" in params, f"Dataset specification is missing `class`: {params}"
    classname = params.pop("class")

    resourcedir = resourcedir or Path(".").resolve()
    Path(resourcedir).mkdir(exist_ok=True, parents=True)

    datadir = datadir or resourcedir

    if "path" in params:
        workdir = Path(params["path"])
    else:
        workdir = Path(datadir) / Path(params.get("name", classname))
    workdir.mkdir(exist_ok=True, parents=True)

    if "download" in params:
        from urllib.request import urlretrieve
        from urllib.parse import urlparse

        urls = params.pop("download", [])
        if type(urls) != list:
            urls = [urls]
        for url in urls:
            fname = urlparse(url).path.split("/")[-1]
            fpath = workdir / Path(fname)
            if not fpath.exists():
                log.info(f"Downloading {url} to {fpath}")
                urlretrieve(url, fpath)

                if len(urls) > 1:
                    dpath = fpath.parent / Path(fpath.name.split(".")[0])
                    dpath.mkdir(parents=True, exist_ok=True)
                else:
                    dpath = workdir

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

    if classname in __all__:
        return globals()[classname](workdir, **params)
