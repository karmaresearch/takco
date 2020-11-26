import os
from pathlib import Path
import pickle
import logging as log
from dataclasses import dataclass, field
from typing import Optional


class HDFSPath(str):
    def __new__(self, val):
        assert val.startswith("hdfs://")
        return super().__new__(self, val)  # type: ignore


@dataclass
class Storage:
    root: str
    name: Optional[str] = None
    is_hdfs: bool = False

    def __init__(self, fdir, name=None, is_hdfs=False):
        if fdir.startswith("hdfs://"):
            self.is_hdfs = True
            fdir = fdir.replace("hdfs://", "")
        else:
            self.is_hdfs = False
        self.root = os.path.join(str(fdir), name) if name else str(fdir)
        self.name = name

    def mkdir(self):
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            fs.mkdir(self.root)
        else:
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def rmtree(self):
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            fs.delete(self.root, recursive=True)
        else:
            import shutil

            shutil.rmtree(self.root, ignore_errors=True)

    def rm(self, fname):
        path = os.path.join(self.root, fname)
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            return fs.delete(path)
        else:
            return os.remove(path)

    def ls(self):
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            return fs.ls(self.root)
        else:
            return os.listdir(self.root)

    def load_df(self, name):
        fname = f"{name}.parquet"
        path = os.path.join(self.root, fname)
        if self.is_hdfs:
            import pyarrow as pa
            import pyarrow.parquet as pq

            log.debug(f"Loading table {path} from HDFS")
            fs = pa.hdfs.connect()
            with fs.open(path) as f:
                return pq.read_table(f).to_pandas()
        else:
            import pandas as pd

            log.debug(f"Loading table {path} from disk")
            return pd.read_parquet(path)

    def save_df(self, df, name):
        fname = f"{name}.parquet"
        path = os.path.join(self.root, fname)
        if self.is_hdfs:
            import pyarrow as pa
            import pyarrow.parquet as pq

            fs = pa.hdfs.connect()
            fs.mkdir(self.root)
            log.debug(f"Saving table {path} to HDFS")
            with fs.open(path, "wb") as fw:
                pq.write_table(pa.Table.from_pandas(df), fw)
        else:
            Path(self.root).mkdir(parents=True, exist_ok=True)
            log.debug(f"Saving table {path} to disk")
            df.to_parquet(path)

    def load_pickle(self, name):
        fname = f"{name}.pickle"
        path = os.path.join(self.root, fname)
        if self.is_hdfs:
            import pyarrow as pa

            log.debug(f"Loading object {path} from HDFS")
            fs = pa.hdfs.connect()
            with fs.open(path) as f:
                return pickle.loads(f.read())
        else:
            log.debug(f"Loading object {path} from disk")
            with open(path, "rb") as f:
                return pickle.loads(f.read())

    def save_pickle(self, obj, name):
        fname = f"{name}.pickle"
        path = os.path.join(self.root, fname)
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            fs.mkdir(self.root)
            log.debug(f"Saving object {path} to HDFS")
            with fs.open(path, "wb") as fw:
                fw.write(pickle.dumps(obj))
        else:
            Path(self.root).mkdir(parents=True, exist_ok=True)
            log.debug(f"Saving object {path} to disk")
            with open(path, "wb") as fw:
                pickle.dump(obj, fw)

    def exists(self, fname=None):
        path = os.path.join(self.root, fname) if fname else self.root
        if self.is_hdfs:
            import pyarrow as pa

            fs = pa.hdfs.connect()
            return fs.exists(path)
        else:
            return Path(path).exists()

    def exists_df(self, name):
        return self.exists(f"{name}.parquet")

    def exists_pickle(self, name):
        return self.exists(f"{name}.pickle")
