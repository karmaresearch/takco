from pathlib import Path
import logging as log
import pickle
import collections
import subprocess
import time
import shutil
import typing

import toml

from .matcher import Matcher
from .. import cluster

try:
    import datasketch  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except:
    log.error(f"Cannot import datasketch / numpy / pandas")


class LSHMatcher(Matcher):
    """MinHash-based jaccard similarity with LSH blocking"""

    def __init__(
        self,
        fdir=None,
        name=None,
        source="body",
        redis_dir=None,
        basename=None,
        port=6379,
        host="localhost",
        num_perm=128,
        threshold=0.5,
        create=False,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.indexed = False
        self.set_storage(fdir)

        self.source = source
        self.redis_dir = redis_dir if redis_dir and Path(redis_dir).exists() else None
        self.basename = str(fdir or self) if (basename is None) else basename
        self.port = port
        self.host = host
        self.num_perm = num_perm
        self.perm = datasketch.MinHash(num_perm=self.num_perm).permutations
        self.threshold = threshold

        self.lshindex = None
        self.ci_tidi = {}
        self.digests: typing.Any = None
        self.digests_list: typing.List[typing.Any] = []

    def cli(self, command):
        return subprocess.run(
            [self.redis_cli, "-h", self.host, "-p", str(self.port), command],
            capture_output=True,
        )

    def make_lshindex(self):
        if self.redis_dir:
            self.redis_dir = Path(self.redis_dir)
            self.redis_cli = str((self.redis_dir / Path("redis-cli")))

            log.info(f"redis cli: {self.redis_cli}")

            # Start redis if not running
            ping = self.cli("ping")
            if ping.returncode != 0:
                subprocess.run(
                    [
                        self.redis_dir / Path("redis-server"),
                        "--daemonize",
                        "yes",
                        "--port",
                        str(self.port),
                        "--pidfile",
                        Path(f"redis-port-{self.port}.pid"),
                        "--dbfilename",
                        f"redis-{self.name}.rdb",
                    ],
                    cwd=Path(self.mdir or "."),
                )
            while ping.returncode != 0:
                time.sleep(1)
                ping = self.cli("ping")
                log.info(f"Ping redis: {'offline' if ping.returncode else 'online'}")

            return datasketch.MinHashLSH(
                num_perm=self.num_perm,
                threshold=self.threshold,
                storage_config={
                    "type": "redis",
                    "basename": self.basename.encode(),
                    "redis": {"host": self.host, "port": self.port},
                },
            )
        else:
            return datasketch.MinHashLSH(
                num_perm=self.num_perm, threshold=self.threshold,
            )

    def add(self, table):

        rows = []
        if self.source != "head":
            rows += list(
                tuple([cell.get("text", "").lower() for cell in r])
                for r in table["tableData"]
            )
        if self.source != "body":
            rows += list(
                tuple([cell.get("text", "").lower() for cell in r])
                for r in table["tableHeaders"]
            )
        cols = list(zip(*rows))

        if not table.get("numericColumns", []):

            def isnum(col):
                num = lambda x: x.translate(str.maketrans("", "", "-.,%")).isnumeric()
                return sum(int(num(c)) for c in col) / len(col) > 0.5

            table["numericColumns"] = [i for i, c in enumerate(zip(*rows)) if isnum(c)]

        ci_range = range(
            table["columnIndexOffset"], table["columnIndexOffset"] + table["numCols"]
        )
        ti = table["tableIndex"]
        for col, (ci, cells) in enumerate(zip(ci_range, cols)):
            if col not in table.get("numericColumns", []):
                cells = set(c for c in cells if c)
                if len(cells) > 0:
                    m = datasketch.MinHash(
                        num_perm=self.num_perm, permutations=self.perm
                    )
                    for c in cells:
                        m.update(c.encode("utf8"))
                    self.ci_tidi[ci] = (ti, len(self.digests_list))
                    self.digests_list.append(m.digest())

    def merge(self, matcher: "LSHMatcher"):
        assert self.num_perm == matcher.num_perm

        offset = len(self.digests_list)
        self.digests_list += matcher.digests_list
        for ci, (ti, di) in matcher.ci_tidi.items():
            self.ci_tidi[ci] = (ti, di + offset)

        return self

    def index(self):

        self.digests = np.array(self.digests_list)
        del self.digests_list

        self.lshindex = self.make_lshindex()
        with self.lshindex.insertion_session() as session:
            ci_tidis = cluster.progress(self.ci_tidi.items(), f"Indexing {self.name}")
            for ci, (ti, di) in ci_tidis:
                mh = self.digests[di]
                name = f"{ti}-{ci}"
                m = datasketch.MinHash(
                    num_perm=self.num_perm, permutations=self.perm, hashvalues=mh
                )
                session.insert(name, m, check_duplication=False)

        if self.redis_dir:
            r = self.cli("save")
            log.info(f"Saved redis with code {r.returncode}")

        self.indexed = True
        if self.storage:
            digestsdf = (
                pd.DataFrame(self.digests)
                .reset_index()
                .melt(id_vars=["index"], var_name="dim", value_name="val")
            )
            self.storage.save_df(digestsdf, "digests")
            self.storage.save_pickle(self.ci_tidi, "ci_tidi")
            self.storage.save_pickle(self.lshindex, "vi_tici")
            self.close()

    def __enter__(self):
        if self.indexed and self.storage:
            digestsdf = self.storage.load_df("digests")
            self.digests = digestsdf.set_index(["index", "dim"]).unstack().values
            self.ci_tidi = self.storage.load_pickle("ci_tidi")
            self.lshindex = self.storage.load_pickle("lshindex")
        return self

    def load_old(self):
        if self.indexed and self.mdir:
            self.digests = np.load(self.mdir / Path("digests.npy"), mmap_mode="r")
            with open(self.mdir / Path("ci_tidi.pickle"), "rb") as fr:
                self.ci_tidi = pickle.load(fr)
            with open(self.mdir / Path("lshindex.pickle"), "rb") as fr:
                self.lshindex = pickle.load(fr)
        return self

    def close(self):
        if self.indexed and self.storage:
            del self.digests
            del self.ci_tidi
            del self.lshindex

        if self.redis_dir:
            r = self.cli("shutdown")
            log.info(f"Shutdown redis with code {r.returncode}")

    def block(self, ti: int, cis: typing.Collection[int]):
        for ci in cis:
            if ci in self.ci_tidi:
                ti, di = self.ci_tidi[ci]
                mh = self.digests[di]
                m = datasketch.MinHash(
                    num_perm=self.num_perm, permutations=self.perm, hashvalues=mh
                )
                for name in self.lshindex.query(m):
                    ti, _ = map(int, name.split("-", 1))
                    yield ti

    def match(self, tableid_colids_pairs):
        pairs = cluster.progress(tableid_colids_pairs, f"Looking up {self.name}")
        inds, dis1, dis2 = [], [], []
        for (ti1, cis1), (ti2, cis2) in pairs:
            for ci1 in cis1:
                if ci1 in self.ci_tidi:
                    _, di1 = self.ci_tidi[ci1]
                    for ci2 in cis2:
                        if ci2 in self.ci_tidi:
                            _, di2 = self.ci_tidi[ci2]
                            inds.append((ti1, ti2, ci1, ci2))
                            dis1.append(di1)
                            dis2.append(di2)

        if inds:
            log.debug(f"Calculating {len(inds)} {self.name} scores")
            scores = (self.digests[dis1, :] == self.digests[dis2, :]).mean(1)
            inds = cluster.progress(inds, f"Yielding {self.name}")
            for ind, score in zip(inds, scores):
                yield ind, score
