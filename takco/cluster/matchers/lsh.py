from pathlib import Path
import logging as log
import pickle
import collections
import subprocess
import time
import shutil

import toml

from .matcher import Matcher

try:
    import datasketch
    import numpy as np
except:
    log.error(f"Cannot import datasketch/numpy")


class LSHMatcher(Matcher):
    """MinHash-based jaccard similarity with LSH blocking"""

    def __init__(
        self,
        fdir,
        name=None,
        source="body",
        redis_dir=None,
        basename=None,
        port=6379,
        num_perm=256,
        threshold=0.5,
        create=False,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.mdir = Path(fdir) / Path(self.name)
        self.indexed = False

        self.source = source
        self.redis_dir = redis_dir if redis_dir and Path(redis_dir).exists() else None
        self.basename = str(fdir) if (basename is None) else basename
        self.port = port
        self.num_perm = num_perm
        self.threshold = threshold
        # self.config(Path(mdir) / Path("config.toml"))

        if self.redis_dir:
            self.redis_dir = Path(self.redis_dir)
            self.redis_cli = str((self.redis_dir / Path("redis-cli")).resolve())

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
                        Path(mdir) / Path(f"{self.port}.pid"),
                        "--dbfilename",
                        "dump.rdb",
                    ]
                )
            while ping.returncode != 0:
                time.sleep(1)
                ping = self.cli("ping")
                log.info(f"Ping redis: {'offline' if ping.returncode else 'online'}")

            self.lshindex = datasketch.MinHashLSH(
                num_perm=self.num_perm,
                threshold=self.threshold,
                storage_config={
                    "type": "redis",
                    "basename": self.basename.encode(),
                    "redis": {"host": "localhost", "port": self.port},
                },
            )
            self.session = None
        else:
            self.lshindex = datasketch.MinHashLSH(
                num_perm=self.num_perm, threshold=self.threshold,
            )
            self.session = None

        self.minhash = None
        self.ci_digest = collections.OrderedDict()
        self.digests = []
        super().__init__(fdir)

    def cli(self, command):
        return subprocess.run(
            [self.redis_cli, "-p", str(self.port), command], capture_output=True,
        )

    def add(self, table):
        if self.session is None:
            self.session = self.lshindex.insertion_session()

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
        for col, (ci, cells) in enumerate(zip(ci_range, cols)):
            if col not in table.get("numericColumns", []):
                cells = set(c for c in cells if c)
                if len(cells) > 0:
                    name = str(ci)

                    m = datasketch.MinHash(num_perm=self.num_perm)
                    for c in cells:
                        m.update(c.encode("utf8"))
                    self.ci_digest[ci] = len(self.digests)
                    self.digests.append(m.digest())
                    self.session.insert(name, m, check_duplication=False)

    def merge(self, matcher: Matcher):
        if matcher is not None:
            matcher.session.close()
            assert self.num_perm == matcher.num_perm

            if self.session is None:
                self.session = self.lshindex.insertion_session()
            for ci, di in matcher.ci_digest.items():
                mh = matcher.digests[di]
                name = str(ci)
                m = datasketch.MinHash(num_perm=self.num_perm, hashvalues=mh)
                self.session.insert(name, m, check_duplication=False)

            self.digests += matcher.digests
            self.ci_digest.update(matcher.ci_digest)

        return self

    def index(self):
        self.session.close()
        del self.session

        self.minhash = np.array(self.digests)
        del self.digests

        self.ci_digest = collections.OrderedDict(
            zip(self.ci_digest.keys(), range(len(self.minhash)))
        )

        if self.redis_dir:
            r = self.cli("save")
            log.info(f"Saved redis with code {r.returncode}")

        self.indexed = True
        self.mdir.mkdir(parents=True, exist_ok=True)
        np.save(self.mdir / Path("minhash.npy"), self.minhash)
        np.save(
            self.mdir / Path("ci_digest.npy"), np.array(list(self.ci_digest.keys()))
        )
        with (self.mdir / Path("lshindex.pickle")).open("wb") as fw:
            pickle.dump(self.lshindex, fw)
        self.__exit__()

    def __enter__(self):
        if self.indexed:
            self.minhash = np.load(self.mdir / Path("minhash.npy"), mmap_mode="r")
            self.ci_digest = collections.OrderedDict(
                (k, v) for v, k in enumerate(np.load(self.mdir / Path("ci_digest.npy")))
            )
            with (self.mdir / Path("lshindex.pickle")).open("rb") as fr:
                self.lshindex = pickle.load(fr)
        return self

    def __exit__(self, *args):
        if self.indexed:
            del self.minhash
            del self.ci_digest
            del self.lshindex

    def block(self, ti: int):
        for ci in self.get_columns(ti):
            if ci in self.ci_digest:
                mh = self.minhash[self.ci_digest[ci]]
                m = datasketch.MinHash(num_perm=self.num_perm, hashvalues=mh)
                for ci in self.lshindex.query(m):
                    yield self.get_table(int(ci))

    def match(self, table_index_pairs):
        tis = set(ti for pair in table_index_pairs for ti in pair)
        ti_dis = {}
        for ti, cs in self.get_columns_multi(tis):
            ti_dis[ti] = [(ci, self.ci_digest[ci]) for ci in cs if ci in self.ci_digest]

        di_pairs = []
        for ti1, ti2 in table_index_pairs:
            for ci1, di1 in ti_dis.get(ti1, []):
                for ci2, di2 in ti_dis.get(ti2, []):
                    di_pairs.append(((ti1, ti2, ci1, ci2), di1, di2))

        if di_pairs:
            inds, dis1, dis2 = zip(*di_pairs)
            scores = (self.minhash[dis1, :] == self.minhash[dis2, :]).mean(1)
            for ind, score in zip(inds, scores):
                yield ind, score

    def close(self):
        if self.redis_dir:
            r = self.cli("shutdown")
            log.info(f"Shutdown redis with code {r.returncode}")
