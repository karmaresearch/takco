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

try:
    import datasketch  # type: ignore
    import numpy as np  # type: ignore
except:
    log.error(f"Cannot import datasketch/numpy")


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
        num_perm=256,
        threshold=0.5,
        create=False,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        self.mdir = (Path(fdir) / Path(self.name)).resolve() if fdir else None
        if self.mdir:
            self.mdir.mkdir(parents=True, exist_ok=True)
        self.indexed = False

        self.source = source
        self.redis_dir = redis_dir if redis_dir and Path(redis_dir).exists() else None
        self.basename = str(fdir or self) if (basename is None) else basename
        self.port = port
        self.host = host
        self.num_perm = num_perm
        self.perm = datasketch.MinHash(num_perm=self.num_perm).permutations
        self.threshold = threshold

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
                        Path(self.mdir or ".") / Path(f"redis-port-{self.port}.pid"),
                        "--dbfilename",
                        f"{self.name}.rdb",
                    ],
                    cwd=Path(self.mdir or "."),
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
                    "redis": {"host": self.host, "port": self.port},
                },
            )
        else:
            self.lshindex = datasketch.MinHashLSH(
                num_perm=self.num_perm, threshold=self.threshold,
            )

        self.ci_tidi = {}
        self.digests: typing.Any = None
        self.digests_list: typing.List[typing.Any] = []

    def cli(self, command):
        return subprocess.run(
            [self.redis_cli, "-h", self.host, "-p", str(self.port), command],
            capture_output=True,
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

        with self.lshindex.insertion_session() as session:
            try:
                import tqdm  # type: ignore

                dis = tqdm.tqdm(self.ci_tidi.items(), desc="Inserting MinHash LSH")
            except:
                dis = self.ci_tidi.items()
            for ci, (ti, di) in dis:
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
        if self.mdir:
            np.save(self.mdir / Path("digests.npy"), self.digests)
            with open(self.mdir / Path("ci_tidi.pickle"), "wb") as fw:
                pickle.dump(self.ci_tidi, fw)
            with open(self.mdir / Path("lshindex.pickle"), "wb") as fw:
                pickle.dump(self.lshindex, fw)
            self.__exit__()

    def __enter__(self):
        super().__enter__()
        if self.indexed and self.mdir:
            self.digests = np.load(self.mdir / Path("digests.npy"), mmap_mode="r")
            with open(self.mdir / Path("ci_tidi.pickle"), "rb") as fr:
                self.ci_tidi = pickle.load(fr)
            with open(self.mdir / Path("lshindex.pickle"), "rb") as fr:
                self.lshindex = pickle.load(fr)
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        if self.indexed and self.mdir:
            del self.digests
            del self.ci_tidi
            del self.lshindex

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

        di_pairs = []
        for (ti1, cis1), (ti2, cis2) in tableid_colids_pairs:
            for ci1 in cis1:
                if ci1 in self.ci_tidi:
                    _, di1 = self.ci_tidi[ci1]
                    for ci2 in cis2:
                        if ci2 in self.ci_tidi:
                            _, di2 = self.ci_tidi[ci2]
                            di_pairs.append(((ti1, ti2, ci1, ci2), di1, di2))

        if di_pairs:
            inds, dis1, dis2 = zip(*di_pairs)
            scores = (self.digests[dis1, :] == self.digests[dis2, :]).mean(1)
            for ind, score in zip(inds, scores):
                yield ind, score

    def close(self):
        if self.redis_dir:
            r = self.cli("shutdown")
            log.info(f"Shutdown redis with code {r.returncode}")
