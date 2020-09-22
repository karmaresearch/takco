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
    def __init__(
        self,
        fdir,
        source="body",
        redis_dir=None,
        basename=None,
        port=6379,
        num_perm=256,
        threshold=0.5,
        create=False,
        **kwargs,
    ):
        """MinHash-based jaccard similarity with LSH blocking"""

        mdir = Path(fdir) / Path("LSHMatcher")
        if create:
            shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.redis_dir = redis_dir if Path(redis_dir).exists() else None
        self.basename = str(fdir) if (basename is None) else basename
        self.port = port
        self.num_perm = num_perm
        self.threshold = threshold
        self.config(Path(mdir) / Path("config.toml"))

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
                num_perm=self.num_perm,
                threshold=self.threshold,
            )
            self.session = None

        self.minhashes_fname = Path(mdir) / Path("minhashes.npy")
        if self.minhashes_fname.exists():
            self.minhash = np.load(self.minhashes_fname, mmap_mode="r")
        else:
            self.minhash = None

        self.column_ids_fname = Path(mdir) / Path("column_ids.npy")
        if self.column_ids_fname.exists():
            self.ci_digest = collections.OrderedDict(
                (k, v) for v, k in enumerate(np.load(self.column_ids_fname))
            )
        else:
            self.ci_digest = collections.OrderedDict()

        self.digests = []
        super().__init__(fdir)

    def cli(self, command):
        return subprocess.run(
            [self.redis_cli, "-p", str(self.port), command],
            capture_output=True,
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
        self.session.close()
        self.digests += matcher.digests
        self.ci_digest.update(matcher.ci_digest)
        matcher.session.close()

    def index(self):
        self.minhash = np.array(self.digests)
        np.save(self.minhashes_fname, self.minhash)

        self.ci_digest = collections.OrderedDict(
            zip(self.ci_digest.keys(), range(len(self.digests)))
        )
        np.save(self.column_ids_fname, np.array(list(self.ci_digest.keys())))

        if self.redis_dir:
            r = self.cli("save")
            log.info(f"Saved redis with code {r.returncode}")

    def block(self, ti: int):
        for ci in self.get_columns(ti):
            if ci in self.ci_digest:
                mh = self.minhash[self.ci_digest[ci]]
                m = datasketch.MinHash(num_perm=self.num_perm, hashvalues=mh)
                for ci in self.lshindex.query(m):
                    yield self.get_table(int(ci))

    def match(self, ti1: int, ti2: int):
        for ci1 in self.get_columns(ti1):
            for ci2 in self.get_columns(ti2):
                if (ci1 in self.ci_digest) and (ci2 in self.ci_digest):
                    mh1 = self.minhash[self.ci_digest[ci1]]
                    mh2 = self.minhash[self.ci_digest[ci2]]
                    yield np.mean((mh1 == mh2)), ci1, ci2

    def close(self):
        if self.redis_dir:
            r = self.cli("shutdown")
            log.info(f"Shutdown redis with code {r.returncode}")
