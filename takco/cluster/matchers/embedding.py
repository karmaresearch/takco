from pathlib import Path
import logging as log
import collections
import shutil

from .matcher import Matcher


class EmbeddingMatcher(Matcher):
    """Blocking matcher using table cells from header or body"""

    def __init__(
        self,
        fdir: Path,
        source="body",
        wordvec_fname: Path = None,
        topn=100,
        exp=1,
        threshold=0,
        create=False,
        **kwargs,
    ):
        """Matcher based on embeddings and FAISS"""

        import faiss
        import pandas as pd
        import numpy as np

        mdir = Path(fdir) / Path("EmbeddingMatcher")
        if create:
            shutil.rmtree(mdir, ignore_errors=True)
        mdir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.wordvec_fname = str(wordvec_fname)
        self.topn = topn
        self.exp = exp
        self.threshold = threshold
        self.config(Path(mdir) / Path("config.toml"))
        self.wordvec_fname = Path(self.wordvec_fname)

        self.faissindex_fname = Path(mdir) / Path("index.faiss")
        if self.faissindex_fname.exists():
            self.faissindex = faiss.read_index(str(self.faissindex_fname))

        self.means_fname = Path(mdir) / Path("means.npy")
        if self.means_fname.exists():
            self.means = np.load(self.means_fname, mmap_mode="r")

        self.column_ids_fname = Path(mdir) / Path("column_ids.npy")
        if self.column_ids_fname.exists():
            self.ci_vec = collections.OrderedDict(
                (k, v) for v, k in enumerate(np.load(self.column_ids_fname))
            )
        else:
            self.ci_vec = collections.OrderedDict()
        self.vec_ci = collections.OrderedDict((v, k) for k, v in self.ci_vec.items())

        log.info(f"Loading word vectors {self.wordvec_fname}")
        vec = pd.read_csv(
            self.wordvec_fname, delimiter=" ", quoting=3, header=None, index_col=0
        )
        self.word_i = {w: i for i, w in enumerate(vec.index)}
        self.wordvecarray = np.array(vec)

        self.vecs = []
        self.ti_block = {}
        super().__init__(fdir)

    def add(self, table):
        # Extract tokens from columns and create embeddings
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
                    cell_locs = [
                        [
                            self.word_i[w]
                            for w in Matcher.tokenize(cell)
                            if (len(w) > 1) and (w in self.word_i)
                        ]
                        for cell in cells
                    ]
                    if any(any(locs) for locs in cell_locs):
                        cell_vecs = [
                            self.wordvecarray[[l for l in locs]].sum(axis=0)
                            for locs in cell_locs
                        ]
                        cell_vecs = np.vstack(cell_vecs)
                        self.ci_vec[ci] = len(self.vecs)
                        self.vecs.append(cell_vecs.mean(axis=0))

    def merge(self, matcher: Matcher):
        self.vecs += matcher.vecs
        self.ci_vec.update(matcher.ci_vec)

    def index(self):
        self.means = np.array(self.vecs).astype("float32")
        np.save(self.means_fname, self.means)

        self.ci_vec = collections.OrderedDict(
            zip(self.ci_vec.values(), range(len(self.vecs)))
        )
        np.save(self.column_ids_fname, np.array(list(self.ci_vec.keys())))
        self.vec_ci = collections.OrderedDict((v, k) for k, v in self.ci_vec.items())

        # Create FAISS index
        V = np.ascontiguousarray(self.means)
        V /= np.sqrt((V ** 2).sum(axis=1))[:, None]

        index = faiss.IndexFlatIP(V.shape[1])  # build the index
        index.add(np.array(V))  # add vectors to the index

        log.debug("faiss info: %s", faiss.MatrixStats(V).comments)
        log.info(f"Writing faiss index to {self.faissindex_fname}")
        faiss.write_index(index, str(self.faissindex_fname))

    def prepare_block(self, tis):
        ci_ti = {ci: ti for ti in tis for ci in self.get_columns(ti)}
        qi_mean, ci_qi = [], {}
        for ci in ci_ti:
            if ci in self.ci_vec:
                ci_qi[ci] = len(qi_mean)
                qi_mean.append(self.means[self.ci_vec[ci]])

        if not len(qi_mean):
            log.warning(f"No column embeddings found for any of {len(tis)} tables!")
        else:
            xq = np.vstack(qi_mean)  # query vectors
            xq /= np.sqrt((xq ** 2).sum(axis=1))[:, None]  # L2 normalize
            xq = xq.astype("float32")
            log.debug(f"Querying faiss index with query matrix of shape {xq.shape}")
            D, I = self.faissindex.search(xq, self.topn)

            for ci1, qi in ci_qi.items():
                indexes, similarities = I[qi], (0.5 + (D[qi] / 2)) ** self.exp
                ti1 = ci_ti[ci1]
                for qi2 in set(indexes[(similarities > self.threshold)]) - set([ci1]):
                    if qi2 in self.vec_ci:
                        ti2 = self.get_table(self.vec_ci[qi2])
                        if ti2 != ti1:
                            self.ti_block.setdefault(ti1, set()).add(ti2)

    def block(self, ti: int):
        return self.ti_block.get(ti, set())

    def match(self, ti1: int, ti2: int):
        for ci1 in self.get_columns(ti1):
            for ci2 in self.get_columns(ti2):
                if (ci1 in self.ci_vec) and (ci2 in self.ci_vec):
                    vi1, vi2 = self.ci_vec[ci1], self.ci_vec[ci2]
                    a1, a2 = self.means[vi1], self.means[vi2]
                    c = 0.5 + (
                        (
                            a1.dot(a2)
                            / (np.sqrt((a1 ** 2).sum()) * np.sqrt((a2 ** 2).sum()))
                        )
                        / 2
                    )
                    c = c ** self.exp
                    yield c, ci1, ci2
