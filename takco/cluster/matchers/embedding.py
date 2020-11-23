from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import logging as log
import collections
import shutil
import pickle
import typing

from .matcher import Matcher, default_tokenize
from .. import cluster

try:
    import faiss  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
except:
    log.error(f"Cannot import faiss/pandas/numpy")


class EmbeddingMatcher(Matcher):
    """Blocking matcher using table cells from header or body"""

    def __init__(
        self,
        fdir: Path = None,
        wordvec_fname: Path = None,
        name=None,
        source="body",
        tokenize=default_tokenize,
        topn=100,
        threshold=0.9,
        create=False,
        **kwargs,
    ):
        """Matcher based on embeddings and FAISS"""
        self.name = name or self.__class__.__name__
        self.indexed = False
        self.set_storage(fdir)

        self.source = source
        self.topn = topn
        self.threshold = threshold
        self.wordvec_fname = Path(wordvec_fname) if wordvec_fname else None
        self.tokenize = tokenize

        self.vi_tici: List[Tuple[int, int]] = []
        self.ci_vi: Dict[int, int] = {}

        self.vecs: List[Any] = []
        self.ti_block: Dict[int, Set[int]] = {}

    def add(self, table):
        # Extract tokens from columns and create embeddings
        rows = []
        if self.source != "head":
            rows += [
                tuple([cell.get("text", "").lower() for cell in r])
                for r in table["tableData"]
            ]
        if self.source != "body":
            rows += [
                tuple([cell.get("text", "").lower() for cell in r])
                for r in table["tableHeaders"]
            ]
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
        for colnum, (ci, cells) in enumerate(zip(ci_range, cols)):
            if colnum not in table.get("numericColumns", []):
                cells = set(c for c in cells if c)
                if len(cells) > 0:
                    cell_vec = self.get_vec(cells)
                    if cell_vec is not None:
                        self.vi_tici.append((ti, ci))
                        self.vecs.append(cell_vec)

    def get_vec(self, cells):
        cell_locs = [
            [
                self.word_i[w]
                for w in self.tokenize(cell)
                if (len(w) > 1) and (w in self.word_i)
            ]
            for cell in cells
        ]
        if any(any(locs) for locs in cell_locs):
            cell_vecs = [
                self.wordvecarray[[l for l in locs]].sum(axis=0) for locs in cell_locs
            ]
            return np.vstack(cell_vecs).mean(axis=0)

    def merge(self, matcher: "EmbeddingMatcher"):
        if matcher is not None:
            self.vecs += matcher.vecs
            self.vi_tici += matcher.vi_tici
        return self

    def index(self):
        if self.indexed:
            return

        self.means = np.array(self.vecs).astype("float32")
        del self.vecs
        self.means = np.ascontiguousarray(self.means)
        self.means /= np.sqrt((self.means ** 2).sum(axis=1))[:, None]

        self.ci_vi = {ci: vi for vi, (_, ci) in enumerate(self.vi_tici)}

        if self.storage:
            meansdf = (
                pd.DataFrame(self.means)
                .reset_index()
                .melt(id_vars=["index"], var_name="dim", value_name="val")
            )
            self.storage.save_df(meansdf, "means")
            self.storage.save_pickle(self.vi_tici, "vi_tici")
            self.close()
        else:
            # Create FAISS index
            faissindex = faiss.IndexFlatIP(self.means.shape[1])  # build the index
            # add vectors to the index
            faissindex.add(np.array(self.means))  # type: ignore
            log.debug("faiss info: %s", faiss.MatrixStats(self.means).comments)
            self.faissindex = faissindex

        self.indexed = True

    def __enter__(self):
        log.info(f"Loading word vectors {self.wordvec_fname}")
        if str(self.wordvec_fname).endswith(".pickle"):
            wordvecs = pd.read_pickle(self.wordvec_fname)
        else:
            wordvecs = pd.read_csv(
                self.wordvec_fname, delimiter=" ", quoting=3, header=None, index_col=0
            )

        self.word_i = {w: i for i, w in enumerate(wordvecs.index)}
        self.wordvecarray = np.array(wordvecs)

        if self.indexed and self.storage:
            meansdf = self.storage.load_df("means")
            self.means = meansdf.set_index(["index", "dim"]).unstack().values
            self.vi_tici = self.storage.load_pickle("vi_tici")
            self.ci_vi = {ci: vi for vi, (_, ci) in enumerate(self.vi_tici)}

        return self

    def load_old(self):
        if self.indexed and self.mdir:
            log.debug(f"Loading {self} from disk...")
            self.means = np.load(self.mdir / Path("means.npy"), mmap_mode="r")
            with open(self.mdir / Path("vi_tici.pickle"), "rb") as fo:
                self.vi_tici = pickle.load(fo)
            self.ci_vi = {ci: vi for vi, (_, ci) in enumerate(self.vi_tici)}

        return self

    def close(self):
        if hasattr(self, "wordvecarray"):
            del self.wordvecarray
        if hasattr(self, "word_i"):
            del self.word_i

        if self.indexed and self.storage:
            del self.means
            del self.vi_tici
            del self.ci_vi

    def prepare_block(self, tableid_colids: Dict[int, Set[int]]):
        if self.storage:
            # Create FAISS index
            faissindex = faiss.IndexFlatIP(self.means.shape[1])  # build the index
            # add vectors to the index
            faissindex.add(np.array(self.means))  # type: ignore
            log.debug("faiss info: %s", faiss.MatrixStats(self.means).comments)
        else:
            faissindex = self.faissindex

        ci_ti = {ci: ti for ti, cs in tableid_colids.items() for ci in cs}
        qi_mean, ci_qi = [], {}  # type: ignore
        for ci in ci_ti:
            if ci in self.ci_vi:
                ci_qi[ci] = len(qi_mean)
                qi_mean.append(self.means[self.ci_vi[ci]])

        if not len(qi_mean):
            log.error(
                f"No column embeddings found in {self.name} for any of tables"
                f"{list(tableid_colids)}"
            )
        else:
            xq = np.vstack(qi_mean)  # query vectors
            xq /= np.sqrt((xq ** 2).sum(axis=1))[:, None]  # L2 normalize
            xq = xq.astype("float32")
            log.debug(
                f"Querying {self.name} faiss index with query matrix of shape {xq.shape}"
            )
            D, I = faissindex.search(xq, self.topn)

            for ci1, qi in ci_qi.items():
                indexes, similarities = I[qi], np.maximum(D[qi], 0)
                ti1 = ci_ti[ci1]
                for qi2 in set(indexes[(similarities > self.threshold)]) - set([ci1]):
                    ti2, _ = self.vi_tici[qi2]
                    if ti2 != ti1:
                        self.ti_block.setdefault(ti1, set()).add(ti2)

    def block(self, ti: int, cis):
        return self.ti_block.get(ti, set())

    def vecsim(self, m1, m2):
        # Positive cosine distance
        n = ((m1 ** 2).sum(1) ** 0.5) * ((m2 ** 2).sum(1) ** 0.5)
        return np.round(np.maximum(0, (m1 * m2).sum(1) / n), 5)

    def match(self, tableid_colids_pairs):
        pairs = cluster.progress(tableid_colids_pairs, f"Looking up {self.name}")
        inds, vis1, vis2 = [], [], []
        for (ti1, cis1), (ti2, cis2) in pairs:
            for ci1 in cis1:
                if ci1 in self.ci_vi:
                    vi1 = self.ci_vi[ci1]
                    for ci2 in cis2:
                        if ci2 in self.ci_vi:
                            vi2 = self.ci_vi[ci2]
                            inds.append((ti1, ti2, ci1, ci2))
                            vis1.append(vi1)
                            vis2.append(vi2)

        if inds:
            log.debug(f"Calculating {len(inds)} {self.name} scores")
            scores = self.vecsim(self.means[vis1, :], self.means[vis2, :])
            inds = cluster.progress(inds, f"Yielding {self.name}")
            for ind, score in zip(inds, scores):
                yield ind, score
