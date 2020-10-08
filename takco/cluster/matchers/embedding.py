from pathlib import Path
import logging as log
import collections
import shutil

from .matcher import Matcher

try:
    import faiss
    import pandas as pd
    import numpy as np
except:
    log.error(f"Cannot import faiss/pandas/numpy")


class EmbeddingMatcher(Matcher):
    """Blocking matcher using table cells from header or body"""

    def __init__(
        self,
        fdir: Path,
        name=None,
        source="body",
        wordvec_fname: Path = None,
        topn=100,
        threshold=0,
        create=False,
        **kwargs,
    ):
        """Matcher based on embeddings and FAISS"""
        self.name = name or self.__class__.__name__
        mdir = Path(fdir) / Path(self.name)
        # if create:
        #     shutil.rmtree(mdir, ignore_errors=True)
        # mdir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.wordvec_fname = str(wordvec_fname)
        self.topn = topn
        self.threshold = threshold
        self.wordvec_fname = Path(self.wordvec_fname)
        # self.config(Path(mdir) / Path("config.toml"))

        self.faissindex_fname = Path(mdir) / Path("index.faiss")

        # self.means_fname = Path(mdir) / Path("means.npy")
        # if self.means_fname.exists():
        #     self.means = np.load(self.means_fname, mmap_mode="r")

        self.column_ids_fname = Path(mdir) / Path("column_ids.npy")
        if self.column_ids_fname.exists():
            pass
            # self.ci_vi = collections.OrderedDict(
            #     (ci, vi) for vi, ci in enumerate(np.load(self.column_ids_fname))
            # )
        else:
            self.ci_vi = collections.OrderedDict()
        self.vi_ci = collections.OrderedDict((v, k) for k, v in self.ci_vi.items())

        self.vecs = []
        self.ti_block = {}
        
        super().__init__(fdir)
        
    def __enter__(self):
        log.info(f"Loading word vectors {self.wordvec_fname}")
        if str(self.wordvec_fname).endswith('.pickle'):
            wordvecs = pd.read_pickle(self.wordvec_fname)
        else:
            wordvecs = pd.read_csv(
                self.wordvec_fname, delimiter=" ", quoting=3, header=None, index_col=0
            )

        self.word_i = {w: i for i, w in enumerate(wordvecs.index)}
        self.wordvecarray = np.array(wordvecs)
        return self
    
    def __exit__(self, *args):
        self.wordvecarray = None
        self.word_i = None
        return

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
        for colnum, (ci, cells) in enumerate(zip(ci_range, cols)):
            if colnum not in table.get("numericColumns", []):
                cells = set(c for c in cells if c)
                if len(cells) > 0:
                    cell_vec = self.get_vec(cells)
                    if cell_vec is not None:
                        self.ci_vi[ci] = len(self.vecs)
                        self.vecs.append(cell_vec)

    def get_vec(self, cells):
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
                self.wordvecarray[[l for l in locs]].sum(axis=0) for locs in cell_locs
            ]
            return np.vstack(cell_vecs).mean(axis=0)

    def merge(self, matcher: Matcher):
        if matcher is not None:
            self.vecs += matcher.vecs
            self.ci_vi.update(matcher.ci_vi)
        return self

    def index(self):
        self.means = np.array(self.vecs).astype("float32")
        # np.save(self.means_fname, self.means)
        # np.save(self.column_ids_fname, np.array(list(self.ci_vi.keys())))
        self.vi_ci = collections.OrderedDict((vi, ci) for ci, vi in self.ci_vi.items())

        # Create FAISS index
        V = np.ascontiguousarray(self.means)
        V /= np.sqrt((V ** 2).sum(axis=1))[:, None]

        faissindex = faiss.IndexFlatIP(V.shape[1])  # build the index
        faissindex.add(np.array(V))  # add vectors to the index

        log.debug("faiss info: %s", faiss.MatrixStats(V).comments)
        log.info(f"Writing faiss index to {self.faissindex_fname}")
        Path(self.faissindex_fname).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faissindex, str(self.faissindex_fname))

    def prepare_block(self, tis):
        faissindex = faiss.read_index(str(self.faissindex_fname))

        ci_ti = {ci: ti for ti in tis for ci in self.get_columns(ti)}
        qi_mean, ci_qi = [], {}
        for ci in ci_ti:
            if ci in self.ci_vi:
                ci_qi[ci] = len(qi_mean)
                qi_mean.append(self.means[self.ci_vi[ci]])

        if not len(qi_mean):
            log.debug(f"No column embeddings found in {self.name} for any of tables {tis}")
        else:
            xq = np.vstack(qi_mean)  # query vectors
            xq /= np.sqrt((xq ** 2).sum(axis=1))[:, None]  # L2 normalize
            xq = xq.astype("float32")
            log.debug(f"Querying {self.name} faiss index with query matrix of shape {xq.shape}")
            D, I = faissindex.search(xq, self.topn)

            for ci1, qi in ci_qi.items():
                indexes, similarities = I[qi], np.maximum(D[qi], 0)
                ti1 = ci_ti[ci1]
                for qi2 in set(indexes[(similarities > self.threshold)]) - set([ci1]):
                    if qi2 in self.vi_ci:
                        ti2 = self.get_table(self.vi_ci[qi2])
                        if ti2 != ti1:
                            self.ti_block.setdefault(ti1, set()).add(ti2)

    def block(self, ti: int):
        return self.ti_block.get(ti, set())

    def vecsim(self, a1, a2):
        c = a1.dot(a2) / (np.sqrt((a1 ** 2).sum()) * np.sqrt((a2 ** 2).sum()))
        return np.round(max(c, 0), 5)

    def match(self, table_index_pairs):
        tis = set(ti for pair in table_index_pairs for ti in pair)
        ti_cols = dict(self.get_columns_multi(tis))
        for ti1, ti2 in table_index_pairs:
            for ci1 in ti_cols.get(ti1, []):
                for ci2 in ti_cols.get(ti2, []):
                    if (ci1 in self.ci_vi) and (ci2 in self.ci_vi):
                        vi1, vi2 = self.ci_vi[ci1], self.ci_vi[ci2]
                        a1, a2 = self.means[vi1], self.means[vi2]
                        yield (ti1, ti2, ci1, ci2), self.vecsim(a1, a2)
