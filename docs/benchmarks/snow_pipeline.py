# Load datasets
import os
import re
import shutil
import json
import tqdm
import logging as log
from pathlib import Path
from typing import *
from collections import Counter
from itertools import combinations

import pandas as pd
import numpy as np

import takco
from takco.link.profile import pfd_prob_pervalue

def progress(*args, **kwargs):
    disable = log.getLogger().getEffectiveLevel() >= 30
    return tqdm.tqdm(*args, disable=disable, **kwargs)


## Dataset


def get_snow_datasets(snow_rootdir: Path):
    root = Path(snow_rootdir).joinpath("datasets").expanduser().absolute()
    for d in root.iterdir():
        if d.is_dir():
            union_path = d.joinpath("union_dedup_json")
            input_dataset = takco.evaluate.dataset.WebDataCommons(
                fnames=list(union_path.glob("*.json"))
            )
            reference_path = d.joinpath("evaluation/normalised_fd_relations")
            output_dataset = takco.evaluate.dataset.WebDataCommons(
                fnames=list(reference_path.glob("*.json"))
            )
            yield d.name, (input_dataset, output_dataset)


## Generic functions


def aggr_by_val(items):
    aggr = {}
    for k, v in items:
        aggr.setdefault(v, set()).add(k)
    return aggr

def looks_datetime(series, threshold=0.75):
    series = series.replace('',np.nan).dropna()
    # First match simple numbers
    if series.str.match('-?\d+[,.]\d+').mean() > threshold:
        return False

    import calendar
    ms = [m.lower() for m in calendar.month_name if m]
    monthnames =  '(?:%s)' % '|'.join([s for m in ms for s in set([m, m[:3]])])
    daynrs = '|'.join(set(f for i in range(1,32) for f in set([f"{i:d}", f"{i:02d}"])))
    monthnrs = '|'.join(f for i in range(1,13) for f in set([f"{i:d}", f"{i:02d}"]))
    regexes = [
        r'(?:\d{4}\W)?(%s)\W(%s)(?:\W\d{4})?\b' % x
        for m in [monthnrs, monthnames]
        for x in [(m, daynrs), (daynrs, m)]
    ] + [r'\d{4}\b'] # year
    rex = '|'.join('(?:%s)'%r for r in regexes)
    flags = re.IGNORECASE
    return series.str.match(rex, flags=flags).mean() > threshold

def looks_numeric(series, threshold=0.75):
    series = series.replace('',np.nan).dropna()
    if looks_datetime(series, threshold=threshold):
        return False # date
    # Only look at first token!
    series = series.str.split(n=1, expand=True)[0]
    return (series.str.count("[\d\.\-%]") / series.str.len()).mean() > threshold


def looks_longtext(series, threshold=30):
    return series.str.len().mean() > threshold


def get_context_headers(headers: Sequence[Sequence[str]]):
    prefixes = ["page title", "table heading", "disambiguation of", "uri"]
    return [cs for cs in headers if any(c.startswith(i) for i in prefixes for c in cs)]


def get_singleton_cols(df):
    return list(df.columns[df.describe().T["unique"] == 1])


def get_longtext_cols(df, threshold=30):
    return [n for n, c in df.iteritems() if looks_longtext(c, threshold=threshold)]


def guess_numeric_cols(df, threshold=0.75):
    return [col for col in df.columns if looks_numeric(df[col], threshold=threshold)]


def guess_datetime_cols(df, threshold=0.75):
    return [col for col in df.columns if looks_datetime(df[col], threshold=threshold)]


def make_clean_lower(df, num_threshold=0.75):
    """Make the dataframe lowercase and remove leading numbers+punct, like snow"""
    re_clean = "^\d+\W\s"
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr].str.lower()
        frac_start_numpunct = series.str.match(re_clean).sum() / len(series)
        if not looks_numeric(series, num_threshold) and (frac_start_numpunct > 0.9):
            series = series.str.replace(re_clean, "")
        df.iloc[:, colnr] = series
    return df


def make_guessed_numeric(df, threshold=0.75):
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr].str.lower()
        if looks_numeric(series, threshold=threshold):
            numcol = series.str.replace("[^\d\.]", "", regex=True)
            numcol = pd.to_numeric(numcol, errors="coerce").astype("float")
            df.iloc[:, colnr] = numcol.fillna("").astype("str")
    return df


def extract_named_columns(df):
    return df.iloc[:, [i for i in range(df.shape[1]) if "NULL" not in df.columns[i]]]


def extract_bracket_disambiguation(
    df, threshold=0.5, re_bracket=re.compile(r"\(([^\)]*)\)")
):
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr]
        frac_bracketed = series.str.contains("\(").sum() / len(df)
        if frac_bracketed > threshold:
            col = df.columns[colnr]
            if type(col) == tuple:
                newcolname = tuple(f"disambiguation of {c}" for c in col)
            else:
                newcolname = f"disambiguation of {col}"
            df[newcolname] = series.str.extract(re_bracket, expand=False)
            df.iloc[:, colnr] = series.str.replace(re_bracket, "")
    return df


## KB matching


class KB:
    def __init__(self, snow_rootdir: Path, **kwargs):
        # Make KB features
        root = Path(snow_rootdir).expanduser().absolute()
        kb_fnames = list(root.joinpath("knowledgebase/tables/").glob("*.csv"))
        kb_text = {}
        for fname in progress(kb_fnames, desc="Loading KB classes"):
            name = fname.name.split(".")[0]
            vals = pd.read_csv(fname, usecols=[1], skiprows=4, header=None, nrows=None)
            kb_text[name] = vals[1]

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.kb_vectorizer = TfidfVectorizer(analyzer=self._analyze)
        self.K = self.kb_vectorizer.fit_transform(kb_text.values())
        log.debug("Made KB feature matrix of shape %s", self.K.shape)
        self.classes = list(kb_text)

    @staticmethod
    def _analyze(xs):
        texts = pd.Series(xs).dropna()
        texts = texts[texts.map(bool)].astype("str").str.replace("[\d\W]+", " ")
        return list(texts.str.strip().str.lower())

    def _get_query(self, df):
        ok_cols = [
            ci
            for ci, (_, col) in enumerate(df.iteritems())
            if (not looks_numeric(col, 0.5)) and (not looks_longtext(col))
        ]
        if not ok_cols:
            return [], None
        qtexts = [df.iloc[:, ci] for ci in ok_cols]
        Q = self.kb_vectorizer.transform(qtexts)
        return ok_cols, Q

    def predict_classes(self, df, threshold=0.01):
        """Predict KB classes for short, non-numeric columns"""
        ok_cols, Q = self._get_query(df)
        if not ok_cols:
            return {}
        simmat = self.K.dot(Q.T).todense()
        sim = pd.DataFrame(simmat, index=self.classes, columns=ok_cols)
        # weight similarities by log-frac of matching cells in column
        sim *= np.log1p(np.array((Q > 0).sum(axis=1)).T[0]) / np.log1p(len(df))
        # also wright by fraction unique
        sim *= (df.describe().T.reset_index().unique / len(df)).astype("float")
        preds = pd.DataFrame({"class": sim.idxmax(), "score": sim.max()})
        return preds[preds.score > threshold].to_dict("index")


## Foreign Keys


class ForeignKeyTracker:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.class_value_fk = {}
        self.class_nfds = Counter()

    def split_fk(self, df, columns, fkcolnr, fkclass):
        value_fk = self.class_value_fk.setdefault(fkclass, {})
        prefix = f"{fkclass}_{self.dataset_name}"
        fks = [
            f"{prefix}~Row{value_fk.setdefault(v, len(value_fk))}"
            for v in df.iloc[:, fkcolnr]
        ]
        df = df.fillna("")
        df[fkcolnr] = fks
        columns = list(columns)
        columns[fkcolnr] = ("FK",)
        return df, columns

    def iter_fk_tables(self):
        for fkclass, value_fk in self.class_value_fk.items():
            body = [
                (f"{fkclass}_{self.dataset_name}~Row{i}", val)
                for val, i in value_fk.items()
            ]
            head = [("PK", "rdf-schema#label")]
            name = f"{fkclass}_{self.dataset_name}.json"
            yield takco.Table(head=head, body=body, _id=name)

    def decompose_fd_tables(
        self, df, keys, fkclass: str, header: pd.Series = None, exclude_cols=None
    ):
        for c in df.columns:
            if exclude_cols and (c in exclude_cols):
                continue
            if (c not in keys) and (len(set(df[c])) > 1):
                fd_df = df[[c] + list(keys)]

                # get filled unique rows
                filled_mask = fd_df[[c]].fillna(False).applymap(bool).any(axis=1)
                fd_df = fd_df[filled_mask].drop_duplicates(ignore_index=True)

                if header is not None:
                    head = list(zip(*header[[c] + list(keys)]))
                else:
                    head = list(zip(*fd_df.columns))

                # Make name based on fkclass count
                nfd = self.class_nfds[fkclass]
                _id = f"{fkclass}_{self.dataset_name}_fd_{nfd}.json"
                self.class_nfds[fkclass] += 1

                yield takco.Table(head=head, body=fd_df.values, _id=_id)


## Matching and stitching


def match_columns(tabid_df, num_threshold=0.5, agg_threshold_col=0.01):
    """Match columns based on tfidf similarities"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    data_vectorizer = TfidfVectorizer()

    # Make global column IDs and features
    colid_to_tabid_and_colnr = {}
    colid_to_text = {}
    for tabid, df in progress(tabid_df.items(), desc="Getting column text"):
        columns = list(df.columns)
        numeric_cis = set(
            guess_numeric_cols(pd.DataFrame(df.values), threshold=num_threshold)
        )
        context_headers = set(get_context_headers(columns))
        context_cis = set(columns.index(c) for c in context_headers)
        singleton_cis = set(get_singleton_cols(pd.DataFrame(df.values)))
        longtext_cis = set(get_longtext_cols(pd.DataFrame(df.values)))
        for colnr, c in enumerate(df):
            text = ""
            # Only cluster non-numeric, short, non-singleton-context columns
            bad_colnrs = numeric_cis | longtext_cis | (singleton_cis & context_cis)
            if colnr not in bad_colnrs:
                text = " ".join(set(df.iloc[:, colnr].astype("str")))
            colid = f"{tabid}~Col{colnr} {c}"
            colid_to_text[colid] = text
            colid_to_tabid_and_colnr[colid] = (tabid, colnr)

    ## Create thresholded similarities between columns in different tables
    ncols = sum(bool(t) for t in colid_to_text.values())
    log.debug("Calculating %d x %d column similarities...", ncols, ncols)
    D = data_vectorizer.fit_transform(colid_to_text.values())
    tabids, _ = zip(*colid_to_tabid_and_colnr.values())
    index = pd.MultiIndex.from_tuples(zip(colid_to_tabid_and_colnr, tabids))
    colsim = (
        pd.DataFrame(D.dot(D.T).todense(), index=index, columns=index).stack().stack()
    )
    colsim.index.names = ("ci1", "ti1", "ti2", "ci2")
    colsim = colsim[colsim > agg_threshold_col].reset_index()
    colsim = colsim[(colsim["ti1"] != colsim["ti2"])]
    colsim = colsim.set_index(["ci1", "ci2"])[0]
    colsim = colsim[~colsim.index.duplicated()]

    # Make symmetric distance matrix
    d = 1 - colsim.unstack().sort_index(0).sort_index(1).fillna(0)
    d = pd.DataFrame(np.minimum(d, d.T))

    # Run clustering
    log.debug("Clustering columns...")
    from sklearn.cluster import AgglomerativeClustering

    clus = AgglomerativeClustering(
        affinity="precomputed",
        linkage="complete",
        n_clusters=None,
        distance_threshold=1,
    )
    clusters = clus.fit(d)
    colid_to_partcolid = dict(zip(d.index, clusters.labels_))

    return colid_to_partcolid, colid_to_tabid_and_colnr


def stitch_colclustered_tables(tabid_df, colid_to_partcolid, colid_to_tabid_and_colnr):
    partcolid_to_colids = aggr_by_val(colid_to_partcolid.items())
    tabid_and_colnr_to_colid = {v: k for k, v in colid_to_tabid_and_colnr.items()}

    # Connected components
    tabid_to_partid = {tabid: i for i, tabid in enumerate(tabid_df.keys())}
    for colids in partcolid_to_colids.values():
        partid = None
        for colid in colids:
            tabid, _ = colid_to_tabid_and_colnr[colid]
            if partid is None:
                partid = tabid_to_partid[tabid]
            tabid_to_partid[tabid] = partid

    # Stitch tables
    partid_to_tabids = aggr_by_val(tabid_to_partid.items())
    partid_to_tabids = {i: v for i, (_, v) in enumerate(partid_to_tabids.items())}
    for partid, tabids in partid_to_tabids.items():
        partcolid_names = {}
        aligned_tables = []
        for tabid in tabids:
            df = tabid_df[tabid]

            colnr_to_partcolid = {}
            for colnr, colname in enumerate(df):
                colid = tabid_and_colnr_to_colid.get((tabid, colnr), len(colid_to_partcolid))

                # Add unaligned columns to global alignment
                if colid not in colid_to_partcolid:
                    if colname in get_context_headers(df.columns):
                        continue  # ignore unaligned context columns
                    n = len(partcolid_to_colids)
                    partcolid = colid_to_partcolid.setdefault(colid, n)
                    partcolid_to_colids.setdefault(partcolid, set()).add(colid)

                partcolid = colnr_to_partcolid[colnr] = colid_to_partcolid[colid]
                partcolid_names.setdefault(partcolid, Counter()).update(colname)

            data = df.values[:, list(colnr_to_partcolid.keys())]
            columns = colnr_to_partcolid.values()
            aligned_tables.append(pd.DataFrame(data=data, columns=columns))

        # Concatenate aligned tables and sort columns by most filled
        log.debug("Stitching %d aligned tables", len(aligned_tables))
        df = pd.concat(aligned_tables)
        sortcols = df.describe().loc["count"].sort_values(ascending=False).index
        df = df[sortcols]

        # Get most frequent column header per partition-column
        columns = [
            [v for v, c in partcolid_names[pci].most_common(1)] for pci in sortcols
        ]
        yield df, columns


## Functional Dependencies
def df_pfd_prob_pervalue(dfi, key, c):
    X = dfi[list(key) + [c]]
    X = X[X.replace('',np.nan).notna().all(axis=1)]
    if len(X):
        return pfd_prob_pervalue( map(tuple, X.values) )
    else:
        return 0.0

def get_keylike_columns(dfi, numeric_threshold=0.5, stdmean=3, meanlen=30):
    # column statistics
    cs = dfi.applymap(len).replace(0, np.nan).describe().T
    numeric_cols = guess_numeric_cols(dfi, threshold=numeric_threshold)
    cs['numeric'] = False
    cs.loc[numeric_cols, 'numeric'] = True
    ok_cols = (cs['std'] / cs['mean'] < stdmean) & (cs['mean'] < meanlen) & (~cs['numeric'])
    return list(cs[ok_cols].index)

def combinations_upto(it, n):
    for i in range(0, n+1):
        yield from combinations(it, i)


def get_pervalue_pdfs(dfi, fkcolnr, stoplevel = 4, numeric_threshold=0.5, minp = 1):
    nonfk = dfi[dfi.columns[dfi.columns != fkcolnr]]
    candkeys = get_keylike_columns(nonfk, numeric_threshold=numeric_threshold )

    cols = set(dfi.columns)
    candidates = sorted(combinations_upto(candkeys, stoplevel))
    dep_dets = {}
    for candkey in progress(candidates, desc='FD candidates'):
        candkey = (fkcolnr,) + candkey
        dfi.sort_values(list(candkey), inplace=True)
        for depcol in cols - set(candkey):
            p = df_pfd_prob_pervalue(dfi, candkey, depcol)
            if p >= minp:
                dep_dets.setdefault(depcol, set()).add( candkey )

    # Find minimal determinants
    det_dep = {}
    for depcol, dets in dep_dets.items():
        dets = [set(d) for d in dets]
        for det in dets:
            if all(d-det for d in dets if d!=det):
                det_dep.setdefault(tuple(det), set()).add( depcol )
    
    return det_dep

def get_tane_pdfs(tane, stoplevel=4, numeric_threshold=.5, g3_threshold=0):
    try:
        fds = tane.rundf(dfi, stoplevel=1, g3_threshold=g3_threshold)
    except tane.TaneException:
        fds = {}
    
    keylike = set(get_keylike_columns(dfi, numeric_threshold=numeric_threshold))
    fds = {det:dep for det, dep in fds.items() if (fkcolnr in det) and not (set(det)-keylike)}
    return fds



## Snow-specific import / export


def preprocess_tables(
    tables: Sequence[takco.Table],
) -> Mapping[str, pd.DataFrame]:
    """Pre-process tables:
    * Decomposes compound columns that contain cells with brackets
    * Lowercases columns and removes leading numbers + punctuation
    * Guesses and formats numeric columns (1-decimal ints)
    """
    tabid_df = {}
    for t in tables:
        df = extract_bracket_disambiguation(t.df)
        df = make_clean_lower(df)
        tabid_df[t._id] = df
    return tabid_df


def postprocess_tables(
    tables: Sequence[takco.Table],
    numeric_threshold: float = 0.75,
) -> Sequence[takco.Table]:
    for t in tables:
        df = make_guessed_numeric(t.df, threshold=numeric_threshold)
        yield takco.Table(body=df.values, head=zip(*df.columns), _id=t._id)


def write_snow(t, name, fd_path):
    doc = takco.evaluate.dataset.WebDataCommons.convert_back(t, snow=True)
    fname = Path(fd_path).joinpath(name)
    with open(fname, "w") as fw:
        json.dump(doc, fw, ensure_ascii=False)


## Loading Snow gold


def iter_evaltuples(snow_root, dataset_name, fname, n):
    evaldir = Path(snow_root).joinpath(f"datasets/{dataset_name}/evaluation/")
    lines = evaldir.joinpath(fname).open().readlines()
    for line in lines:
        yield line.strip().split("\t", n - 1)


def load_gold_colmatches(snow_root, dataset_name):
    fnames = [
        "union_goldstandard.tsv",
        "context_correspondences.tsv",
        "union_goldstandard.tsv_generated_context.tsv",
    ]
    tuples = [
        t
        for fname in fnames
        for t in iter_evaltuples(snow_root, dataset_name, fname, 2)
    ]
    colid_to_partcolid, colid_to_tabid_and_colnr = {}, {}
    for partcolid, (partcolname, colids) in enumerate(tuples):
        for colid in colids.split(","):
            colid_to_partcolid[colid] = partcolid
            tabid, colnr = colid.split("~Col")
            colid_to_tabid_and_colnr[colid] = (tabid, int(colnr))
    return colid_to_partcolid, colid_to_tabid_and_colnr


def load_gold_fds(snow_root, dataset_name):
    tuples = iter_evaltuples(snow_root, dataset_name, "functional_dependencies.tsv", 3)
    fds = {}
    for fkclass, det, dep in tuples:
        det, dep = tuple(det.split(",")), tuple(dep.split(","))
        fds[det] = dep
    return fds


def load_fkclasses(snow_root, dataset_name):
    tuples = iter_evaltuples(snow_root, dataset_name, "entity_structure.tsv", 3)
    colids_fkclass = {}
    for fkclass, partcolname, colids in tuples:
        if partcolname == "rdf-schema#label":
            colids = colids.split(",")
            colids_fkclass[tuple(set(colids))] = fkclass
    return colids_fkclass


## Pipelines


def predict_fkclasses(tabid_df: Mapping[str, pd.DataFrame], dataset_name: str, kb: KB):
    tabid_to_colnr_and_fkclass = {}
    for tabid, df in tabid_df.items():
        fk_pred = kb.predict_classes(df, threshold=0)
        log.debug(
            "[%s] [%s] Class predictions: %s",
            dataset_name,
            tabid,
            {df.columns[c]: p["class"] for c, p in fk_pred.items()},
        )
        if fk_pred:
            fkcolnr, pred = max(fk_pred.items(), key=lambda x: x[1]["score"])
            fkclass = pred["class"]
            tabid_to_colnr_and_fkclass[tabid] = (fkcolnr, fkclass)
    return tabid_to_colnr_and_fkclass


def iter_binary_decomposed(
    tabid_df: Mapping[str, pd.DataFrame],
    dataset_name: str,
    tabid_to_colnr_and_fkclass: dict,
):
    """Decompose tables based on kbclass-based binary key

    Args:
        tabid_df: List of dataset Tables
        dataset_name: Name of dataset for directory path
        tabid_to_colnr_and_fkclass: FK class predictions

    Yields:
        Table: Decomposed table (FD tables and class tables)
    """

    def debug(msg, *args):
        log.debug("[%s] " + msg, dataset_name, *args)

    fktrack = ForeignKeyTracker(dataset_name)
    for tabid, df in tabid_df.items():

        def tdebug(msg, *args):
            debug("[%s] " + msg, tabid, *args)

        # Only decompose tables that have a matching KB class FK
        if tabid in tabid_to_colnr_and_fkclass:
            fkcolnr, fkclass = tabid_to_colnr_and_fkclass[tabid]
            fkcolname = df.columns[fkcolnr]
            tdebug("Decomposing class %s for col %d (%s)", fkclass, fkcolnr, fkcolname)

            # From column names, find context columns that don't get their own table
            columns = list(df.columns)
            context_headers = set(get_context_headers(columns))
            context_colnrs = set(columns.index(c) for c in context_headers)

            # Add FK values to global FK table
            df = pd.DataFrame(df.values)
            df, columns = fktrack.split_fk(df, columns, fkcolnr, fkclass)
            for t in fktrack.decompose_fd_tables(
                df, [fkcolnr], fkclass, pd.Series(columns), context_colnrs
            ):
                yield t

    for t in fktrack.iter_fk_tables():
        yield t

    log.info(
        "[%s] Created binary tables for classes: %s",
        dataset_name,
        dict(fktrack.class_nfds),
    )


def main(
    snow_root: Path,
    run_name: str,
    *,
    use_datasets: List[str] = (),
    match: bool = False,
    gold_colmatches: bool = False,
    gold_fkclasses: bool = False,
    gold_supertables: bool = False,
    gold_fds: bool = False,
    verbose: bool = False,
):
    loglevel = os.environ.get("LOGLEVEL", "info" if verbose else "warning").upper()
    log.getLogger().setLevel(getattr(log, loglevel))

    snow_root = Path(snow_root).expanduser().absolute()
    datasets_root = snow_root.joinpath("datasets")
    benchmark_datasets = dict(get_snow_datasets(snow_root))
    use_datasets = set(use_datasets) & set(benchmark_datasets)
    if not use_datasets:
        use_datasets = set(benchmark_datasets)
    log.info(f"Got {len(benchmark_datasets)} datasets, using {len(use_datasets)}")

    if not gold_fkclasses:
        kb = KB(snow_root)

    for dataset_name in use_datasets:

        def debug(msg, *args):
            log.debug("[%s] " + msg, dataset_name, *args)

        # Re-create directory
        outdir = f"{dataset_name}/normalised_{run_name}_fd_relations"
        fd_path = datasets_root.joinpath(outdir)
        shutil.rmtree(fd_path, ignore_errors=True)
        Path(fd_path).mkdir(parents=True, exist_ok=True)

        tables = list(takco.TableSet.dataset(benchmark_datasets[dataset_name][0]))
        tabid_df = preprocess_tables(tables)

        if match or gold_colmatches:
            if gold_colmatches:
                debug("Using gold column matches")
                partcols, idpairs = load_gold_colmatches(snow_root, dataset_name)
            else:
                debug("Matching %d tables", len(tabid_df))
                partcols, idpairs = match_columns(tabid_df, agg_threshold_col=0.01)

            if gold_fkclasses:
                colids_fkclass = load_fkclasses(snow_root, dataset_name)
                partcolid_to_fkclass = {}
                for partcolid, colids in aggr_by_val(partcols.items()).items():
                    if tuple(colids) in colids_fkclass:
                        partcolid_to_fkclass[partcolid] = colids_fkclass[tuple(colids)]
                log.debug("Using gold fkclasses %s", partcolid_to_fkclass)
                tabid_to_colnr_and_fkclass = {}

            debug("Stitching %d from matched cols", len(partcols))
            stitched = stitch_colclustered_tables(tabid_df, partcols, idpairs)
            tabid_df = {}
            for partid, (df, columns) in enumerate(stitched):
                tabid = f"part-{partid}"
                if gold_fkclasses:
                    # Look up gold fkclass
                    for partcolid in set(df.columns) & set(partcolid_to_fkclass):
                        colnr = list(df.columns).index(partcolid)
                        fkclass = partcolid_to_fkclass[partcolid]
                        tabid_to_colnr_and_fkclass[tabid] = (colnr, fkclass)

                df.columns = pd.MultiIndex.from_tuples(columns)
                tabid_df[tabid] = df

        if not gold_fkclasses:
            tabid_to_colnr_and_fkclass = predict_fkclasses(tabid_df, dataset_name, kb)

        decomposed = iter_binary_decomposed(
            tabid_df, dataset_name, tabid_to_colnr_and_fkclass
        )
        for t in postprocess_tables(decomposed, numeric_threshold=0.5):
            write_snow(t, t._id, fd_path)


if __name__ == "__main__":
    import defopt

    defopt.run(main)
