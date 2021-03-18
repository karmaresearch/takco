# Load datasets
import os
import re
import shutil
import json
import pickle
import logging as log
from pathlib import Path
from typing import *
from collections import Counter
from itertools import combinations, groupby

import tqdm
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
                fnames=sorted(union_path.glob("*.json"))
            )
            reference_path = d.joinpath("evaluation/normalised_fd_relations")
            output_dataset = takco.evaluate.dataset.WebDataCommons(
                fnames=sorted(reference_path.glob("*.json"))
            )
            yield d.name, (input_dataset, output_dataset)


## Generic functions


def aggr_by_val(items):
    aggr = {}
    for k, v in items:
        aggr.setdefault(v, set()).add(k)
    return aggr


def _make_date_regex():
    import calendar

    ms = [m.lower() for m in calendar.month_name if m]
    monthnames = "(?:%s)" % "|".join([s for m in ms for s in set([m, m[:3]])])
    dns = [f for i in range(1, 32) for f in set([str(i), f"{i:02d}"])]
    daynrs = "(%s)" % "|".join(dns)
    mns = [f for i in range(1, 13) for f in set([str(i), f"{i:02d}"])]
    monthnrs = "(%s)" % "|".join(mns)
    regexes = [
        r"(?:\d{4}[/\-\.\s])?%s[/\-\.\s]%s(?:[/\-\.\s]\d{4})?\b" % x
        for m in [monthnrs, monthnames]
        for d in [daynrs, daynrs + "(?:th|nd)?"]
        for x in [(m, d), (d, m)]
    ] + [
        r"\d{4}\b",
        monthnames,
    ]  # year
    return "|".join("(?:%s)" % r for r in regexes)


def looks_date(series, threshold=0.75):
    series = series.replace("", np.nan).dropna().astype("str")
    # First match simple numbers
    if series.str.match("-?\d+[,.]\d+").mean() > threshold:
        return False

    rex = _make_date_regex()
    flags = re.IGNORECASE
    return series.str.match(rex, flags=flags).mean() > threshold


def looks_numeric(series, threshold=0.75):
    series = series.replace("", np.nan).dropna().astype("str")
    if looks_date(series, threshold=threshold):
        return False  # date
    return (series.str.count("[\d\.\-%]") / series.str.len()).mean() > threshold


def looks_longtext(series, threshold=30):
    return series.str.len().mean() > threshold


def get_context_headers(headers: Sequence[Sequence[str]]):
    p = ["page title", "table heading", "disambiguation of", "uri"]
    return [j for j, h in enumerate(headers) for i in p for c in h if c.startswith(i)]


def get_singleton_cols(df):
    return list(df.columns[df.describe().T["unique"] == 1])


def get_longtext_cols(df, threshold=30):
    return [
        n
        for n, (_, c) in enumerate(df.iteritems())
        if looks_longtext(c, threshold=threshold)
    ]


def guess_numeric_cols(df, threshold=0.75):
    return [
        n
        for n, (_, c) in enumerate(df.iteritems())
        if looks_numeric(c, threshold=threshold)
    ]


def guess_date_cols(df, threshold=0.75):
    return [
        n
        for n, (_, c) in enumerate(df.iteritems())
        if looks_date(c, threshold=threshold)
    ]


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

        self.kb_vectorizer = TfidfVectorizer(analyzer=self._analyze, **kwargs)
        self.K = self.kb_vectorizer.fit_transform(kb_text.values())
        log.debug("Made KB feature matrix of shape %s", self.K.shape)
        self.classes = list(kb_text)

    @staticmethod
    def _analyze(xs):
        """Extract full-cell and BoW features"""
        texts = pd.Series(xs).dropna()
        texts = texts[texts.map(bool)].astype("str").str.replace("[\d\W]+", " ")
        texts = texts.str.strip().str.lower()
        return list(texts) + list(texts.str.split(expand=True).stack())

    def _get_query(self, df):
        ok_cols = [
            ci
            for ci, (_, col) in enumerate(df.iteritems())
            if (
                (not looks_numeric(col, 0.5))
                and (not looks_longtext(col))
                and (not looks_date(col, 0.5))
                and col.notna().mean() > 0.5
            )
        ]
        if not ok_cols:
            return [], None
        qtexts = [df.iloc[:, ci] for ci in ok_cols]
        Q = self.kb_vectorizer.transform(qtexts)
        return ok_cols, Q

    def _get_sim(self, df):
        ok_cols, Q = self._get_query(df)
        if not ok_cols:
            return None
        simmat = self.K.dot(Q.T).todense()
        sim = pd.DataFrame(simmat, index=self.classes, columns=ok_cols)
        # weight similarities by log-frac of matching cells in column
        sim *= np.log1p(np.array((Q > 0).sum(axis=1)).T[0]) / np.log1p(len(df))
        # also weight by fraction unique
        sim *= (df.describe().T.reset_index().unique / len(df)).astype("float")
        return sim

    def predict_classes(self, df, threshold=0.01):
        """Predict KB classes for short, non-numeric columns"""
        sim = self._get_sim(df)
        if sim is None:
            return {}
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


class TfidfMatcher:
    def __init__(self, name=None, num_threshold=0.5, **kwargs):
        self.num_threshold = num_threshold
        kwargs["analyzer"] = self.__class__._analyzer
        self.tfidf_kwargs = kwargs
        self.name = name or self.__class__.__name__

    @staticmethod
    def iter_text(tabid_df, num_threshold=0.5):
        for tabid, df in progress(tabid_df.items(), desc="Getting column text"):
            columns = list(df.columns)
            numeric_cis = set(guess_numeric_cols(df, threshold=num_threshold))
            context_cis = set(get_context_headers(columns))
            singleton_cis = set(get_singleton_cols(df))
            longtext_cis = set(get_longtext_cols(df))
            for colnr, c in enumerate(df):
                text = []
                # Only cluster non-numeric, short, non-singleton-context columns
                bad_colnrs = numeric_cis | longtext_cis | (singleton_cis & context_cis)
                if colnr not in bad_colnrs:
                    text = df.iloc[:, colnr].unique()
                yield tabid, colnr, text

    @staticmethod
    def _analyzer(values):
        series = pd.Series(values).replace("", np.nan).dropna().astype("str")
        # replace money
        curr = r"(?:\$|kr|\€|usd|chf|\£|\¥|\₹|s\$|hk\$|nt\$|tl|р|aed)"
        re_money = f"(?:{curr}[\d\s,\.]+)|(?:[\d\s,\.]+{curr})|free|gratis"
        series = series.str.replace(re_money, "$MONEY$")
        # replace numbers
        series = series.str.replace("\d", "$")
        return list(series.str.split().sum() or [])

    def _get_index_and_features(self, tabid_df, tabid_and_colnr_to_colid):
        from sklearn.feature_extraction.text import TfidfVectorizer

        tabids, colnrs, texts = zip(
            *self.iter_text(tabid_df, num_threshold=self.num_threshold)
        )
        colids = [tabid_and_colnr_to_colid[tc] for tc in zip(tabids, colnrs)]
        index = pd.MultiIndex.from_tuples(zip(colids, tabids))

        texts = progress(texts, desc="Extracting features")
        D = TfidfVectorizer(**self.tfidf_kwargs).fit_transform(texts)
        return index, D

    def match(self, tabid_df, tabid_and_colnr_to_colid):
        index, D = self._get_index_and_features(tabid_df, tabid_and_colnr_to_colid)
        log.debug("Got %s column features. Calculating similarities...", D.shape)
        simmat = pd.DataFrame(D.dot(D.T).todense(), index=index, columns=index)
        simseries = simmat.stack().stack()
        simseries.index.names = ("ci1", "ti1", "ti2", "ci2")
        return simseries.rename(self.name)


class ExactHeadMatcher:
    def __init__(self, name=None, include_context=True):
        self.include_context = include_context
        self.name = name or self.__class__.__name__

    def match(self, tabid_df, tabid_and_colnr_to_colid):
        D = pd.DataFrame(
            [
                (ti, ci, " ".join(tabid_df[ti].columns[cn]))
                for (ti, cn), ci in tabid_and_colnr_to_colid.items()
                for context in [set(get_context_headers(tabid_df[ti].columns))]
                if self.include_context or (cn not in context)
            ],
            columns=["ti", "ci", "h"],
        )
        hsim = D.merge(D, on="h", suffixes=("1", "2")).set_index(
            ["ci1", "ti1", "ti2", "ci2"]
        )
        return hsim.assign(h=1).h.rename(self.name)


class KBClassMatcher:
    def __init__(self, kb, name=None, include_context=True):
        self.kb = kb
        self.include_context = include_context
        self.name = name or self.__class__.__name__

    def match(self, tabid_df, tabid_and_colnr_to_colid):
        tabid_fkclass = predict_fkclasses(tabid_df, self.name, self.kb)
        D = pd.DataFrame(
            [
                (ti, ci, tabid_fkclass[ti][1])
                for (ti, cn), ci in tabid_and_colnr_to_colid.items()
                if ti in tabid_fkclass
            ],
            columns=["ti", "ci", "c"],
        )
        sim = D.merge(D, on="c", suffixes=("1", "2")).set_index(
            ["ci1", "ti1", "ti2", "ci2"]
        )
        return sim.assign(c=1).c.rename(self.name)


def match_columns(tabid_df, matchers, agg_func="max", agg_threshold_col=0.01):
    """Match columns based on matcher similarities, aggregating by ``agg_func`` """
    # Make global column IDs
    colid_to_tabid_and_colnr = {
        f"{tabid}~Col{colnr} {c}": (tabid, colnr)
        for tabid, df in tabid_df.items()
        for colnr, c in enumerate(df)
    }
    tabid_and_colnr_to_colid = {v: k for k, v in colid_to_tabid_and_colnr.items()}

    ## Create thresholded similarities between columns in different tables
    matches = [m.match(tabid_df, tabid_and_colnr_to_colid) for m in matchers]
    simdf = pd.concat([m.to_frame() for m in matches], axis=1)

    # Aggregate matches
    colsim = takco.cluster.aggregate_match_sims(simdf, agg_func)

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


def partition_connected_components(
    tabid_df, partcolid_to_colids, colid_to_tabid_and_colnr
):
    # Connected components
    tabid_to_partid = {tabid: i for i, tabid in enumerate(tabid_df.keys())}
    for colids in partcolid_to_colids.values():
        partid = None
        for colid in colids:
            tabid, _ = colid_to_tabid_and_colnr[colid]
            if partid is None:
                partid = tabid_to_partid[tabid]
            tabid_to_partid[tabid] = partid
    partid_to_tabids = aggr_by_val(tabid_to_partid.items())
    return {i: v for i, (_, v) in enumerate(partid_to_tabids.items())}


def stitch_colclustered_tables(
    tabid_df, colid_to_partcolid, colid_to_tabid_and_colnr, min_len: int = 10
):
    colid_to_partcolid = dict(colid_to_partcolid)
    colid_to_tabid_and_colnr = dict(colid_to_tabid_and_colnr)

    partcolid_to_colids = aggr_by_val(colid_to_partcolid.items())
    tabcol_to_colid = {v: k for k, v in colid_to_tabid_and_colnr.items()}
    partid_to_tabids = partition_connected_components(
        tabid_df, partcolid_to_colids, colid_to_tabid_and_colnr
    )

    # Stitch tables
    for partid, tabids in partid_to_tabids.items():
        partcolid_names = {}
        aligned_tables = []
        for tabid in tabids:
            df = tabid_df[tabid]

            colnr_to_partcolid = {}
            for colnr, colname in enumerate(df):
                colid = tabcol_to_colid.get((tabid, colnr), len(colid_to_partcolid))

                # Add unaligned columns to global alignment
                if colid not in colid_to_partcolid:
                    if colname in df.columns[get_context_headers(df.columns)]:
                        n = colname  # add unaligned context columns
                    else:
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
        if (not len(df.columns)) or (len(df) < min_len):
            continue
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
    X = X[X.replace("", np.nan).notna().all(axis=1)]
    if len(X):
        return pfd_prob_pervalue(map(tuple, X.values))
    else:
        return 0.0


def get_keylike_columns(dfi, numeric_threshold=0.5, stdmean=3, meanlen=50, cover=0.5):
    # column statistics
    cs = dfi.applymap(len).replace(0, np.nan).describe().T
    numeric_cols = guess_numeric_cols(dfi, threshold=numeric_threshold)
    cs["numeric"] = False
    cs.loc[numeric_cols, "numeric"] = True
    ok_cols = (
        (cs["std"] / cs["mean"] < stdmean)
        & (cs["mean"] < meanlen)
        & (~cs["numeric"])
        & (cs["count"] / len(dfi) > cover)
    )
    return list(cs[ok_cols].index)


def combinations_upto(it, n):
    for i in range(0, n + 1):
        yield from combinations(it, i)


def get_pervalue_pdfs(dfi, fkcolnr, candidate_keys, stoplevel=4, minp=1):
    cols = set(dfi.columns)
    candidate_keys = set(candidate_keys) - set([fkcolnr])
    candidates = sorted(combinations_upto(candidate_keys, stoplevel))
    dep_dets = {}
    for candkey in progress(candidates, desc="FD candidates"):
        candkey = candkey + (fkcolnr,)
        dfi.sort_values(list(candkey), inplace=True)
        for depcol in cols - set(candkey):
            p = df_pfd_prob_pervalue(dfi, candkey, depcol)
            if p >= minp:
                dep_dets.setdefault(depcol, set()).add(candkey)

    # Find minimal determinants
    det_dep = {}
    for depcol, dets in dep_dets.items():
        dets = [set(d) for d in dets]
        for det in dets:
            if all(d - det for d in dets if d != det):
                det_dep.setdefault(tuple(det), set()).add(depcol)

    return det_dep


def violation_selection(det_dep):
    return max(det_dep, key=lambda x: len(x))


def get_tane_pdfs(tane, stoplevel=4, numeric_threshold=0.5, g3_threshold=0):
    try:
        fds = tane.rundf(dfi, stoplevel=1, g3_threshold=g3_threshold)
    except tane.TaneException:
        fds = {}

    keylike = set(get_keylike_columns(dfi, numeric_threshold=numeric_threshold))
    fds = {
        det: dep
        for det, dep in fds.items()
        if (fkcolnr in det) and not (set(det) - keylike)
    }
    return fds


## Snow-specific processing


def write_snow(t, name, fd_path):
    doc = takco.evaluate.dataset.WebDataCommons.convert_back(t, snow=True)
    fname = Path(fd_path).joinpath(name)
    with open(fname, "w") as fw:
        json.dump(doc, fw, ensure_ascii=False)


def heuristic_bracket_disambiguation(
    tabid_df, threshold=0.5, re_bracket=re.compile(r"\(([^\)]*)\)")
):
    """Extract bracketed strings into new columns"""
    tabid_df = dict(tabid_df)
    for tabid, df in tabid_df.items():
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
        tabid_df[tabid] = df
    return tabid_df


def heuristic_context_date(tabid_df, threshold=0.5):
    """Rename year and month context columns"""
    import calendar

    month_re = "|".join([m.lower() for m in calendar.month_name if m])
    year_re = "\d{4}$"

    tabid_df = dict(tabid_df)
    for tabid, df in tabid_df.items():
        newcolumns = {}
        for colname, series in df.iloc[:, get_context_headers(df.columns)].iteritems():
            series = series.astype("str")
            if series.str.match(month_re).mean() > threshold:
                newcolumns[colname] = (f"{colname[0]} (month)",)
            if series.str.match(year_re).mean() > threshold:
                newcolumns[colname] = (f"{colname[0]} (year)",)
        if newcolumns:
            df = df.copy()
            for old, new in newcolumns.items():
                df[new] = df[old]
                del df[old]
            tabid_df[tabid] = df
    return tabid_df


def heuristic_uri_pattern(tabid_df):
    """Change name of URI columns preceded by singletons

    * First, group tables by number of URI parts
    * Then, find URI columns that are always singletons
    * Put the singleton value into the title of the next column
    """

    def get_uri_headers(headers):
        return [j for j, c in enumerate(headers) if re.match("uri \d", c[0])]

    def uri_len(tid_df):
        return len(get_uri_headers(tid_df[1].columns))

    tabid_df = dict(tabid_df)
    for n, tables in groupby(sorted(tabid_df.items(), key=uri_len), uri_len):
        if n < 2:
            continue
        tables = list(tables)
        pattern_cols = set()
        for tid, df in tables:
            dfi = pd.DataFrame(df.values)
            uricols = set(get_uri_headers(df.columns))
            singletons = set(get_singleton_cols(dfi[uricols]))
            candidates = set(s for s in singletons if s + 1 in uricols)
            pattern_cols = (pattern_cols & candidates) if pattern_cols else candidates

        for tid, df in tables:
            if pattern_cols:
                leftcolnr = list(pattern_cols)[-1]
                leftcol = df.columns[leftcolnr]
                rightcol = df.columns[leftcolnr + 1]

                name = f"{leftcol[0]} ({ df.iloc[0, leftcolnr] })"
                col = df[rightcol].copy().astype("str")
                df[(name,)] = col
                tabid_df[tid] = df.drop(columns=[leftcol, rightcol])
    return tabid_df


def make_clean_lower(df, num_threshold=0.75):
    """Make the dataframe lowercase and remove leading numbers+punct, like snow"""
    re_clean = r"^\d+\.\s\b"
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr].str.lower()
        frac_start_numpunct = series.str.match(re_clean).sum() / len(series)
        if not looks_numeric(series, num_threshold) and (frac_start_numpunct > 0.9):
            series = series.str.replace(re_clean, "")
        df.iloc[:, colnr] = series
    return df


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
        tabid_df[t._id] = make_clean_lower(t.df)

    tabid_df = heuristic_bracket_disambiguation(tabid_df)
    tabid_df = heuristic_context_date(tabid_df)
    tabid_df = heuristic_uri_pattern(tabid_df)
    return tabid_df


def make_guessed_numeric(df, threshold=0.75):
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr].str.lower()
        if looks_numeric(series, threshold=threshold):
            numcol = series.str.replace("[^\d\.]", "", regex=True)
            numcol = pd.to_numeric(numcol, errors="coerce").astype("float")
            df.iloc[:, colnr] = numcol.fillna("").astype("str")
    return df


def make_guessed_date(df, threshold=0.75):
    df = df.copy()
    for colnr in range(df.shape[1]):
        series = df.iloc[:, colnr].str.lower()
        if looks_date(series, threshold=threshold):
            pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M")
            df.iloc[:, colnr] = series.fillna("").astype("str")
    return df


def postprocess_tables(
    tables: Sequence[takco.Table],
    date_threshold: float = 0.75,
    numeric_threshold: float = 0.75,
) -> Sequence[takco.Table]:
    for t in tables:
        df = make_guessed_date(t.df, threshold=date_threshold)
        df = make_guessed_numeric(t.df, threshold=numeric_threshold)
        yield takco.Table(body=df.values, head=zip(*df.columns), _id=t._id)


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


def predict_fkclasses(
    tabid_df: Mapping[str, pd.DataFrame], dataset_name: str, kb: KB, threshold=threshold
):
    tabid_to_colnr_and_fkclass = {}
    for tabid, df in tabid_df.items():
        fk_pred = kb.predict_classes(df, threshold=threshold)
        log.debug(
            "[%s] [%s] Class predictions: %s",
            dataset_name,
            tabid,
            [
                "{}:{class}/{score:.2e}".format(*df.columns[c], **p)
                for c, p in fk_pred.items()
            ],
        )
        if fk_pred:
            fkcolnr, pred = max(fk_pred.items(), key=lambda x: x[1]["score"])
            fkclass = pred["class"]
            tabid_to_colnr_and_fkclass[tabid] = (fkcolnr, fkclass)
    return tabid_to_colnr_and_fkclass


def iter_decomposed(
    tabid_df: Mapping[str, pd.DataFrame],
    dataset_name: str,
    tabid_to_colnr_and_fkclass: dict,
    nary: bool = False,
    nary_minp: float = 0.95,
    nary_stoplevel: int = 2,
):
    """Decompose tables, possibly inferring nary keys

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

            # Add FK values to global FK table
            dfi = pd.DataFrame(df.values)
            columns = list(df.columns)
            dfi, columns = fktrack.split_fk(dfi, columns, fkcolnr, fkclass)

            # From column names, find context columns that don't get their own table
            badcols = set(get_context_headers(columns))
            columns = pd.Series(columns)
            tdebug("Not decomposing context columns %s", list(columns[badcols]))

            if nary:

                def schema(cs):
                    return "[%s]" % "|".join(" ".join(c) for c in columns[list(cs)])

                candkeys = get_keylike_columns(dfi, numeric_threshold=0.5)
                tdebug("Inferring FDs for %s", schema(candkeys))
                fds = get_pervalue_pdfs(
                    dfi, fkcolnr, candkeys, stoplevel=nary_stoplevel, minp=nary_minp
                )
                # undetermined = set(dfi.columns) - badcols
                # for det, dep in fds.items():
                #     tdebug("Got FD key %s -> %s", schema(det), schema(dep - badcols) )
                #     deps.append((det, list(det) + list(dep)))
                #     undetermined -= set(dep) | set(det)
                # tdebug("Undetermined cols: %s", schema(undetermined))
                det = violation_selection(fds)
                dep = set(dfi.columns) - badcols - set(det)
                tdebug("Got FD key %s -> %s", schema(det), schema(dep))
                deps = [(list(det), dfi.columns)]
            else:
                deps = [([fkcolnr], dfi.columns)]

            for key, cols in deps:
                for t in fktrack.decompose_fd_tables(
                    dfi[cols], key, fkclass, columns, badcols
                ):
                    yield t

    for t in fktrack.iter_fk_tables():
        yield t

    log.info(
        "[%s] Created tables for classes: %s",
        dataset_name,
        dict(fktrack.class_nfds),
    )


def main(
    snow_root: Path,
    run_name: str,
    *,
    use_datasets: List[str] = (),
    match_head: bool = False,
    match_tfidf: bool = False,
    match_kbclass: bool = False,
    agg_func: str = "max",
    agg_threshold_col: float = 0.01,
    nary_induction: bool = False,
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
        kbpickle = snow_root.joinpath("knowledgebase/kb.pickle")
        if kbpickle.exists():
            with open(kbpickle, "rb") as fr:
                kb = pickle.load(fr)
        else:
            kb = KB(snow_root)
            with open(kbpickle, "wb") as fw:
                pickle.dump(kb, fw)

    matchers = []
    if match_head:
        matchers.append(ExactHeadMatcher(include_context=False))
    if match_tfidf:
        matchers.append(TfidfMatcher(num_threshold=0.75, min_df=2))
    if match_kbclass:
        matchers.append(KBClassMatcher(kb))

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

        if matchers or gold_colmatches:
            if gold_colmatches:
                debug("Using gold column matches")
                partcols, idpairs = load_gold_colmatches(snow_root, dataset_name)
            else:
                debug("Matching %d tables", len(tabid_df))
                partcols, idpairs = match_columns(
                    tabid_df,
                    matchers,
                    agg_func=agg_func,
                    agg_threshold_col=agg_threshold_col,
                )

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

        decomposed = iter_decomposed(
            tabid_df, dataset_name, tabid_to_colnr_and_fkclass, nary=nary_induction
        )
        for t in postprocess_tables(decomposed, numeric_threshold=0.5):
            write_snow(t, t._id, fd_path)


if __name__ == "__main__":
    import defopt

    defopt.run(main)
