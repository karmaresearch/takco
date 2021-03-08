# Load datasets
from pathlib import Path
from typing import *
from collections import Counter
import os
import re
import shutil
import json
import tqdm
import logging as log


def progress(*args, **kwargs):
    disable = log.getLogger().getEffectiveLevel() >= 30
    return tqdm.tqdm(*args, disable=disable, **kwargs)


import takco
import pandas as pd

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
def looks_numeric(series, threshold=0.75):
    return (series.str.count("[\d\.]") / series.str.len()).mean() > threshold


def guess_numeric_cols(df, threshold=0.75):
    return [col for col in df.columns if looks_numeric(df[col], threshold=threshold)]


def get_context_headers(headers: Sequence[Sequence[str]]):
    prefixes = ["page title", "table heading", "disambiguation of", "uri"]
    return [cs for cs in headers if any(c.startswith(i) for i in prefixes for c in cs)]


def get_singleton_cols(df):
    return list(df.columns[df.describe().T["unique"] == 1])


def make_guessed_numeric(df, threshold=0.75):
    df = df.copy()
    for col in df:
        if looks_numeric(df[col], threshold=threshold):
            numcol = df[col].str.replace("[^\d\.]", "", regex=True)
            numcol = pd.to_numeric(numcol, errors="coerce").astype("float")
            df[col] = numcol.fillna("").astype("str")
    return df


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
    def __init__(self, snow_rootdir: Path):
        # Make KB features
        root = Path(snow_rootdir).expanduser().absolute()
        kb_fnames = list(root.joinpath("knowledgebase/tables/").glob("*.csv"))
        kb_text = {}
        for fname in progress(kb_fnames, desc="Loading KB classes"):
            name = fname.name.split(".")[0]
            vals = pd.read_csv(fname, usecols=[1], skiprows=4, header=None, nrows=None)
            kb_text[name] = " ".join(vals[1].astype("str"))

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.kb_vectorizer = TfidfVectorizer(max_df=0.9)
        self.K = self.kb_vectorizer.fit_transform(kb_text.values())
        self.classes = list(kb_text)

    def predict_classes(self, df, threshold=0.04):
        non_numeric_cols = [
            ci for ci in range(df.shape[1]) if not looks_numeric(df.iloc[:, ci])
        ]
        qtexts = [" ".join(set(df.iloc[:, ci].fillna(""))) for ci in non_numeric_cols]
        Q = self.kb_vectorizer.transform(qtexts)
        simmat = self.K.dot(Q.T).todense()
        sim = pd.DataFrame(simmat, index=self.classes, columns=non_numeric_cols)
        sim = (sim * (df.describe().T.reset_index().unique / len(df))).astype("float")
        # display(sim.style.background_gradient(cmap ='viridis', vmax=.05))
        preds = pd.DataFrame({"class": sim.idxmax(), "score": sim.max()})
        return preds[preds.score > threshold].to_dict("index")


# Foreign Keys
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
                filled_mask = fd_df.fillna(False).applymap(bool).any(axis=1)
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


def write_snow(t, name, fd_path):
    doc = takco.evaluate.dataset.WebDataCommons.convert_back(t, snow=True)
    fname = Path(fd_path).joinpath(name)
    with open(fname, "w") as fw:
        json.dump(doc, fw, ensure_ascii=False)


def iter_fkclass_binary_decomposed(
    tables: Sequence[takco.Table], dataset_name: str, kb: KB
):
    """Decompose tables based on kbclass-based binary key

    Also:
    * Decomposes compound columns that contain cells with brackets
    * Guesses and formats numeric columns (1-decimal ints)

    Args:
        tables: List of dataset Tables
        dataset_name: Name of dataset for directory path

    Yields:
        Table: Decomposed table (FD tables and class tables)
    """
    def debug(msg, *args):
        log.debug("[%s]" + msg, dataset_name, *args)

    fktrack = ForeignKeyTracker(dataset_name)
    for ti, t in enumerate(tables):
        debug("Processing table %s of shape %s", ti, df.shape)
        df = extract_bracket_disambiguation(t.df)
        debug("Extracted bracketed cols from table %s, new shape: %s", ti, df.shape)

        fk_pred = kb.predict_classes(t.df, threshold=0)
        # Only decompose tables that have a matching KB class FK
        if fk_pred:
            fkcolnr, pred = max(fk_pred.items(), key=lambda x: x[1]["score"])
            fkclass = pred["class"]

            # From column names, find context columns that don't get their own table
            columns = list(df.columns)
            context_headers = set(get_context_headers(columns))
            exclude_colnrs = set(columns.index(c) for c in context_headers)

            # Add FK values to global FK table
            df = make_guessed_numeric(pd.DataFrame(df.values), threshold=0.5)
            df, columns = fktrack.split_fk(df, zip(*t.head), fkcolnr, fkclass)
            for t in fktrack.decompose_fd_tables(
                df, [fkcolnr], fkclass, pd.Series(columns), exclude_colnrs
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
    verbose: bool = False,
):
    if 'LOGLEVEL' in os.environ:
        log.getLogger().setLevel(getattr(log, os.environ['LOGLEVEL'].upper()))
    else:
        log.getLogger().setLevel(log.INFO if verbose else log.WARNING)

    snow_root = Path(snow_root)
    benchmark_datasets = dict(get_snow_datasets(snow_root))
    use_datasets = set(use_datasets) & set(benchmark_datasets)
    if not use_datasets:
        use_datasets = set(benchmark_datasets)
    log.info(f"Got {len(benchmark_datasets)} datasets, using {len(use_datasets)}")
    kb = KB(snow_root)
    datasets_root = snow_root.joinpath("datasets")

    for dataset_name in use_datasets:
        # Re-create directory
        outdir = f"{dataset_name}/normalised_{run_name}_fd_relations"
        fd_path = datasets_root.joinpath(outdir)
        shutil.rmtree(fd_path, ignore_errors=True)
        Path(fd_path).mkdir(parents=True, exist_ok=True)

        tables = list(takco.TableSet.dataset(benchmark_datasets[dataset_name][0]))
        for t in iter_fkclass_binary_decomposed(tables, dataset_name, kb):
            name = t._id
            write_snow(t, name, fd_path)


if __name__ == "__main__":
    import defopt

    defopt.run(main)
