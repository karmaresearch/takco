from pathlib import Path
import logging as log
from collections import Counter
import warnings
import sqlite3
import hashlib


from . import matchers as matcher_classes
import sys, inspect

all_matchers = {
    name: cls
    for name, cls in inspect.getmembers(sys.modules[matcher_classes.__name__])
    if inspect.isclass(cls)
}


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def load_matchers(dirpath, matcher_kwargs, create=False):
    for name, kwargs in matcher_kwargs.items():
        matcher_class = all_matchers[kwargs["class"]]
        m = matcher_class(dirpath, create=create, **kwargs)
        m.name = name
        yield m


def matcher_add_tables(tables, dirpath, matcher_kwargs):
    # Create matchers
    matchers = list(load_matchers(dirpath, matcher_kwargs, create=True))
    log.info(f"Loading tables into matchers")
    for table in tables:
        for m in matchers:
            m.add(table)
    return matchers


def yield_blocked_matches(table_indices, dirpath, matcher_kwargs):
    # Re-load matchers
    matchers = list(load_matchers(dirpath, matcher_kwargs))

    for matcher in matchers:
        matcher.prepare_block(table_indices)

    for ti1 in table_indices:
        block = set()
        for matcher in matchers:
            block |= set(matcher.block(ti1))

        log.debug(f"Found {len(block)} blocked candidates for table {ti1}")

        for ti2 in block:
            for mi, matcher in enumerate(matchers):
                for s, ci1, ci2 in matcher.match(ti1, ti2):
                    yield mi, (ti1, ti2, ci1, ci2), s


def yield_tablepairs_matches(table_index_pairs, dirpath, matcher_kwargs):
    # Re-load matchers
    matchers = list(load_matchers(dirpath, matcher_kwargs))

    for ti1, ti2 in table_index_pairs:
        for mi, matcher in enumerate(matchers):
            for s, ci1, ci2 in matcher.match(ti1, ti2):
                yield mi, (ti1, ti2, ci1, ci2), s


def aggregate_similarities(sims, agg_func):
    """Aggregate similarities using a numexpr aggregation function.
    
    See also:
        `Pandas eval <https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#supported-syntax>`_
    """
    import pandas as pd
    import numpy as np

    funcs = {
        "max": lambda *args: np.nanmax(args, axis=0),
        "min": lambda *args: np.nanmin(args, axis=0),
        "mean": lambda *args: np.nanmean(args, axis=0),
    }
    if agg_func in funcs:
        agg = funcs[agg_func](*(sims[c] for c in sims))
    else:
        agg = sims.eval(agg_func, local_dict=funcs, engine="python")
    return pd.Series(agg, index=sims.index)


def max_align(g, return_total=False):
    lr, rl, t = {}, {}, 0
    for (_, _, l, r), v in g.sort_values(ascending=False).items():
        if (l not in lr) and (r not in rl):
            lr[l], rl[r], t = r, l, t + v
    return t if return_total else lr


try:
    from pandas import DataFrame
    from sklearn.cluster import AgglomerativeClustering
except:
    import typing

    DataFrame = typing.Any
    AgglomerativeClustering = typing.Any


def cluster_columns(colsim: DataFrame, clus: AgglomerativeClustering, pi=None):
    """Cluster columns from different tables together within a cluster of tables
    
    Column similarities within one table are set to 0 to prevent different columns 
    within one table from linking.
    
    Args:
        colsim: Dataframe of column similarities
        clus: Agglomerative clustering method
        pi: Partition information (for debugging)
    """
    # Don't allow different columns within one table to link
    colsim = colsim[(colsim["ti1"] != colsim["ti2"]) | (colsim["ci1"] == colsim["ci2"])]
    colsim = colsim.set_index(["ci1", "ci2"])[0]
    colsim = colsim[~colsim.index.duplicated()]
    d = 1 - colsim.unstack().sort_index(0).sort_index(1).fillna(0)
    if (d != d.T).any().any() or (d.shape[0] != d.shape[1]):
        log.warn(f"Distance matrix of partition {pi} is not symmetric!")

    partcols = clus.fit_predict(d)

    # Sort cluster columns by frequency
    partcol_rank = {
        pci: r for r, (pci, _) in enumerate(Counter(partcols).most_common())
    }
    partcols = [partcol_rank[pci] for pci in partcols]

    return dict(zip(d.index, partcols))


def align_columns(rows, alignment, empty_cell):
    for row in rows:
        yield [
            row[c] if (c != None and c < len(row)) else empty_cell
            for _, c in sorted(alignment.items())
        ]


def get_top_headers(tableHeaders, merge_headers=None, topn=None):
    if merge_headers is None:
        merge_headers = [[{}] * len(tableHeaders[0])] if tableHeaders else []
    top = []
    if merge_headers:
        for merge, hcol in zip(merge_headers[0], zip(*tableHeaders)):
            c = Counter(
                cell.get("text", "").strip()
                for cell in hcol
                if cell.get("text", "").strip()
            )
            if merge:
                c += Counter(merge.get("freq", {}))

            txt = "\t".join(txt for txt, _ in c.most_common(topn))
            top.append({"text": txt, "tdHtmlString": f"<th>{txt}</th>", "freq": c})
        return [top]
    else:
        return []


def merge_partition_tables(mergetable, table):
    empty_cell = {"text": ""}
    pi = table["part"]

    if mergetable.get("type") != "partition":
        tableData = list(
            align_columns(
                mergetable["tableData"], mergetable["partColAlign"], empty_cell
            )
        )
        tableHeaders = list(
            align_columns(
                mergetable["tableHeaders"], mergetable["partColAlign"], empty_cell
            )
        )
        tableHeaders = get_top_headers(tableHeaders)
        headerText = tuple(
            tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
        )

        mergetable = {
            "_id": f"{pi}-0",
            "pgId": pi,
            "tbNr": 0,
            "type": "partition",
            "pgTitle": f"Partition {pi}",
            "sectionTitle": "",
            "headerId": get_headerId(headerText),
            "numCols": len(tableData[0]),
            "numDataRows": len(tableData),
            "numHeaderRows": len(tableHeaders),
            "numericColumns": [],
            "numTables": mergetable.get("numTables", 1),
            "tableHeaders": tableHeaders,
            "tableData": tableData,
            "pivots": mergetable.get("pivots", [mergetable.get("pivot")]),
            "partColAligns": [
                {
                    "tableIndex": mergetable["tableIndex"],
                    "tableHeaders": mergetable["tableHeaders"],
                    "local": mergetable["partColAlign"],
                    "global": {
                        pc: mergetable["columnIndexOffset"] + c
                        for pc, c in mergetable["partColAlign"].items()
                        if c is not None
                    },
                }
            ],
        }

    tableHeaders = list(
        align_columns(table["tableHeaders"], table["partColAlign"], empty_cell)
    )
    tableHeaders = get_top_headers(
        tableHeaders, merge_headers=mergetable["tableHeaders"]
    )
    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
    )

    for row in align_columns(table["tableData"], table["partColAlign"], empty_cell):
        mergetable["tableData"].append(row)

    mergetable.update(
        {
            "tableHeaders": tableHeaders,
            "headerId": get_headerId(headerText),
            "numDataRows": len(mergetable["tableData"]),
            "numTables": mergetable["numTables"] + table.get("numTables", 1),
            "pivots": mergetable["pivots"] + table.get("pivots", [table.get("pivot")]),
            "partColAligns": mergetable["partColAligns"]
            + [
                {
                    "tableIndex": table["tableIndex"],
                    "tableHeaders": table["tableHeaders"],
                    "local": table["partColAlign"],
                    "global": {
                        pc: table["columnIndexOffset"] + c
                        for pc, c in table["partColAlign"].items()
                        if c is not None
                    },
                }
            ],
        }
    )
    return mergetable
