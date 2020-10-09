from typing import List, Dict, Container, Tuple, Iterator
from pathlib import Path
import logging as log
from collections import Counter
import warnings
import sqlite3
import hashlib
import contextlib

from .matchers import Matcher

try:
    from pandas import DataFrame, Series
    from sklearn.cluster import AgglomerativeClustering
except:
    import typing

    DataFrame = typing.Any
    Series = typing.Any
    AgglomerativeClustering = typing.Any


def louvain(tablesim, edge_exp=1) -> List[List[int]]:
    """Louvain clustering

    .. math::

        Q = \\frac{1}{2m}\\sum\\limits_{ij}\\bigg[A_{ij} - \\frac{k_i k_j}{2m}\\bigg]\\delta (c_i,c_j)

    where

    - :math:`A_{ij}` represents the edge weight between nodes :math:`i` and :math:`j`;
    - :math:`k_i` and :math:`k_j` are the sum of the weights of the edges attached to nodes :math:`i` and :math:`j`, respectively;
    - :math:`m` is the sum of all of the edge weights in the graph;
    - :math:`c_i` and :math:`c_j` are the communities of the nodes; and
    - :math:`\\delta` is the Kronecker delta function (:math:`\\delta_{x,y}= 1` if :math:`x=y`, :math:`0` otherwise).

    See also:

        - `igraph.Graph.community_multilevel <https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel>`_
        - `Louvain modularity <https://en.wikipedia.org/wiki/Louvain_modularity>`_

    """
    import igraph as ig

    # Make graph
    G = ig.Graph(
        edges=tablesim.index, edge_attrs={"weight": tablesim.values ** edge_exp}
    )
    log.info("Created graph %s", G.summary().replace("\n", " "))
    louvain_partition = G.community_multilevel(
        weights=G.es["weight"], return_levels=False
    )
    return louvain_partition


def max_align(g: Series, return_total=False):
    """Maximum alignment score for soft jaccard index

    Args:
        g (Series): Series of (_, _, left, right) -> value
        return_total: Whether to return total or alignment.

    Returns:
        Either the total alignment score, or the max alignment dict ``{left: right}``
    """
    lr, rl, t = {}, {}, 0
    for (_, _, l, r), v in g.sort_values(ascending=False).items():
        if (l not in lr) and (r not in rl):
            lr[l], rl[r], t = r, l, t + v
    return t if return_total else lr


def make_column_index_df(tables):
    """Yield a dataframe for the tables' column indexes"""
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "i": t["tableIndex"],
                "numCols": t["numCols"],
                "columnIndexOffset": t["columnIndexOffset"],
            }
            for t in tables
        ]
    )
    if len(df):
        df = df.set_index("i")
        log.debug(f"Indexed {len(df)} tables")
        yield df


def aggregate_similarities(sims: DataFrame, agg_func: str):
    """Aggregate similarities using a numexpr aggregation function.

    Extra functions available: ``max(*a)``, ``min(*a)``, ``mean(*a)``, ``pow(a,b)``.

    See also:
        `Pandas eval <https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#supported-syntax>`_
        `Numexpr <https://numexpr.readthedocs.io/>`_

    Args:
        sims: DataFrame of similarities, where columns are matcher names.
        agg_func: Numexpr-style function.
    """
    import pandas as pd
    import numpy as np

    funcs = {
        "max": lambda *args: np.nanmax(args, axis=0),
        "min": lambda *args: np.nanmin(args, axis=0),
        "mean": lambda *args: np.nanmean(args, axis=0),
        "pow": lambda a, b: a ** b,
    }
    if agg_func in funcs:
        agg = funcs[agg_func](*(sims[c] for c in sims))
    else:
        agg = sims.eval(agg_func, local_dict=funcs, engine="python")
    return pd.Series(agg, index=sims.index)


def make_blocked_matches_df(table_indices: Container[int], matchers: List[Matcher]):
    """Yield a dataframe for similarities from blocked matches
    
    Args:
        table_indices: Set of table indices
        matchers: Matcher instances
    """
    import pandas as pd

    with contextlib.ExitStack() as matcherstack:
        matchers = [matcherstack.enter_context(m) for m in matchers]

        matches = yield_blocked_matches(table_indices, matchers)
        simdf = {mi: {} for mi, _ in enumerate(matchers)}
        for mi, indexes, score in matches:
            simdf[mi][indexes] = score
        simdf = pd.DataFrame.from_dict(simdf)
        if len(simdf):
            simdf.index.names = ["ti1", "ti2", "ci1", "ci2"]
            simdf.columns = [m.name for m in matchers]
            yield simdf


def matcher_add_tables(tables: Container[dict], matchers: List[Matcher]):
    """Add tables to matchers

    Args:
        tables: Tables
        matchers: Matcher instances

    Returns:
        Updated matchers
    """
    with contextlib.ExitStack() as matcherstack:
        matchers = [matcherstack.enter_context(m) for m in matchers]
        log.info(f"Loading tables into matchers")
        for table in tables:
            for m in matchers:
                m.add(table)
        return matchers


def cluster_partition_columns(
    iparts: Container[Tuple[int, int]],
    clus: AgglomerativeClustering,
    aggsim: Series,
    agg_func: str,
    agg_threshold: float,
    matchers: List[Matcher],
) -> Iterator[Tuple[Dict[int, int], Dict[int, int], Dict[int, Dict[int, int]]]]:
    """Cluster columns withing a partition

    Args:
        iparts: Pairs of (partition index, [table indices])
        clus: Scikit-learn clustering instance
        aggsim: Aggregated similarities Series of {(p1,p2,c1,c2): normalized sim }
        agg_func: Numexpr-style aggregation function
        agg_threshold: Aggregation threshold value for column similarities
        matchers: Matcher instances

    Yields:
        ``({table:partition}, {partition: ncols}, {column:{column index: partition column index}})``
    """
    import pandas as pd

    partition_has_unblocked_pairs = {}
    unblocked_pairs = []
    ti_pi = {}
    for pi, part in iparts:
        for ti in part:
            ti_pi[ti] = pi

        colsim = aggsim.loc[part, part, :, :]

        # Match unblocked table pairs
        blocked_pairs = colsim.groupby(level=[0, 1]).agg("any")
        blocked_pairs = blocked_pairs[blocked_pairs]
        partition_unblocked_pairs = [
            (ti1, ti2)
            for ti1 in part
            for ti2 in part
            if ti2 >= ti1
            and not any(i in blocked_pairs.index for i in [(ti1, ti2), (ti2, ti1)])
        ]
        if len(partition_unblocked_pairs):
            log.debug(f"Partition {pi} has {len(unblocked_pairs)} unblocked pairs")
            unblocked_pairs += partition_unblocked_pairs
            partition_has_unblocked_pairs[pi] = True

    with contextlib.ExitStack() as matcherstack:
        matchers = [matcherstack.enter_context(m) for m in matchers]
        # Make a dataframe of extra similarities
        tablepairs_matches = yield_tablepairs_matches(unblocked_pairs, matchers)

    unblocked_sims = {i: {} for i, _ in enumerate(matchers)}
    for m, (ti1, ti2, ci1, ci2), s in tablepairs_matches:
        pi = ti_pi[ti1]
        unblocked_sims.setdefault(m, {})[(pi, ti1, ti2, ci1, ci2)] = s

    if all(unblocked_sims.values()):
        unblocked_sims = pd.DataFrame.from_dict(unblocked_sims)
        unblocked_sims.index.names = ["pi", "ti1", "ti2", "ci1", "ci2"]
        unblocked_sims.columns = [m.name for m in matchers]
        unblocked_aggsim = aggregate_similarities(unblocked_sims, agg_func)
        unblocked_aggsim = unblocked_aggsim[unblocked_aggsim > agg_threshold]

    pi_ncols = {}
    ci_pci = {}
    for pi, part in iparts:
        colsim = aggsim.loc[part, part, :, :]
        if partition_has_unblocked_pairs.get(pi):
            colsim = pd.concat([colsim, unblocked_aggsim.loc[pi]])

        if not len(colsim):
            # TODO: find out what's going on here.
            # there should always be self-similarities
            pi_ncols[pi] = 0
            log.warning(f"No similarities for partition {pi}: {part}")
        else:
            col_cluster = cluster_columns(colsim.reset_index(), clus, pi=pi)
            ci_pci.update(col_cluster)
            ncols = len(set(col_cluster.values()))
            pi_ncols[pi] = ncols
            log.debug(
                f"Partition {pi} has {len(part)} tables and {ncols} column clusters"
            )

    yield ti_pi, pi_ncols, ci_pci


def merge_partition_tables(
    mergetable: dict,
    table: dict,
    mergeheaders_topn: int = None,
    store_align_meta=["tableHeaders"],
) -> dict:
    """Merge tables within partition

    Args:
        mergetable: Big table
        table: Small table
        store_align_meta: Which fields to keep in ``partColAlign`` tables
        mergeheaders_topn: Number of top headers to keep when merging

    Returns:
        Merged table
    """

    empty_cell = {"text": ""}
    pi = table["part"]

    if mergetable.get("type") != "partition":
        # Create new mergetable

        # partColAlign is a mapping from partition cols to local cols
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
        tableHeaders = get_top_headers(tableHeaders, topn=mergeheaders_topn)
        headerText = tuple(
            tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
        )

        partColAlign = {
            "tableIndex": mergetable["tableIndex"],
            "partcol_local": mergetable["partColAlign"],
            "partcol_global": {
                pci: mergetable["columnIndexOffset"] + c
                for pci, c in mergetable["partColAlign"].items()
                if c is not None
            },
        }
        for field in store_align_meta:
            partColAlign[field] = mergetable[field]

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
            "partColAligns": [partColAlign],
        }

    tableHeaders = list(
        align_columns(table["tableHeaders"], table["partColAlign"], empty_cell)
    )
    tableHeaders = get_top_headers(
        tableHeaders, merge_headers=mergetable["tableHeaders"], topn=mergeheaders_topn
    )
    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
    )

    for row in align_columns(table["tableData"], table["partColAlign"], empty_cell):
        mergetable["tableData"].append(row)

    partColAlign = {
        "tableIndex": table["tableIndex"],
        "partcol_local": table["partColAlign"],
        "partcol_global": {
            pci: table["columnIndexOffset"] + c
            for pci, c in table["partColAlign"].items()
            if c is not None
        },
    }
    for field in store_align_meta:
        partColAlign[field] = table[field]

    mergetable.update(
        {
            "tableHeaders": tableHeaders,
            "headerId": get_headerId(headerText),
            "numDataRows": len(mergetable["tableData"]),
            "numTables": mergetable["numTables"] + table.get("numTables", 1),
            "pivots": mergetable["pivots"] + table.get("pivots", [table.get("pivot")]),
            "partColAligns": mergetable["partColAligns"] + [partColAlign],
        }
    )
    return mergetable


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def yield_blocked_matches(table_indices: Container[int], matchers: List[Matcher]):
    """Match table columns using matchers

    Args:
        table_indices: Table indices
        matchers: Matcher instances

    Yields:
        (matcher index, ``(t1, t2, c1, c2)``, score)
    """
    table_indices = set(table_indices)

    for matcher in matchers:
        log.debug(f"Preparing block for matcher {matcher.name}")
        matcher.prepare_block(table_indices)

    for ti1 in table_indices:
        block = set()
        for matcher in matchers:
            block |= set(matcher.block(ti1) or [])

        log.debug(f"Matching {len(block)} blocked candidates for table {ti1}")

        table_index_pairs = set((ti1, ti2) for ti2 in block if ti1 != ti2)
        for mi, matcher in enumerate(matchers):
            for pairs, score in matcher.match(table_index_pairs):
                yield mi, pairs, score


def yield_tablepairs_matches(
    table_index_pairs: Container[Tuple[int, int]], matchers: List[Matcher]
):
    """Get matches for table pairs

    Args:
        table_index_pairs: Table index pairs
        matchers: Matcher instances

    Yields:
        (matcher index, ``(t1, t2, c1, c2)``, score)
    """
    table_index_pairs = set(table_index_pairs)
    for mi, matcher in enumerate(matchers):
        for pairs, score in matcher.match(table_index_pairs):
            yield mi, pairs, score


def cluster_columns(
    colsim: DataFrame, clus: AgglomerativeClustering, pi=None
) -> Dict[int, int]:
    """Cluster columns from different tables together within a cluster of tables

    Column similarities within one table are set to 0 to prevent different columns
    within one table from linking.

    Args:
        colsim: Dataframe of column similarities
        clus: Agglomerative clustering method
        pi: Partition information (for debugging)
    
    Returns:
        ``{column index: partition column index}``
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
    if any(tableHeaders) and any(merge_headers):
        for merge, hcol in zip(merge_headers[0], zip(*tableHeaders)):
            c = Counter(
                cell.get("text", "").strip()
                for cell in hcol
                if cell.get("text", "").strip()
            )
            if merge:
                c += Counter(merge.get("freq", {}))

            txt = ""
            if c:
                txts, _ = zip(*c.most_common())
                txts_nohidden = [t for t in txts if t and t[0] != "_"]
                txt = "\t".join(t for t in (txts_nohidden or txts)[:topn])
            top.append({"text": txt, "tdHtmlString": f"<th>{txt}</th>", "freq": c})

        return [top]
    else:
        if any(tableHeaders):
            return tableHeaders
        else:
            return merge_headers
