from typing import List, Dict, Container, Tuple, Iterator
from pathlib import Path
import logging as log
from collections import Counter
import warnings
import sqlite3
import hashlib
import contextlib
import time

from .matchers import Matcher

try:
    from pandas import DataFrame, Series
    import pandas as pd
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
except:
    import typing

    DataFrame = typing.Any
    Series = typing.Any
    AgglomerativeClustering = typing.Any


class Timer(dict):
    @contextlib.contextmanager
    def track(self, name):
        t0 = time.time()
        try:
            yield self
        finally:
            self[name] = self.get(name, 0) + (time.time() - t0)


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


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
    return [tuple(part) for part in louvain_partition]


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


def get_colsims(
    table_indices: Container[int],
    matchers: List[Matcher],
    agg_func: str,
    agg_threshold: float,
    table_numcols: Dict[int, int],
):
    """Block and match tables for column sims, then aggregate and threshold them. """
    simdf = make_blocked_matches_df(table_indices, matchers)
    if simdf is not None:
        simdf = aggregate_similarities(simdf, agg_func)
        yield threshold_softjacc(simdf, agg_threshold, table_numcols)


def make_blocked_matches_df(table_indices: Container[int], matchers: List[Matcher]):
    """Yield a dataframe for similarities from blocked matches
    
    Args:
        table_indices: Set of table indices
        matchers: Matcher instances
    """

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
            return simdf


def aggregate_similarities(simdf: DataFrame, agg_func: str):
    """Aggregate similarities using a numexpr aggregation function.

    Extra functions available: ``max(*a)``, ``min(*a)``, ``mean(*a)``, ``pow(a,b)``.

    See also:
        `Pandas eval <https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html#supported-syntax>`_
        `Numexpr <https://numexpr.readthedocs.io/>`_

    Args:
        simdf: DataFrame of similarities, where columns are matcher names.
        agg_func: Numexpr-style function.
    """

    import warnings, tqdm

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        funcs = {
            "max": lambda *args: np.nanmax(args, axis=0),
            "min": lambda *args: np.nanmin(args, axis=0),
            "mean": lambda *args: np.nanmean(args, axis=0),
            "pow": lambda a, b: a ** b,
        }
        if agg_func in funcs:
            agg = funcs[agg_func](*(simdf[c] for c in simdf))
        else:
            agg = simdf.eval(agg_func, local_dict=funcs, engine="python")
        return pd.Series(agg, index=simdf.index)


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


def threshold_softjacc(aggsim, agg_threshold, table_numcols):

    # Compute soft column alignment jaccard
    if log.getLogger().level < log.INFO:
        import warnings, tqdm

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            tqdm.tqdm.pandas(desc="Aggregating column scores")

    aligned_total = aggsim.groupby(level=[0, 1]).aggregate(max_align, return_total=True)
    aligned_total[aligned_total < 0] = 0
    j = (
        pd.DataFrame({"total": aligned_total})
        .join(table_numcols.rename("n1"), on="ti1")
        .join(table_numcols.rename("n2"), on="ti2")
    )
    tablesim = j["total"] / (j["n1"] + j["n2"] - j["total"])
    return tablesim[tablesim > agg_threshold]


def make_column_index_df(tables):
    """Yield a dataframe for the tables' column indexes"""

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


def yield_blocked_matches(table_indices: Container[int], matchers: List[Matcher]):
    """Match table columns using matchers

    Args:
        table_indices: Table indices
        matchers: Matcher instances

    Yields:
        (matcher index, ``(t1, t2, c1, c2)``, score)
    """
    table_indices = set(table_indices)

    timer = Timer()

    for matcher in matchers:
        log.debug(f"Preparing block for matcher {matcher.name}")
        with timer.track(f"prepare-{matcher.name}"):
            matcher.prepare_block(table_indices)

    table_block = {}
    for ti in table_indices:
        block = set()
        for matcher in matchers:
            with timer.track(f"block-{matcher.name}"):
                block |= set(matcher.block(ti) or [])
        table_block[ti] = block

    table_index_pairs = set(
        (ti1, ti2) for ti1, block in table_block.items() for ti2 in block if ti1 != ti2
    )
    for mi, matcher in enumerate(matchers):
        log.debug(f"Matching {len(table_index_pairs)} pairs with {matcher.name}...")
        if log.getLogger().level < log.INFO:
            try:
                import tqdm

                table_index_pairs = tqdm.tqdm(table_index_pairs)
            except:
                pass

        with timer.track(f"match-{matcher.name}"):
            for pairs, score in matcher.match(table_index_pairs):
                yield mi, pairs, score

    log.debug(f"times: {timer}")


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
        for indexes, score in matcher.match(table_index_pairs):
            yield mi, indexes, score


def cluster_partition_columns(
    partitions: Container[Tuple[int, List[int]]],
    clus: AgglomerativeClustering,
    agg_func: str,
    agg_threshold: float,
    matchers: List[Matcher],
) -> Iterator[Tuple[Dict[int, int], Dict[int, int], Dict[int, Dict[int, int]]]]:
    """Cluster columns withing a partition

    Args:
        partitions: Pairs of (partition index, [table indices])
        clus: Scikit-learn clustering instance
        aggsim: Aggregated similarities Series of {(p1,p2,c1,c2): normalized sim }
        agg_func: Numexpr-style aggregation function
        agg_threshold: Aggregation threshold value for column similarities
        matchers: Matcher instances

    Yields:
        ``({table:partition}, {partition: ncols}, {column:{column index: partition column index}})``
    """
    ti_pi = {}
    pi_ncols = {}
    ci_pci = {}
    pi_colsim = {}

    with contextlib.ExitStack() as matcherstack:
        entered = [matcherstack.enter_context(m) for m in matchers]

        for pi, part in partitions:
            for ti in part:
                ti_pi[ti] = pi

            # Match unblocked table pairs
            pairs = [(t1, t2) for t1 in part for t2 in part if t2 >= t1]
            # Make a dataframe of extra similarities
            matches = yield_tablepairs_matches(pairs, entered)
            sims = {mi: {} for mi, _ in enumerate(matchers)}
            for mi, indexes, score in matches:
                sims[mi][indexes] = score
            if not sims:
                continue

            sims = pd.DataFrame.from_dict(sims)
            sims.index.names = ["ti1", "ti2", "ci1", "ci2"]
            sims.columns = [m.name for m in matchers]
            colsim = aggregate_similarities(sims, agg_func)
            colsim = colsim[colsim > agg_threshold]

            for ti, ci_range in entered[0].get_columns_multi(part):
                for ci in ci_range:
                    colsim[(ti, ti, ci, ci)] = 1

            pi_colsim[pi] = colsim

            if not len(colsim):
                # TODO: find out what's going on here.
                # there should always be self-similarities
                pi_ncols[pi] = 0
                log.warning(f"No similarities for partition {pi}: {part}")
            else:
                col_clustercol = cluster_columns(colsim.reset_index(), clus, pi=pi)
                ci_pci.update(col_clustercol)
                ncols = len(set(col_clustercol.values()))
                pi_ncols[pi] = ncols
                log.debug(
                    f"Partition {pi} has {len(part)} tables and {ncols} column clusters"
                )

    yield ti_pi, pi_ncols, ci_pci, pi_colsim


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
    if "part" not in table:
        return table

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

    # Make symmetric distance matrix
    d = 1 - colsim.unstack().sort_index(0).sort_index(1).fillna(0)
    d = pd.DataFrame(np.minimum(d, d.T))

    try:
        partcols = clus.fit_predict(d)
    except:
        partcols = range(len(d.index))

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
