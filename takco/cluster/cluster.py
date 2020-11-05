from typing import List, Dict, Collection, Set, Tuple, Iterator, Any
from pathlib import Path
import logging as log
from collections import Counter
import warnings
import sqlite3
import hashlib
import contextlib
import time
import itertools

from .matchers import Matcher, ScoredMatch

Table = dict

try:
    from pandas import DataFrame, Series  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
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

    def __repr__(self):
        return "Timer(%s)" % ", ".join(f"{k}={v:1.1e}" for k, v in self.items())


def progress(it, desc=None):
    if log.getLogger().level <= log.INFO:
        try:
            import tqdm  # type: ignore

            return tqdm.tqdm(it, desc=desc)
        except:
            pass
    return it


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def get_table_ids(tables):
    for t in tables:
        cols = range(t["columnIndexOffset"], t["columnIndexOffset"] + t["numCols"])
        yield int(t["tableIndex"]), set(cols)


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
    import igraph as ig  # type: ignore

    # Make graph
    G = ig.Graph(
        edges=tablesim.index, edge_attrs={"weight": tablesim.values ** edge_exp}
    )
    log.info("Created graph %s", G.summary().replace("\n", " "))
    louvain_partition = G.community_multilevel(
        weights=G.es["weight"],
        return_levels=False,  # pylint: disable=unsubscriptable-object
    )
    return [[int(p) for p in part] for part in louvain_partition]


def matcher_add_tables(tables: Collection[Table], matchers: List[Matcher]):
    """Add tables to matchers

    Args:
        tables: Tables
        matchers: Matcher instances

    Returns:
        Updated matchers
    """
    with contextlib.ExitStack() as matcherstack:
        matchers = [matcherstack.enter_context(m) for m in matchers]
        for table in progress(tables, "Loading tables into matchers"):
            for m in matchers:
                m.add(table)
        return matchers


def get_tablesims(
    tables: Collection[Table],
    tableid_colids: Dict[int, Set[int]],
    matchers: List[Matcher],
    agg_func: str,
    agg_threshold: float,
    filters: List[Matcher] = (),
    align_columns: str = "greedy",
    align_width_norm: str = "jacc",
    align_use_total_width: bool = True,
):
    """Block and match tables for column sims, then aggregate and threshold them.

    For alignment, see :meth:`takco.cluster.aggregate_aligned_column_sims` .

    Args:
        tables: Tables
        tableid_colids: Mapping of global table IDs to column IDs
        matchers: Matchers
        agg_func: Aggregation function for :meth:`takco.cluster.aggregate_match_sims`
        agg_threshold: Threshold value
        align_columns ({'max1', 'max2', 'greedy'}): Alignment method.
        align_width_norm ({'wide', 'narrow', 'jacc'}): Table width difference normalisation method.
        align_use_total_width: Whether to use total table width. Defaults to True. 

    Yields:
        Column similarity dataframes
    """

    simdf = make_blocked_matches_df(tables, tableid_colids, matchers, filters)
    if simdf is not None:
        simdf = aggregate_match_sims(simdf, agg_func)
        tablesim = aggregate_aligned_column_sims(
            simdf, 
            tableid_colids,
            align_columns = align_columns,
            align_width_norm = align_width_norm,
            align_use_total_width = align_use_total_width,
        )
        yield tablesim[tablesim > agg_threshold]


def make_blocked_matches_df(
    tables: Collection[Table],
    tableid_colids: Dict[int, Set[int]],
    matchers: List[Matcher],
    filters: List[Matcher] = (),
):
    """Yield a dataframe for similarities from blocked matches
    
    Args:
        tableid_colids: Mapping of table indices to column indices
        matchers: Matcher instances
    """

    with contextlib.ExitStack() as matcherstack:
        matchers = [matcherstack.enter_context(m) for m in matchers]
        filters = [matcherstack.enter_context(m) for m in filters]

        matches = yield_blocked_matches(tables, tableid_colids, matchers, filters)
        simscore = {mi: {} for mi, _ in enumerate(matchers)}  # type: ignore
        for mi, indexes, score in matches:
            simscore[mi][indexes] = score

        log.debug(f"Creating dataframe of column match scores")
        simdf = pd.DataFrame.from_dict(simscore)
        if len(simdf):
            simdf.index.names = ["ti1", "ti2", "ci1", "ci2"]
            simdf.columns = [m.name for m in matchers]
            return simdf


def aggregate_match_sims(simdf: DataFrame, agg_func: str):
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
            agg = funcs[agg_func](*(simdf[c] for c in simdf))  # type: ignore
        else:
            agg = simdf.eval(agg_func, local_dict=funcs, engine="python")
        return pd.Series(agg, index=simdf.index)


def greedy_align(g: Series) -> float:
    """Greedy alignment score for soft jaccard index

    Args:
        g: Sorted series of (_, _, left, right) -> value

    Returns:
        Total alignment score
    """
    lr, rl, t = {}, {}, 0
    for (_, _, l, r), v in g.iteritems():
        if (l not in lr) and (r not in rl):
            lr[l], rl[r], t = r, l, t + v
    return t


def aggregate_aligned_column_sims(
    aggsim: DataFrame,
    tableid_colids: Dict[int, Set[int]],
    align_columns: str = "greedy",
    align_width_norm: str = "jacc",
    align_use_total_width: bool = True,
) -> DataFrame:
    """Aggregate column similarities.

    To create a table similarity graph, the column similarities need to be aggregated.
    This aggregation must be based on several assumptions which influence the accuracy
    and speed.

    First of all, how to align columns. Do you allow multiple columns from one table to
    align with a single column in the other? In that case, choose one of the fast 'max'
    values for the ``align`` parameter, depending on whether to allow the first or 
    the second table to match multiple columns in the other. 
    
    Otherwise, choose 'greedy'. This calculates a kind of soft-jaccard score.
    In that case, you'll need to decide how to handle columns for which no similarity score
    could be calculated. To ignore those columns, set ``align_use_total_width=False``.
    Otherwise, they will be assumed to be non-matching.
    Also, the alignment score is then normalized. This expresses your view about whether
    you want wide and narrow tables to match. If so, choose 'wide'. If you want the
    tables to have the similar widths, choose 'narrow'. For a middle ground, choose
    'jacc', which will calculate ``score / (cols1 + cols2 - score)``.

    Args:
        aggsim: Column similarities (aggregated match scores)
        tableid_colids: Global column IDs per table ID
        align_columns ({'max1', 'max2', 'greedy'}): Alignment method. Defaults to 'greedy'.
        align_width_norm ({'wide', 'narrow', 'jacc'}): Table width difference normalisation method. Defaults to 'jacc'.
        align_use_total_width: Whether to use total table width. Defaults to True.

    Returns:
        Table similarities
    """
    assert align_columns in {"max1", "max2", "greedy"}
    assert align_width_norm in {"wide", "narrow", "jacc"}

    try:  # Maybe show progress
        if log.getLogger().level <= log.INFO:
            import warnings, tqdm

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tqdm.tqdm.pandas(desc="Aggregating column scores")
        agg = lambda gs, align: gs.progress_aggregate(align)
    except Exception as e:
        log.debug(f"When trying to show aggregation progress, {e}")
        agg = lambda gs, align: gs.agg(align)

    if align_columns == "greedy":
        # Compute soft column alignment jaccard
        aggsim.sort_values(ascending=False, inplace=True)
        total = agg(aggsim.groupby(level=[0, 1]), greedy_align)
        if align_use_total_width:
            # Use total column widths
            table_numcols = pd.Series({ti: len(cis) for ti, cis in tableid_colids.items()})
            j = (
                pd.DataFrame({"total": total})
                .join(table_numcols.rename("n1"), on="ti1")
                .join(table_numcols.rename("n2"), on="ti2")
            )
        else:
            # Only use number of matched columns
            n1 = aggsim.groupby(level=[0, 1, 2]).count().groupby(level=[0, 1]).first()
            n2 = aggsim.groupby(level=[0, 1, 3]).count().groupby(level=[0, 1]).first()
            j = pd.DataFrame({"total": total, "n1": n1, "n2": n2})
        
        # 
        if align_width_norm == "jacc":
            return j["total"] / (j["n1"] + j["n2"] - j["total"])
        elif align_width_norm == "wide":
            return j["total"] / j[["n1", "n2"]].max(1)
        elif align_width_norm == "narrow":
            return j["total"] / j[["n1", "n2"]].min(1)

    else:
        level = 2 if align_columns == "max1" else 3
        return aggsim.groupby(level=[0, 1, level]).max().groupby(level=[0, 1]).mean()

    

    


def yield_blocked_matches(
    tables: Collection[Table],
    tableid_colids: Dict[int, Set[int]],
    matchers: List[Matcher],
    filters: List[Matcher] = (),
):
    """Match table columns using matchers

    Args:
        tableid_colids: Mapping of table indices to column indices
        matchers: Matcher instances

    Yields:
        (matcher index, ``(t1, t2, c1, c2)``, score)
    """

    timer = Timer()

    partition_tableid_colids = dict(get_table_ids(tables))

    for matcher in matchers:
        log.debug(f"Preparing block for matcher {matcher.name}")
        with timer.track(f"prepare_{matcher.name}"):
            matcher.prepare_block(partition_tableid_colids)

    table_block = {}
    for ti, cis in progress(partition_tableid_colids.items(), "Blocking"):
        block = set()
        for matcher in matchers:
            with timer.track(f"block_{matcher.name}"):
                for bi in matcher.block(ti, cis):
                    block.add(bi)
        
        table_block[ti] = block - set([ti])

    block_size = pd.Series([len(b) for b in table_block.values()])
    mean, std = block_size.agg(['mean', 'std'])
    log.debug(f"Found {block_size.sum()} pairs; {mean:.0f} ± {std:.0f} per table")

    tableid_colids_pairs = {
        (ti1,ti2): ((ti1, partition_tableid_colids[ti1]), (ti2, tableid_colids[ti2]))
        for ti1, block in table_block.items()
        for ti2 in block
    }

    # Filter blocks
    if filters:
        filtered_tableid_colids_pairs = {}
        for filt in filters:
            it = iter(tableid_colids_pairs.values())
            with timer.track(f"filter_{filt.name}"):
                while True:
                    chunk = tuple(itertools.islice(it, 10**6)) # chunk a million
                    if not chunk:
                        break
                    for (ti1, ti2, _, _), score in filt.match(chunk):
                        if score > 0:
                            pair = tableid_colids_pairs[(ti1, ti2)]
                            filtered_tableid_colids_pairs[(ti1, ti2)] = pair

        log.debug(f"Filtered down to {len(filtered_tableid_colids_pairs)} pairs")
    else:
        filtered_tableid_colids_pairs = tableid_colids_pairs
    
    for mi, matcher in enumerate(matchers):
        it = iter(filtered_tableid_colids_pairs.values())
        with timer.track(f"match_{matcher.name}"):
            while True:
                chunk = tuple(itertools.islice(it, 10**6)) # chunk a million
                if not chunk:
                    break
                for indices, score in matcher.match(chunk):
                    yield mi, indices, score

    log.debug(f"times: {timer}")


def cluster_partition_columns(
    partitions: Collection[Tuple[int, List[int]]],
    clus: AgglomerativeClustering,
    tableid_colids: Dict[int, Set[int]],
    matchers: List[Matcher],
    agg_func: str,
    agg_threshold_col: float,
) -> Iterator[Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, Any]]]:
    """Cluster columns withing a partition

    Args:
        partitions: Pairs of (partition index, [table indices])
        clus: Scikit-learn clustering instance
        aggsim: Aggregated similarities Series of {(p1,p2,c1,c2): normalized sim }
        agg_func: Numexpr-style aggregation function
        agg_threshold_col: Aggregation threshold value for column similarities
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

            # Match all table pairs
            tableid_colids_pairs = [
                ((t1, tableid_colids[t1]), (t2, tableid_colids[t2]))
                for t1 in part
                for t2 in part
                if t2 >= t1
            ]

            # Make a dataframe of all similarities
            def yield_tablepairs_matches():
                for mi, matcher in enumerate(matchers):
                    pairs = progress(
                        tableid_colids_pairs, f"Matching with {matcher.name}"
                    )
                    for indices, score in matcher.match(pairs):
                        yield mi, indices, score

            simscore = {mi: {} for mi, _ in enumerate(matchers)}  # type: ignore
            for mi, indexes, score in yield_tablepairs_matches():
                simscore[mi][indexes] = score
            if not simscore:
                continue

            sims = pd.DataFrame.from_dict(simscore)
            sims.index.names = ["ti1", "ti2", "ci1", "ci2"]
            sims.columns = [m.name for m in matchers]
            colsim = aggregate_match_sims(sims, agg_func)
            colsim = colsim[colsim > agg_threshold_col]

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
    partcol_rank = {  # type: ignore
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
