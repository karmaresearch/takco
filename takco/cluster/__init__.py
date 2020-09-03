from typing import List

from pathlib import Path
import logging as log


from .headerunions import combine_by_first_header
from .context import tables_add_context_rows
from . import clustering


def table_get_headerId(table):
    """Get the hash for a table header (create it if it isn't set)"""
    if "headerId" not in table:
        tableHeaders = table["tableHeaders"]
        headerText = tuple(
            tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
        )
        return util.get_headerId(headerText)
    else:
        return table["headerId"]


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
    ).set_index("i")
    log.debug(f"Indexed {len(df)} tables")
    yield df


def make_blocked_matches_df(table_indices, dirpath, matcher_kwargs):
    """Yield a dataframe for similarities from blocked matches"""
    import pandas as pd

    matches = clustering.yield_blocked_matches(table_indices, dirpath, matcher_kwargs)
    simdf = {mi: {} for mi, _ in enumerate(matcher_kwargs)}
    for mi, indexes, score in matches:
        simdf[mi][indexes] = score
    simdf = pd.DataFrame.from_dict(simdf)
    simdf.index.names = ["ti1", "ti2", "ci1", "ci2"]
    simdf.columns = list(matcher_kwargs)
    yield simdf


def make_aggsim_df(similarity_dataframes):
    """Yield aggregated similarity dataframe"""
    pass


def cluster_partition_columns(iparts, clus, aggsim, agg_func, dirpath, matcher_kwargs):
    ti_pi = {}
    pi_ncols = {}
    ci_pci = {}
    for pi, part in iparts:
        for ti in part:
            ti_pi[ti] = pi

        colsim = aggsim.loc[part, part, :, :]

        # Match unblocked table pairs
        blocked_pairs = colsim.groupby(level=[0, 1]).agg("any")
        blocked_pairs = blocked_pairs[blocked_pairs]
        unblocked_pairs = [
            (ti1, ti2)
            for ti1 in part
            for ti2 in part
            if ti2 >= ti1
            and not any(i in blocked_pairs.index for i in [(ti1, ti2), (ti2, ti1)])
        ]
        if len(unblocked_pairs):
            log.debug(f"Partition {pi}: {part} has unblocked pairs {unblocked_pairs}")
            tablepairs_matches = clustering.yield_tablepairs_matches(
                unblocked_pairs, dirpath, matcher_kwargs
            )
            ub_sims = {i: {} for i, _ in enumerate(matcher_kwargs)}
            for m, i, s in tablepairs_matches:
                ub_sims.setdefault(m, {})[i] = s

            if all(ub_sims.values()):
                log.debug(f"ub_sims {ub_sims}")
                ub_sims = pd.DataFrame.from_dict(ub_sims)
                ub_sims.index.names = ["ti1", "ti2", "ci1", "ci2"]
                ub_sims.columns = list(matcher_kwargs)
                ub_aggsim = clustering.aggregate_similarities(ub_sims, agg_func)
                ub_aggsim = ub_aggsim[ub_aggsim > agg_threshold]
                colsim = pd.concat([colsim, ub_aggsim])

        if not len(colsim):
            # TODO: find out what's going on here.
            # there should always be self-similarities
            pi_ncols[pi] = 0
            log.warning(f"No similarities for partition {pi}: {part}")
        else:
            col_cluster = clustering.cluster_columns(colsim.reset_index(), clus, pi=pi)
            ci_pci.update(col_cluster)
            ncols = len(set(col_cluster.values()))
            pi_ncols[pi] = ncols
            log.debug(
                f"Partition {pi} has {len(part)} tables and {ncols} column clusters"
            )
    yield ti_pi, pi_ncols, ci_pci
