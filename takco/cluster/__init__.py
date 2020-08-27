from typing import List

from pathlib import Path
import logging as log
import warnings


from .headerunions import combine_by_first_header
from .context import tables_add_context_rows
from .clustering import all_matchers

from . import clustering


def table_get_headerId(table):
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


def cluster(
    tables, dirpath, matcher_kwargs, agg_func=None, agg_threshold=0, edge_exp=1
):
    """Run clustering
    
    Args:
        tables: Tables to cluster
        
    """
    import tqdm
    import pandas as pd

    tables = list(tables)
    assert len(tables)

    # TODO: parallelize map_partition

    log.info(f"Using matchers {matcher_kwargs}")

    matchers = clustering.matcher_add_tables(tables, dirpath, matcher_kwargs)
    for m in matchers:
        m.index()
    matcher_names = list(matcher_kwargs)

    # Get blocked column match scores
    # TODO: parallelize map_partition
    table_indices = set(t["tableIndex"] for t in tables)

    log.info(f"Blocking and matching with {len(table_indices)} indexes")
    sims = {}
    for m, i, s in clustering.yield_blocked_matches(
        table_indices, dirpath, matcher_kwargs
    ):
        sims.setdefault(m, {})[i] = s
    sims = pd.DataFrame.from_dict(sims)
    sims.index.names = ["ti1", "ti2", "ci1", "ci2"]
    sims.columns = matcher_names
    log.info(f"Computed {len(sims)} column similarities")

    sims.to_csv(Path(dirpath) / Path("sims.csv"))

    log.info(f"Aggregating matcher results using `{agg_func} > {agg_threshold}` ")
    aggsim = clustering.aggregate_similarities(sims, agg_func)
    aggsim = aggsim[aggsim > agg_threshold]

    # Compute soft column alignment jaccard
    import sqlite3

    con = sqlite3.connect(Path(dirpath) / Path("indices.sqlite"))
    n = pd.read_sql("select i,numCols from indices", con).set_index("i")["numCols"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        tqdm.tqdm.pandas(desc="Aggregating column scores")
    aligned_total = aggsim.groupby(level=[0, 1]).progress_aggregate(
        clustering.max_align, return_total=True
    )
    j = (
        pd.DataFrame({"total": aligned_total})
        .join(n.rename("n1"), on="ti1")
        .join(n.rename("n2"), on="ti2")
    )
    tablesim = j["total"] / (j["n1"] + j["n2"] - j["total"])
    # TODO: end of parallel loop

    tablesim[tablesim < 0] = 0
    louvain_partition = louvain(tablesim, edge_exp=edge_exp)
    log.info(f"Found {len(louvain_partition)} clusters")

    # Cluster columns
    # TODO: paralellize?
    from sklearn.cluster import AgglomerativeClustering

    clus = AgglomerativeClustering(
        affinity="precomputed",
        linkage="complete",
        n_clusters=None,
        distance_threshold=1,
    )
    ti_pi = {}
    pi_ncols = {}
    ci_pci = {}
    for pi, part in enumerate(tqdm.tqdm(louvain_partition, desc="Clustering columns")):
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
            ub_sims = {i: {} for i, _ in enumerate(matcher_names)}
            for m, i, s in tablepairs_matches:
                ub_sims.setdefault(m, {})[i] = s

            if all(ub_sims.values()):
                log.debug(f"ub_sims {ub_sims}")
                ub_sims = pd.DataFrame.from_dict(ub_sims)
                ub_sims.index.names = ["ti1", "ti2", "ci1", "ci2"]
                ub_sims.columns = matcher_names
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

    # TODO: serialize partitions & cluster alignments for UI

    # TODO: parallelize map_partition
    for table in tables:
        table["part"] = ti_pi[table["tableIndex"]]
        #         assert all( len(row)==table['numCols'] for row in table['tableData'] ) # fails
        ci_range = range(
            table["columnIndexOffset"], table["columnIndexOffset"] + table["numCols"]
        )
        pci_c = {ci_pci[ci]: c for c, ci in enumerate(ci_range) if ci in ci_pci}
        table["partColAlign"] = {
            pci: pci_c.get(pci, None) for pci in range(pi_ncols[table["part"]])
        }
        log.debug(
            f"Table {table['tableIndex']} has pci_c {pci_c}, partColAlign {table['partColAlign']}"
        )

    # TODO: parallelize foldby
    pi_mergetable = {}
    for table in tables:
        pi = table["part"]
        pi_mergetable[pi] = (
            clustering.merge_partition_tables(pi_mergetable[pi], table)
            if (pi in pi_mergetable)
            else table
        )

    yield from pi_mergetable.values()
