from __future__ import annotations

__doc__ = """
tacko is a modular system for extracting knowledge from tables. For example, you can 
use it to extend Wikidata with information from Wikipedia tables.

See http://karmaresearch.github.io/takco for further documentation.
"""

import typing
from pathlib import Path
import logging as log

from .util import *


class TableSet:
    """A set of tables that can be clustered and linked."""

    @classmethod
    def csvs(
        cls,
        path: Path = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):
        executor, exkw = get_executor_kwargs(executor, assets)

        log.info(f"Reading csv tables")
        import csv

        data = (
            {
                "tableData": [
                    [{"text": c} for c in row] for row in csv.reader(Path(fname).open())
                ],
                "_id": fname,
            }
            for fname in path
        )
        return executor(data, **exkw)

    @classmethod
    def load(
        cls,
        path: Path = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):
        log.debug(f"Loading tables {path} using executor {executor}")
        if all(str(val).endswith("csv") for val in path):
            return cls.csvs(path, executor=executor, assets=assets)

        executor, exkw = get_executor_kwargs(executor, assets)
        return executor._load(path, **exkw)

    @classmethod
    def dataset(
        cls,
        params: Config,
        datadir: Path = None,
        resourcedir: Path = None,
        sample: int = None,
        withgold: bool = False,
        executor: Config = None,
        assets: typing.List[Config] = (),
        tables: typing.Any = None,
    ):
        """Load tables from a dataset

        See also: :mod:`takco.evaluate.dataset`

        Args:
            params: Dataset parameters
            datadir: Data directory
            resourcedir: Resource directory
            sample: Sample N tables
        """
        executor, exkw = get_executor_kwargs(executor, assets)

        import itertools
        from . import evaluate

        params = Config(params, assets)
        log.info(f"Loading dataset {params}")
        ds = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **params)
        return executor(itertools.islice(ds.get_unannotated_tables(), sample), **exkw)

    @classmethod
    def extract(
        cls,
        source: typing.Union[Config, HashBag],
        executor: Config = None,
        assets: typing.List[Config] = (),
        tables: typing.Any = None,
    ):
        """Collect tables from HTML files

        See also: :mod:`takco.extract`

        Args:
            source: Source of HTML files
        """
        if isinstance(source, Config):
            from . import pages

            source.update(dict(executor=executor, assets=assets))
            htmlpages = source.init_class(**pages.__dict__).get()
        elif isinstance(source, HashBag):
            htmlpages = source
        else:
            htmlpages = HashBag(source)

        from .extract import extract_tables

        tables = htmlpages._pipe(extract_tables, desc="Extract")
        return tables

    def reshape(
        tables: TableSet,
        restructure: bool = True,
        unpivot_configs: typing.List[Config] = (),
        centralize_pivots: bool = False,
        split_compound_columns: bool = False,
        assets: typing.List[Config] = (),
    ):
        """Reshape tables

        See also: :mod:`takco.reshape`

        Args:
            restructure: Whether to restructure tables heuristically
            unpivot_configs: Use a subset of available heuristics
            centralize_pivots: If True, find pivots on unique headers instead of tables
            split_compound_columns: Whether to split compound columns (with NER)

        """
        from . import reshape
        from . import link

        if restructure:
            tables = tables._pipe(reshape.restructure, desc="Restructure")

        unpivot_heuristics = {}
        for h in unpivot_configs:
            name = h.get("name", h.get("class"))
            unpivot_heuristics[name] = Config(h, assets).init_class(
                **{**reshape.__dict__, **link.__dict__}
            )

        if unpivot_heuristics:
            log.info(f"Reshaping with heuristics: {', '.join(unpivot_heuristics)}")

            headerId_pivot = None
            if centralize_pivots:
                tables.persist()
                headers = tables._fold(reshape.table_get_headerId, lambda x, y: x)
                headers = headers._pipe(reshape.get_headerobjs)

                pivots = headers._pipe(
                    reshape.yield_pivots, heuristics=unpivot_heuristics
                )

                headerId_pivot = {p["headerId"]: p for p in pivots}
                log.info(f"Found {len(headerId_pivot)} pivots")

            tables = tables._pipe(
                reshape.unpivot_tables,
                headerId_pivot,
                heuristics=unpivot_heuristics,
                desc="Unpivoting",
            )

        if split_compound_columns:
            tables = tables._pipe(reshape.split_compound_columns, desc="Uncompounding")

        return tables

    def get_tables_index(tables: TableSet):
        """Make a dataframe of table and column indexes"""
        import pandas as pd
        from . import cluster

        tables = tables._offset("tableIndex", "tableIndex", default=1)
        tables = tables._offset("numCols", "columnIndexOffset")
        tables.persist()

        index = pd.concat(tables._pipe(cluster.make_column_index_df))
        return tables, index

    def cluster(
        tables: TableSet,
        workdir: Path = None,
        addcontext: typing.List[str] = (),
        headerunions: bool = True,
        matcher_configs: typing.List[Config] = (),
        use_match_cache: bool = False,
        agg_func: str = "mean",
        agg_threshold: float = 0,
        edge_exp: float = 1,
        agg_threshold_col: float = None,
        store_align_meta: typing.List[str] = ["tableHeaders"],
        mergeheaders_topn: int = None,
        assets: typing.List[Config] = (),
    ):
        """Cluster tables

        See also: :mod:`takco.cluster`

        Args:
            tables: Tables
            workdir: Working Directory

            addcontext: Add these context types
            headerunions: make header unions
            matcher_configs: Use only these matchers
            agg_func: Matcher aggregation function
            agg_threshold: Matcher aggregation threshold
            edge_exp : Exponent of edge weight for Louvain modularity
            agg_threshold_col: Matcher aggregation threshold (default: agg_threshold)
            mergeheaders_topn: Number of top headers to keep when merging
        """
        from .cluster import matchers as matcher_classes

        matchers = [
            Config({"fdir": workdir, **m}, assets).init_class(
                **matcher_classes.__dict__
            )
            for m in matcher_configs
        ]
        agg_threshold_col = agg_threshold_col or agg_threshold

        import tqdm
        import sqlite3
        import pandas as pd
        from . import cluster

        if addcontext:
            tables = tables._pipe(cluster.tables_add_context_rows, fields=addcontext)

        if headerunions:
            tables = tables._fold(
                cluster.table_get_headerId, cluster.combine_by_first_header
            )

        if matchers:

            # Collect index
            tables, index = TableSet.get_tables_index(tables)

            Path(workdir or ".").mkdir(exist_ok=True, parents=True)
            fpath = Path(workdir) / Path("indices.sqlite")
            if fpath.exists():
                fpath.unlink()
            log.info(f"Writing {len(index)} index rows to {fpath}")
            with sqlite3.connect(fpath) as con:
                index.to_sql("indices", con, index_label="i", if_exists="replace")
                con.execute("create index colOffset on indices(columnIndexOffset)")

            table_indices = set(t["tableIndex"] for t in tables)
            simpath = Path(workdir) / Path("sims.csv")
            if use_match_cache and simpath.exists():
                sims = pd.read_csv(simpath, index_col=[0, 1, 2, 3])
            else:
                log.info(f"Using matchers: {', '.join(m.name for m in matchers)}")
                matchers = tables._pipe(cluster.matcher_add_tables, matchers)

                matchers = matchers._fold(lambda x: x.name, lambda a, b: a.merge(b))
                matchers = list(matchers)
                for m in matchers:
                    log.info(f"Indexing {m}")
                    m.index()

                # Get blocked column match scores
                log.info(f"Computing column similarities...")
                simdfs = tables.__class__(table_indices)._pipe(
                    cluster.make_blocked_matches_df, matchers
                )
                sims = pd.concat(list(simdfs))
                log.info(f"Computed {len(sims)} column similarities")

                sims.to_csv(simpath)

            log.info(
                f"Aggregating matcher results using `{agg_func} > {agg_threshold}` "
            )
            aggsim = cluster.aggregate_similarities(sims, agg_func)

            # Compute soft column alignment jaccard
            import warnings

            con = sqlite3.connect(Path(workdir) / Path("indices.sqlite"))
            n = pd.read_sql("select i,numCols from indices", con).set_index("i")[
                "numCols"
            ]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tqdm.tqdm.pandas(desc="Aggregating column scores")
            threshold_sim = aggsim[aggsim > agg_threshold]
            aligned_total = threshold_sim.groupby(level=[0, 1]).progress_aggregate(
                cluster.max_align, return_total=True
            )
            j = (
                pd.DataFrame({"total": aligned_total})
                .join(n.rename("n1"), on="ti1")
                .join(n.rename("n2"), on="ti2")
            )
            tablesim = j["total"] / (j["n1"] + j["n2"] - j["total"])

            tablesim.to_csv(Path(workdir) / Path("tablesim.csv"))

            # TODO: end of parallel loop

            tablesim[tablesim < 0] = 0
            for ti in table_indices:
                tablesim.loc[(ti, ti)] = 1  # assure all tables are clustered
            louvain_partition = cluster.louvain(tablesim, edge_exp=edge_exp)
            log.info(f"Found {len(louvain_partition)} clusters")

            ## Cluster columns
            from sklearn.cluster import AgglomerativeClustering

            clus = AgglomerativeClustering(
                affinity="precomputed",
                linkage="complete",
                n_clusters=None,
                distance_threshold=1,
            )

            # TODO: parallelize this with partition object
            iparts = list(enumerate(louvain_partition))
            ti_pi, pi_ncols, ci_pci = {}, {}, {}
            chunks = cluster.cluster_partition_columns(
                iparts, clus, aggsim, agg_func, agg_threshold_col, matchers,
            )
            for chunk_ti_pi, chunk_pi_ncols, chunk_ci_pci in chunks:
                ti_pi.update(chunk_ti_pi)
                pi_ncols.update(chunk_pi_ncols)
                ci_pci.update(chunk_ci_pci)

            # TODO: serialize partitions & cluster alignments for UI

            # TODO: parallelize map_partition
            listtables = list(tables)
            for table in listtables:
                table["part"] = ti_pi[table["tableIndex"]]
                #         assert all( len(row)==table['numCols'] for row in table['tableData'] ) # fails
                ci_range = range(
                    table["columnIndexOffset"],
                    table["columnIndexOffset"] + table["numCols"],
                )

                # TODO: add similarity scores
                pci_c = {ci_pci[ci]: c for c, ci in enumerate(ci_range) if ci in ci_pci}
                # partColAlign is a mapping from partition cols to local cols
                table["partColAlign"] = {
                    pci: pci_c.get(pci, None) for pci in range(pi_ncols[table["part"]])
                }

            # TODO: parallelize foldby
            pi_mergetable = {}
            for table in listtables:
                pi = table["part"]
                pi_mergetable[pi] = (
                    cluster.merge_partition_tables(
                        pi_mergetable[pi],
                        table,
                        store_align_meta=store_align_meta,
                        mergeheaders_topn=mergeheaders_topn,
                    )
                    if (pi in pi_mergetable)
                    else table
                )

            tables = tables.__class__(list(pi_mergetable.values()))

        return tables

    def coltypes(
        tables: TableSet,
        typer_config: Config = {"class": "SimpleTyper"},
        assets: typing.List[Config] = (),
    ):
        """Find column types.
        
        Args:
            tables: Tables to find column types
            typer_config: Typer config
        """

        from . import link

        typer = Config(typer_config, assets).init_class(**link.__dict__)
        return tables._pipe(link.coltypes, typer=typer)

    def integrate(
        tables: TableSet,
        db_config: Config = {
            "class": "RDFSearcher",
            "statementURIprefix": "http://www.wikidata.org/entity/statement/",
            "store": {"class": "SparqlStore"},
        },
        typer_config: Config = {"class": "SimpleTyper"},
        pfd_threshold: float = 0.9,
        assets: typing.List[Config] = (),
    ):
        """Integrate tables with KB properties and classes.

        See also: :meth:`takco.link.integrate`

        Args:
            tables: Tables to integrate
            db_config: Nary DB config
            typer_config: Typer config
            pfd_threshold: Probabilistic Functional Dependency threshold for key column
                prediction
        """
        from . import link

        db = Config(db_config, assets).init_class(**link.__dict__)
        log.info(f"Integrating with {db}")

        typer = None
        if typer_config:
            typer = Config(typer_config, assets).init_class(**link.__dict__)
            log.info(f"Typing with {typer}")

        return tables._pipe(
            link.integrate, db=db, typer=typer, pfd_threshold=pfd_threshold,
        )

    def link(
        tables: TableSet,
        lookup_config: Config = {"class": "MediaWikiAPI"},
        lookup_cells: bool = False,
        linker_config: Config = {
            "class": "First",
            "searcher": {"class": "MediaWikiAPI"},
        },
        usecols: str = [],
        assets: typing.List[Config] = (),
    ):
        """Link table entities to KB.
        
        Depending on the Linker, also integrates table with KB classes and properties.
        
        See also: :meth:`takco.link.link`

        Args:
            tables: Tables to link
            linker: Entity Linker config
            usecols: Columns to use
        """
        from . import link

        if lookup_config:
            lookup = Config(lookup_config, assets).init_class(**link.__dict__)
            log.debug(f"Lookup with {lookup}")
            tables = tables._pipe(
                link.lookup_hyperlinks, lookup=lookup, lookup_cells=lookup_cells,
            )

        if linker_config:
            linker = Config(linker_config, assets).init_class(**link.__dict__)
            log.info(f"Linking with {linker}")
            tables = tables._pipe(link.link, linker, usecols=usecols,)

        return tables

    def score(
        tables: TableSet,
        labels: Config,
        datadir: Path = None,
        resourcedir: Path = None,
        report: typing.Dict = None,
        keycol_only: bool = False,
        assets: typing.List[Config] = (),
    ):
        """Calculate evaluation scores

        See also: :mod:`takco.evaluate.score`

        Args:
            tables: Table with predictions to score
            labels: Annotated data config
            datadir: Data directory
            resourcedir: Resource directory
            report: Report config
            keycol_only: Only report results for key column
        """
        from . import evaluate

        annot = Config(labels, assets)
        dset = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **annot)
        table_annot = dset.get_annotated_tables()
        log.info(f"Loaded {len(table_annot)} annotated tables")

        return tables._pipe(evaluate.table_score, table_annot, keycol_only=keycol_only)

    def novelty(
        tables: TableSet,
        searcher_config: Config = None,
        assets: typing.List[Config] = (),
    ):
        from . import evaluate
        from . import link

        searcher = Config(searcher_config, assets).init_class(**link.__dict__)
        return tables._pipe(evaluate.table_novelty, searcher)

    def triples(tables: TableSet, include_type: bool = True):
        """Make triples for predictions"""
        from . import evaluate

        return tables._pipe(evaluate.table_triples, include_type=include_type)

    def report(tables: TableSet, keycol_only: bool = False, curve: bool = False):
        from . import evaluate

        data = {}

        all_gold = {}
        all_pred = {}
        kb_kind_novelty_hashes = {}
        for table in tables:
            key = [table.get("_id")]
            for task, flatten in evaluate.task_flatten.items():
                gold = table.get("gold", {}).get(task, {})
                if gold:

                    pred = table.get(task, {})
                    if keycol_only and pred.get(str(table.get("keycol"))):
                        keycol = str(table.get("keycol"))
                        pred = {keycol: pred.get(keycol)}

                    golds = dict(flatten(gold, key=key))
                    preds = dict(flatten(pred, key=key))
                    all_gold.setdefault(task, {}).update(golds)
                    all_pred.setdefault(task, {}).update(preds)

            if "novelty" in table:
                for kbname, novelty in table["novelty"].items():
                    kind_novelty_hashes = kb_kind_novelty_hashes.setdefault(kbname, {})
                    # aggregate all hashes
                    for kind, novelty_hashes in novelty.get("hashes", {}).items():
                        nhs = kind_novelty_hashes.setdefault(kind, {})
                        for novelty, hashes in novelty_hashes.items():
                            hs = nhs.setdefault(novelty, set())
                            hs |= set(hashes)

        scores = {}
        curves = {}
        for task in evaluate.task_flatten:
            gold, pred = all_gold.get(task, {}), all_pred.get(task, {})
            if pred:
                log.info(
                    f"Collected {len(gold)} gold and {len(pred)} pred for task {task}"
                )

                task_scores = evaluate.score.classification(gold, pred)
                task_scores["predictions"] = len(pred)
                scores[task] = task_scores
                if curve:
                    curves[task] = evaluate.score.pr_curve(gold, pred)
        if scores:
            data["scores"] = scores
        if curves:
            data["curves"] = curves

        if kb_kind_novelty_hashes:
            data["novelty"] = {}
            for kb, kind_novelty_hashes in kb_kind_novelty_hashes.items():
                data["novelty"][kb] = evaluate.novelty.count_noveltyhashes(
                    kind_novelty_hashes
                )

        return data

    @classmethod
    def run(
        cls,
        pipeline: Config,
        input_tables: typing.List[typing.Union[Path, Config]] = (),
        workdir: Path = None,
        datadir: Path = None,
        resourcedir: Path = None,
        force: typing.List[int] = (),
        cache: bool = False,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):
        """Run entire pipeline

        Args:
            pipeline: Pipeline config
            input_tables: Either path(s) to tables as json/csv or config with ``input``
            workdir: Working directory (also for cache)
            datadir: Data directory
            resourcedir: Resource directory
            assets: Asset definitions
            force: Force execution of steps if cache files are already present
            step_force: Force execution of steps starting at this number
            cache: Cache intermediate results
            executor: Executor engine
        """
        import shutil
        from inspect import signature

        pipeline = Config(pipeline)
        if "name" in pipeline:
            name = pipeline["name"]
        else:
            import datetime

            name = "takco-run-" + str(datetime.datetime.now().isoformat())

        pipeline_assets = [Config(a, assets) for a in pipeline.get("assets", [])]
        conf = {
            "workdir": workdir or pipeline.get("workdir") or ".",
            "datadir": datadir or pipeline.get("datadir"),
            "resourcedir": resourcedir or pipeline.get("resourcedir"),
            "assets": list(assets or []) + pipeline_assets,
            "cache": cache or pipeline.get("cache"),
            "executor": executor or pipeline.get("executor"),
        }
        executor, assets, cache = conf["executor"], conf["assets"], conf["cache"]
        executor, exkw = get_executor_kwargs(executor, assets)

        force = True if (force == []) else force

        if cache:
            if force:
                shutil.rmtree(conf["workdir"], ignore_errors=True)
            Path(conf["workdir"]).mkdir(exist_ok=True, parents=True)

        def wrap_step(stepfunc, stepargs, stepdir):
            if cache:
                shutil.rmtree(stepdir, ignore_errors=True)
                stepdir.mkdir(exist_ok=True, parents=True)
                log.info(f"Writing cache to {stepdir}")
                tablefile = str(stepdir) + "/*.jsonl"
                return stepfunc(**stepargs)._dump(tablefile)
            else:
                return stepfunc(**stepargs)

        # Prepend input tables to pipeline
        input_tables = input_tables or pipeline.get("input_tables")
        if not input_tables:
            raise Exception(f"No input tables specified in config or pipeline!")
        log.info(f"Using input tables {input_tables}")
        if isinstance(input_tables, Config):
            input_tables["step"] = input_tables.pop("input")
        else:
            input_tables = Config({"step": "load", "path": input_tables})

        pipeline.setdefault("step", []).insert(0, input_tables)

        log.info(f"Running pipeline '{name}' in {workdir}")
        tables = []
        for si, stepargs in enumerate(pipeline.get("step", [])):
            if "step" in stepargs:
                stepfuncname = stepargs.pop("step")
                stepname = f"{si}-{stepargs.get('name', stepfuncname)}"
                stepdir = Path(conf["workdir"]) / Path(stepname)

                nodir = (not stepdir.exists()) or (not any(stepdir.iterdir()))
                if (force == True) or (si >= min(force)) or nodir:
                    stepfunc = getattr(TableSet, stepfuncname)
                    if not stepfunc:
                        raise Exception(
                            f"Pipeline step '{stepfuncname}' does not exist"
                        )

                    sig = signature(stepfunc)
                    local_config = dict(tables=tables, **conf)
                    local_config["workdir"] = stepdir
                    for k, v in local_config.items():
                        if (k in sig.parameters) and (k not in stepargs):
                            stepargs[k] = v

                    log.info(f"Chaining pipeline step {stepname}")
                    tables = wrap_step(stepfunc, stepargs, stepdir)
                else:
                    log.warn(f"Skipping step {stepname}, using cache instead")
                    tablefile = str(stepdir) + "/*.jsonl"
                    tables = executor._load(tablefile, **exkw)
            else:
                raise Exception(f"Pipeline step {si} has no step type")

        if cache:
            for _ in tables:
                pass
        else:
            return tables
