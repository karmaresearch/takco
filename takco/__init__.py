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
from . import pages, reshape, link, cluster, evaluate

__all__ = [
    "TableSet",
]


class TableSet:
    """A set of tables that can be clustered and linked."""

    def __init__(self, tables):
        if isinstance(tables, TableSet):
            self.tables = tables.tables
        else:
            if not isinstance(tables, HashBag):
                tables = HashBag(tables)
            self.tables = tables

    def __iter__(self):
        return self.tables.__iter__()

    def dump(self, *args, **kwargs):
        return TableSet(self.tables.dump(*args, **kwargs))

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
        return TableSet(executor(data, **exkw))

    @classmethod
    def load(
        cls,
        path: Path = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
        **_,
    ):
        log.debug(f"Loading tables {path} using executor {executor}")
        if all(str(val).endswith("csv") for val in path):
            return cls.csvs(path, executor=executor, assets=assets)

        executor, exkw = get_executor_kwargs(executor, assets)
        return TableSet(executor.load(path, **exkw))

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

        params = Config(params, assets)
        log.info(f"Loading dataset {params}")
        ds = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **params)
        it_sample = itertools.islice(ds.get_unannotated_tables(), sample)
        return TableSet(executor(it_sample, **exkw))

    @classmethod
    def extract(
        cls,
        source: typing.Union[Config, HashBag] = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):
        """Collect tables from HTML files

        See also: :mod:`takco.extract`

        Args:
            source: Source of HTML files
        """
        if isinstance(source, Config):
            source = Config.create(source, assets, pages.__dict__)
            htmlpages = source.get(executor, assets)
        elif isinstance(source, HashBag):
            htmlpages = source
        else:
            htmlpages = HashBag(source)

        from .extract import extract_tables

        return TableSet(htmlpages.pipe(extract_tables))

    def reshape(
        self: TableSet,
        restructure: bool = True,
        prefix_header_rules: typing.List[Config] = (),
        unpivot_configs: typing.List[Config] = (),
        workdir: Path = None,
        centralize_pivots: bool = False,
        compound_splitter_config: Config = None,
        discard_headerless_tables: bool = False,
        assets: typing.List[Config] = (),
    ):
        """Reshape tables

        See also: :mod:`takco.reshape`

        Args:
            restructure: Whether to restructure tables heuristically
            unpivot_configs: Use a subset of available heuristics
            centralize_pivots: If True, find pivots on unique headers instead of tables
            compound_splitter_config: Splitter for compound columns

        """
        tables = TableSet(self).tables

        if restructure:
            log.info(f"Restructuring with rules: {prefix_header_rules}")
            tables = tables.pipe(reshape.restructure, prefix_header_rules)

        classes = {**reshape.__dict__, **link.__dict__}
        unpivot_heuristics = {
            getattr(h, "name", h.__class__.__name__): h
            for h in Config.create(unpivot_configs, assets, classes)
        }

        if unpivot_heuristics:
            log.info(f"Unpivoting with heuristics: {', '.join(unpivot_heuristics)}")

            tables = tables.persist()

            for name, h in tables.pipe(reshape.build_heuristics, heuristics=unpivot_heuristics):
                unpivot_heuristics[name].merge(h)

            headerId_pivot = None
            if centralize_pivots:
                headers = tables.fold(reshape.table_get_headerId, reshape.get_header)

                pivots = headers.pipe(
                    reshape.yield_pivots, heuristics=unpivot_heuristics
                )

                headerId_pivot = {p["headerId"]: p for p in pivots}
                log.info(f"Found {len(headerId_pivot)} pivots")

            tables = tables.pipe(
                reshape.unpivot_tables, headerId_pivot, heuristics=unpivot_heuristics,
            )

        if compound_splitter_config is not None:
            from .reshape import compound

            compound_splitter = Config.create(
                compound_splitter_config, assets, compound.__dict__
            )
            tables = tables.pipe(reshape.split_compound_columns, compound_splitter)

        if discard_headerless_tables:

            def filter_headerless(ts):
                for t in ts:
                    headers = t.get("tableHeaders", [])
                    if any(h.get("text") for hrow in headers for h in hrow):
                        yield t

            tables = tables.pipe(filter_headerless)

        return TableSet(tables)

    @staticmethod
    def number_table_columns(tables):
        """Create global table and column IDs"""

        log.info(f"Numbering tables...")
        tables = tables.offset("tableIndex", "tableIndex", default=1)
        tables = tables.offset("numCols", "columnIndexOffset")
        return tables

    @staticmethod
    def build_matchers(matchers, tables):
        matchers = list(
            tables.pipe(cluster.matcher_add_tables, matchers).fold(
                lambda x: x.name, lambda a, b: a.merge(b)
            )
        )
        for m in matchers:
            log.info(f"Indexing {m.name}")
            m.index()
        return matchers

    def cluster(
        self: TableSet,
        workdir: Path = None,
        addcontext: typing.List[str] = (),
        headerunions: bool = True,
        matcher_configs: typing.List[Config] = (),
        filter_configs: typing.List[Config] = (),
        agg_func: str = "mean",
        agg_threshold: float = 0,
        align_columns: str = "greedy",
        align_width_norm: str = "jacc",
        align_use_total_width: bool = True,
        edge_exp: float = 1,
        agg_threshold_col: float = None,
        keep_partition_meta: typing.List[str] = ["tableHeaders"],
        mergeheaders_topn: int = None,
        assets: typing.List[Config] = (),
    ):
        """Cluster tables

        See also: :mod:`takco.cluster`

        Args:
            tables: Tables
            workdir: Working Directory

            addcontext: Add these context types
            headerunions: Make header unions
            matcher_configs: Matcher configs
            filter_configs: Filter configs
            agg_func: Aggregation function for :meth:`takco.cluster.aggregate_match_sims`
            agg_threshold: Threshold value
            align_columns: Column alignment method  ({'max1', 'max2', 'greedy'}).
            align_width_norm: Table width difference normalisation method ({'wide', 'narrow', 'jacc'}).
            align_use_total_width: Whether to use total table width. Defaults to True. 
            edge_exp : Exponent of edge weight for Louvain modularity
            agg_threshold_col: Matcher aggregation threshold (default: agg_threshold)
            keep_partition_meta: Attributes to keep for partition table analysis
            mergeheaders_topn: Number of top headers to keep when merging
        """
        tables = TableSet(self).tables
        from .cluster import matchers as matcher_classes

        matchers = Config.create(
            matcher_configs, assets, matcher_classes.__dict__, fdir=workdir
        )
        filters = Config.create(
            filter_configs, assets, matcher_classes.__dict__, fdir=workdir
        )
        agg_threshold_col = agg_threshold_col or agg_threshold

        import tqdm
        import sqlite3
        import pandas as pd

        if addcontext:
            tables = tables.pipe(cluster.tables_add_context_rows, fields=addcontext)

        if headerunions:
            tables = tables.fold(
                cluster.table_get_headerId, cluster.combine_by_first_header
            )

        if matchers:

            ## Partition table similarity graph
            tables = TableSet.number_table_columns(tables).persist()
            log.info(f"Building matchers: {', '.join(m.name for m in matchers)}")
            matchers = TableSet.build_matchers(matchers, tables)
            filters = TableSet.build_matchers(filters, tables)

            # Get blocked column match scores
            tableid_colids = dict(tables.pipe(cluster.get_table_ids))
            log.info(f"Blocking tables; computing and aggregating column sims...")

            tablesim = pd.concat(
                tables.pipe(
                    cluster.get_tablesims,
                    tableid_colids=tableid_colids,
                    matchers=matchers,
                    filters=filters,
                    agg_func=agg_func,
                    agg_threshold=agg_threshold,
                    align_columns=align_columns,
                    align_width_norm=align_width_norm,
                    align_use_total_width=align_use_total_width,
                )
            )
            # assure all tables are clustered using identity matrix
            itups = ((ti, ti) for ti in tableid_colids)
            ii = pd.MultiIndex.from_tuples(itups, names=["ti1", "ti2"])
            tablesim = pd.concat([tablesim, pd.Series(1, index=ii)])
            reduction = (len(tableid_colids)**2) / len(tablesim)
            log.info(f"Got {len(tablesim)} table similarities; {reduction:.0f}x reduction")
            # tablesim.to_csv(Path(workdir) / Path("tablesim.csv"))

            louvain_partition = cluster.louvain(tablesim, edge_exp=edge_exp)
            nonsingle = [p for p in louvain_partition if len(p) > 1]
            log.info(f"Found {len(nonsingle)}/{len(louvain_partition)} >1 partitions")

            ## Cluster columns
            log.info(f"Clustering columns...")
            chunks = tables.__class__(enumerate(nonsingle)).pipe(
                cluster.cluster_partition_columns,
                tableid_colids=tableid_colids,
                matchers=matchers,
                agg_func=agg_func,
                agg_threshold_col=agg_threshold_col,
            )
            from collections import ChainMap

            ti_pi, pi_ncols, ci_pci, ti_colsim = (
                {k: v for d in ds for k, v in d.items()} for ds in zip(*chunks)
            )

            if workdir:
                colsimdir = Path(workdir) / Path("colsims")
                colsimdir.mkdir(exist_ok=True, parents=True)
                for ti, colsim in ti_colsim.items():
                    colsim.to_csv(colsimdir / Path(f"{ti}.csv"), header=True)

            log.info(f"Merging clustered tables...")
            tables = tables.pipe(
                cluster.set_partition_columns, ti_pi, pi_ncols, ci_pci
            ).fold(
                lambda t: t["_id"],
                lambda a, b: cluster.merge_partition_tables(
                    a,
                    b,
                    keep_partition_meta=keep_partition_meta,
                    mergeheaders_topn=mergeheaders_topn,
                ),
            )

        return TableSet(tables)

    def coltypes(
        self: TableSet,
        typer_config: Config = {"class": "SimpleTyper"},
        assets: typing.List[Config] = (),
    ):
        """Find column types.
        
        Args:
            tables: Tables to find column types
            typer_config: Typer config
        """
        tables = TableSet(self).tables
        typer = Config.create(typer_config, assets, link.__dict__)
        return TableSet(tables.pipe(link.coltypes, typer=typer))

    def integrate(
        self: TableSet,
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
        tables = TableSet(self).tables

        db = Config.create(db_config, assets, link.__dict__)
        log.info(f"Integrating with {db}")

        typer = None
        if typer_config:
            typer = Config.create(typer_config, assets, link.__dict__)
            log.info(f"Typing with {typer}")

        tables = tables.pipe(
            link.integrate, db=db, typer=typer, pfd_threshold=pfd_threshold,
        )
        return TableSet(tables)

    def link(
        self: TableSet,
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
        tables = TableSet(self).tables

        if lookup_config:
            lookup = Config.create(lookup_config, assets, link.__dict__)
            log.debug(f"Lookup with {lookup}")
            tables = tables.pipe(
                link.lookup_hyperlinks, lookup=lookup, lookup_cells=lookup_cells,
            )

        if linker_config:
            linker = Config.create(linker_config, assets, link.__dict__)
            log.info(f"Linking with {linker}")
            tables = tables.pipe(link.link, linker, usecols=usecols,)

        return TableSet(tables)

    def score(
        self: TableSet,
        labels: Config,
        datadir: Path = None,
        resourcedir: Path = None,
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
            keycol_only: Only calculate results for key column
        """
        tables = TableSet(self).tables

        annot = Config(labels, assets)
        dset = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **annot)
        table_annot = dset.get_annotated_tables()
        log.info(f"Loaded {len(table_annot)} annotated tables")

        pairs = tables.__class__([(t, table_annot.get(t["_id"], {})) for t in tables])
        tables = pairs.pipe(evaluate.table_score, keycol_only=keycol_only)
        return TableSet(tables)

    def novelty(
        self: TableSet,
        searcher_config: Config = None,
        assets: typing.List[Config] = (),
    ):
        tables = TableSet(self).tables

        searcher = Config.create(searcher_config, assets, link.__dict__)
        return TableSet(tables.pipe(evaluate.table_novelty, searcher))

    def triples(self: TableSet, include_type: bool = True):
        """Make triples for predictions"""
        tables = TableSet(self).tables

        tables = tables.pipe(evaluate.table_triples, include_type=include_type)
        return TableSet(tables)

    def report(self: TableSet, keycol_only: bool = False, curve: bool = False):
        """Generate report

        Args:
            keycol_only: Only analyse keycol predictions
            curve: Calculate precision-recall tradeoff curve
        """
        tables = TableSet(self).tables

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

        return tables.__class__([data])

    @classmethod
    def run(
        cls,
        pipeline: Config,
        input_tables: typing.List[typing.Union[Config, Path]] = (),
        workdir: Path = None,
        datadir: Path = None,
        resourcedir: Path = None,
        forcesteps: typing.List[int] = (),
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
            forcesteps: Force execution of steps if cache files are already present
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

        stepforce = None if (forcesteps == ()) else min(forcesteps, default=0)
        force = forcesteps == []

        def wrap_step(stepfunc, stepargs, stepdir):
            if cache:
                shutil.rmtree(stepdir, ignore_errors=True)
                stepdir.mkdir(exist_ok=True, parents=True)
                log.info(f"Writing cache to {stepdir}")
                tablefile = str(stepdir) + "/*.jsonl"
                return stepfunc(**stepargs).dump(tablefile)
            else:
                return stepfunc(**stepargs)

        def chain_step(tableset, workdir, si, stepargs):
            stepargs = dict(stepargs)
            if "step" in stepargs:
                stepfuncname = stepargs.pop("step")
                stepname = f"{si}-{stepargs.get('name', stepfuncname)}"
                stepdir = workdir / Path(stepname)

                nodir = (not stepdir.exists()) or (not any(stepdir.iterdir()))
                if force or (stepforce is not None and si >= stepforce) or nodir:
                    stepfunc = getattr(TableSet, stepfuncname)
                    if not stepfunc:
                        raise Exception(
                            f"Pipeline step '{stepfuncname}' does not exist"
                        )

                    sig = signature(stepfunc)
                    local_config = {"self": tableset, **conf, "workdir": stepdir}
                    for k, v in local_config.items():
                        if (k in sig.parameters) and (k not in stepargs):
                            stepargs[k] = v

                    log.info(f"Chaining pipeline step {stepname}")
                    yield workdir, wrap_step(stepfunc, stepargs, stepdir)
                else:
                    log.warn(f"Skipping step {stepname}, using cache instead")
                    tablefile = str(stepdir) + "/*.jsonl"
                    yield workdir, TableSet(executor.load(tablefile, **exkw))
            elif "split" in stepargs:
                # Persist pre-split tables
                tableset.tables.persist()
                for split, splitargs in enumerate(stepargs["split"]):
                    splitname = f"{si}-split-{split}"
                    splitdir = workdir / Path(splitname)
                    yield from chain_step(tableset, splitdir, si, splitargs)
            else:
                raise Exception(f"Pipeline step {si} has no step type")

        
        # Prepend input tables to pipeline
        input_tables = input_tables or pipeline.get("input_tables")
        if isinstance(input_tables, HashBag):
            streams = [(Path(conf["workdir"]), input_tables)]
        else:
            if not input_tables:
                raise Exception(f"No input tables specified in config or pipeline!")
            if not isinstance(input_tables, list):
                input_tables = [input_tables]
            if any(isinstance(inp, Config) for inp in input_tables):
                input_tables = input_tables[0]
                input_tables["step"] = input_tables.pop("input")
            else:
                input_tables = Config({"step": "load", "path": input_tables})
            log.info(f"Using input tables {input_tables}")
            pipeline.setdefault("step", []).insert(0, input_tables)
            streams = [(Path(conf["workdir"]), [])]

        if cache:
            if force:
                shutil.rmtree(conf["workdir"], ignore_errors=True)
            Path(conf["workdir"]).mkdir(exist_ok=True, parents=True)

        log.info(f"Running pipeline '{name}' in {workdir} using {executor}")

        for si, args in enumerate(pipeline.get("step", [])):
            streams = [c for wd, ts in streams for c in chain_step(ts, wd, si, args)]

        if cache:
            for workdir, tableset in streams:
                for _ in tableset:
                    pass
        else:
            _, tablesets = zip(*streams)
            return tablesets[0].tables.__class__.concat([t.tables for t in tablesets])
