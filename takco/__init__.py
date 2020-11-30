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
from . import config
from .storage import Storage, HDFSPath

from . import pages, reshape, link, cluster, evaluate


class Pipeline(list):
    pass


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
        cls, *path: typing.Union[HDFSPath, Path], executor: HashBag = HashBag(),
    ):

        log.info(f"Reading csv tables")
        import csv

        # TODO: HDFSPath
        data = (
            {
                "tableData": [
                    [{"text": c} for c in row] for row in csv.reader(Path(fname).open())
                ],
                "_id": fname,
            }
            for fname in path
        )
        return TableSet(executor.new(data))

    @classmethod
    def load(
        cls, *path: typing.Union[HDFSPath, Path], executor: HashBag = HashBag(), **_,
    ):
        log.debug(f"Loading tables {path} using executor {executor}")
        if path and all(str(val).endswith("csv") for val in path):
            return cls.csvs(*path, executor=executor)

        return TableSet(executor.load(*path))

    @classmethod
    def dataset(
        cls,
        dataset: evaluate.dataset.Dataset,
        datadir: Path = None,
        resourcedir: Path = None,
        take: int = None,
        executor: HashBag = HashBag(),
    ):
        """Load tables from a dataset

        See also: :mod:`takco.evaluate.dataset`

        Args:
            dataset: Dataset
            datadir: Data directory
            resourcedir: Resource directory
            take: Use only first N tables
        """

        tables = dataset.get_unannotated_tables()
        if not isinstance(tables, HashBag):
            tables = executor.new(tables)
        if take:
            tables = tables.take(take)
        return TableSet(tables)

    @classmethod
    def extract(
        cls,
        source: typing.Union[pages.PageSource, HashBag] = None,
        executor: HashBag = HashBag(),
    ):
        """Collect tables from HTML files

        See also: :mod:`takco.extract`

        Args:
            source: Source of HTML files
        """
        if isinstance(source, HashBag):
            htmlpages = source
        elif isinstance(source, pages.PageSource):
            htmlpages = source.get(executor)
        else:
            htmlpages = HashBag(source)

        from .extract import extract_tables

        return TableSet(htmlpages.pipe(extract_tables))

    def reshape(
        self: TableSet,
        restructure: bool = True,
        prefix_header_rules: typing.List[typing.Dict] = [],
        unpivot_heuristics: typing.List[reshape.PivotFinder] = [],
        centralize_pivots: bool = False,
        compound_splitter: reshape.CompoundSplitter = None,
        discard_headerless_tables: bool = False,
    ):
        """Reshape tables

        See also: :mod:`takco.reshape`

        Args:
            restructure: Whether to restructure tables heuristically
            unpivot_heuristics: Specifications of pivot finders
            centralize_pivots: If True, find pivots on unique headers instead of tables
            compound_splitter: Splitter for compound columns

        """
        tables = TableSet(self).tables

        if restructure:
            log.info(f"Restructuring with rules: {prefix_header_rules}")
            tables = tables.pipe(reshape.restructure, prefix_header_rules)

        if unpivot_heuristics is not None:
            unpivoters = {
                getattr(h, "name", h.__class__.__name__): h for h in unpivot_heuristics
            }
            log.info(f"Unpivoting with heuristics: {', '.join(unpivoters)}")

            tables = tables.persist()

            log.debug(f"Building heuristics...")
            for name, h in tables.pipe(reshape.build_heuristics, heuristics=unpivoters):
                unpivoters[name].merge(h)

            headerId_pivot = None
            if centralize_pivots:
                log.debug(f"Finding unpivots centrally...")
                headers = tables.fold(reshape.table_get_headerId, reshape.get_header)

                pivots = headers.pipe(reshape.yield_pivots, heuristics=unpivoters)

                headerId_pivot = {p["headerId"]: p for p in pivots}
                log.info(f"Found {len(headerId_pivot)} pivots")

            log.debug(f"Unpivoting...")
            tables = tables.pipe(
                reshape.unpivot_tables, headerId_pivot, heuristics=unpivoters,
            )

        if compound_splitter is not None:
            tables = tables.pipe(reshape.split_compound_columns, compound_splitter)

        if discard_headerless_tables:

            def filter_headerless(ts):
                for t in ts:
                    headers = t.get("tableHeaders", [])
                    if any(h.get("text") for hrow in headers for h in hrow):
                        yield t

            tables = tables.pipe(filter_headerless)

        return TableSet(tables)

    def filter(self: TableSet, filters: typing.List[str]):
        tables = TableSet(self).tables

        filters = [eval("lambda table: " + f) for f in filters]

        def filt(tables, filters):
            for table in tables:
                if not any(not c(table) for c in filters):
                    yield table

        return tables.pipe(filt, filters)

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
            tables.pipe(cluster.matcher_add_tables, matchers).fold_tree(
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
        addcontext: typing.List[str] = [],
        headerunions: bool = True,
        headerunions_attributes: typing.List[str] = [],
        matchers: typing.List[cluster.Matcher] = [],
        filters: typing.List[cluster.Matcher] = [],
        agg_func: str = "mean",
        agg_threshold: float = 0,
        align_columns: str = "greedy",
        align_width_norm: str = "jacc",
        align_use_total_width: bool = True,
        edge_exp: float = 1,
        agg_threshold_col: float = None,
        keep_partition_meta: typing.List[str] = ["tableHeaders"],
        mergeheaders_topn: int = None,
        max_cluster_size: int = None,
    ):
        """Cluster tables

        See also: :mod:`takco.cluster`

        Args:
            tables: Tables
            workdir: Working Directory

            addcontext: Add these context types
            headerunions: Make header unions
            headerunions_attributes: Extra attribute names for restricting header unions
            matchers: Matcher configs
            filters: Filter configs
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
        agg_threshold_col = agg_threshold_col or agg_threshold

        import tqdm
        import sqlite3
        import pandas as pd

        if addcontext:
            tables = tables.pipe(cluster.tables_add_context_rows, fields=addcontext)

        if headerunions:
            if headerunions_attributes:

                def key(t):
                    hid = cluster.table_get_headerId(t)
                    return str([hid] + [t.get(a) for a in headerunions_attributes])

            else:
                key = cluster.table_get_headerId
            tables = tables.fold(key, cluster.combine_by_first_header)

        if matchers:
            storage = None
            if workdir:
                log.info(f"Using workdir {workdir}")
                storage = Storage(workdir)

            ## Create or load matchers
            if storage and storage.exists_df("tablenumbers"):
                tablenums = storage.load_df("tablenumbers").to_dict(orient="index")

                def put_table_numbers(tables, tablenums):
                    for t in tables:
                        t.update(tablenums[t["_id"]])
                        yield t

                tables = tables.pipe(put_table_numbers, tablenums).persist()
            else:
                tables = TableSet.number_table_columns(tables).persist()

                if storage:

                    def get_table_numbers(tables):
                        for t in tables:
                            yield t["_id"], {
                                "tableIndex": t["tableIndex"],
                                "columnIndexOffset": t["columnIndexOffset"],
                            }

                    tablenums = dict(tables.pipe(get_table_numbers))
                    storage.save_df(
                        pd.DataFrame.from_dict(tablenums, orient="index"),
                        "tablenumbers",
                    )

            for m in matchers + filters:
                m.set_storage(workdir)

            if all(m.storage and m.indexed for m in matchers):
                log.info(f"Loading matchers: {', '.join(m.name for m in matchers)}")
            else:
                log.info(f"Building matchers: {', '.join(m.name for m in matchers)}")
                matchers = TableSet.build_matchers(matchers, tables)

            if all(m.storage and m.indexed for m in filters):
                log.info(f"Loading filters: {', '.join(m.name for m in filters)}")
            else:
                log.info(f"Building filters: {', '.join(m.name for m in filters)}")
                filters = TableSet.build_matchers(filters, tables)

            ## Partition table similarity graph
            log.info(f"Getting table IDs")
            tableid_colids = dict(tables.pipe(cluster.get_table_ids))

            if storage and storage.exists_df("tablesim"):
                tablesim = storage.load_df("tablesim")["tablesim"]
                tableids = pd.Index(list(tableid_colids))
                tablesim = tablesim.loc[tableids, tableids]
            else:
                # Get blocked column match scores
                ntables = len(tableid_colids)
                chunksize = cluster.get_table_chunk_size(ntables)
                log.info(f"Comparing {ntables} tables in chunks of size {chunksize}")

                # chunksize = round(10**4 / (len(tableid_colids) ** .5)) + 1
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
                reduction = (len(tableid_colids) ** 2) / len(tablesim)
                log.info(
                    f"Got {len(tablesim)} table similarities; {reduction:.0f}x reduction"
                )
                if storage:
                    storage.save_df(tablesim.to_frame("tablesim"), "tablesim")

            log.info(f"Clustering with {len(tablesim)} similarities")
            louvain_partition = cluster.louvain(tablesim, edge_exp=edge_exp)
            nonsingle = [p for p in louvain_partition if len(p) > 1]
            log.info(f"Found {len(nonsingle)}/{len(louvain_partition)} >1 partitions")
            largest = max(map(len, louvain_partition))
            log.info(f"Largest partition has {largest} tables")

            if (max_cluster_size is not None) and (largest > max_cluster_size):
                log.info(f"That's too big. Re-sizing to {max_cluster_size}")
                louvain_partition = [
                    part[i : i + max_cluster_size]
                    for part in louvain_partition
                    for i in range(0, len(part), max_cluster_size)
                ]
                nonsingle = [p for p in louvain_partition if len(p) > 1]
                log.info(
                    f"Made {len(nonsingle)}/{len(louvain_partition)} >1 partitions"
                )
                largest = max(map(len, louvain_partition))
                log.info(f"Largest partition has {largest} tables")

            ## Cluster columns
            log.info(f"Clustering columns...")
            chunks = tables.new(enumerate(nonsingle)).pipe(
                cluster.cluster_partition_columns,
                tableid_colids=tableid_colids,
                matchers=matchers,
                agg_func=agg_func,
                agg_threshold_col=agg_threshold_col,
            )
            ti_pi, pi_ncols, ci_pci, pi_colsim = (
                {k: v for d in ds for k, v in d.items()} for ds in zip(*chunks)
            )

            if workdir:
                for pi, colsim in pi_colsim.items():
                    Storage(workdir, "colsim").save_df(colsim, f"pi={pi}")

            log.info(f"Merging clustered tables...")
            tables = tables.pipe(
                cluster.set_partition_columns, ti_pi, pi_ncols, ci_pci
            ).fold(
                lambda t: t.get("_id", "untitled-0"),
                lambda a, b: cluster.merge_partition_tables(
                    a,
                    b,
                    keep_partition_meta=keep_partition_meta,
                    mergeheaders_topn=mergeheaders_topn,
                ),
            )

        return TableSet(tables)

    def coltypes(
        self: TableSet, typer: link.Typer = link.SimpleTyper(),
    ):
        """Find column types.
        
        Args:
            tables: Tables to find column types
            typer: Typer config
        """
        tables = TableSet(self).tables
        return TableSet(tables.pipe(link.coltypes, typer=typer))

    def integrate(
        self: TableSet, db: link.Database = None, pfd_threshold: float = 0.9,
    ):
        """Integrate tables with KB properties and classes.

        See also: :meth:`takco.link.integrate`

        Args:
            tables: Tables to integrate
            db: Nary DB config
            typer: Typer config
            pfd_threshold: Probabilistic Functional Dependency threshold for key column
                prediction
        """
        tables = TableSet(self).tables

        if db is None:
            db = link.RDFSearcher(
                statementURIprefix="http://www.wikidata.org/entity/statement/",
                store=link.SparqlStore(),
            )

        log.info(f"Integrating with {db}")
        tables = tables.pipe(link.integrate, db=db, pfd_threshold=pfd_threshold,)
        return TableSet(tables)

    def link(
        self: TableSet,
        lookup: link.Lookup = None,
        lookup_cells: bool = False,
        linker: link.Linker = None,
        usecols: typing.List[str] = [],
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
        usecols = usecols or []

        if lookup:
            log.debug(f"Lookup with {lookup}")
            tables = tables.pipe(
                link.lookup_hyperlinks, lookup=lookup, lookup_cells=lookup_cells,
            )

        if linker:
            log.info(f"Linking with {linker}")
            tables = tables.pipe(link.link, linker, usecols=usecols,)

        return TableSet(tables)

    def score(
        self: TableSet,
        annotations: evaluate.dataset.Dataset = None,
        datadir: Path = None,
        resourcedir: Path = None,
        keycol_only: bool = False,
        any_annotated: bool = False,
        only_annotated: bool = False,
    ):
        """Calculate evaluation scores

        See also: :mod:`takco.evaluate.score`

        Args:
            tables: Table with predictions to score
            annotations: Annotated dataset
            datadir: Data directory
            resourcedir: Resource directory
            keycol_only: Only calculate results for key column
        """
        tables = TableSet(self).tables

        if annotations:
            table_annot = annotations.get_annotated_tables()
            log.info(f"Loaded {len(table_annot)} annotated tables")

            def add_gold(tables, table_annot):
                tasks = ["entities", "classes", "properties"]
                for t in tables:
                    goldtable = table_annot.get(t["_id"], {})
                    t["gold"] = {t: goldtable.get(t, {}) for t in tasks}
                    yield t

            tables = tables.pipe(add_gold, table_annot)
        tables = tables.pipe(
            evaluate.table_score,
            keycol_only=keycol_only,
            any_annotated=any_annotated,
            only_annotated=only_annotated,
        )
        return TableSet(tables)

    def novelty(
        self: TableSet, searcher: link.Searcher,
    ):
        tables = TableSet(self).tables

        return TableSet(tables.pipe(evaluate.table_novelty, searcher))

    def triples(self: TableSet, include_type: bool = True):
        """Make triples for predictions"""
        tables = TableSet(self).tables

        tables = tables.pipe(evaluate.table_triples, include_type=include_type)
        return TableSet(tables)

    def report(
        self: TableSet,
        keycol_only: bool = False,
        curve: bool = False,
        any_annotated: bool = False,
        only_annotated: bool = False,
    ) -> HashBag:
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

                task_scores = evaluate.score.classification(
                    gold,
                    pred,
                    any_annotated=any_annotated,
                    only_annotated=only_annotated,
                )
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

        return TableSet([data])

    @classmethod
    def run(
        cls,
        pipeline: Pipeline,
        input_tables: typing.Union[pages.PageSource, HDFSPath, Path] = None,
        workdir: typing.Union[HDFSPath, Path] = None,
        datadir: Path = None,
        resourcedir: Path = None,
        forcesteps: typing.List[int] = [],
        cache: bool = False,
        executor: HashBag = HashBag(),
    ):
        """Run entire pipeline

        Args:
            pipeline: Pipeline config
            input_tables: Either path(s) to tables as json/csv or config with ``input``
            workdir: Working directory (also for cache)
            datadir: Data directory
            resourcedir: Resource directory
            forcesteps: Force execution of steps if cache files are already present
            cache: Cache intermediate results
            executor: Executor engine
        """
        import glob, os, shutil
        from inspect import signature

        stepforce = min(forcesteps or [], default=0) if forcesteps != None else None
        force = forcesteps == []

        conf = {
            "workdir": workdir,
            "datadir": datadir,
            "resourcedir": resourcedir,
            "cache": cache,
            "executor": executor,
        }

        def wrap_step(stepfunc, stepargs, stepdir):
            if cache:
                Storage(stepdir).mkdir()
                log.info(f"Writing cache to {stepdir}")
                tablefile = str(stepdir) + "/*.jsonl"
                for f in Storage(stepdir).ls():
                    if f.endswith(".jsonl"):
                        Storage(stepdir).rm(f)
                return stepfunc(**stepargs).dump(tablefile)
            else:
                return stepfunc(**stepargs)

        def chain_step(tableset, steppath, si, stepargs):
            stepargs = dict(stepargs)
            if "step" in stepargs:
                stepfuncname = stepargs.pop("step")
                stepname = f"{si}-{stepargs.get('name', stepfuncname)}"
                stepdir = os.path.join(workdir, steppath, stepname) if workdir else None

                nodir = (not stepdir) or (not Storage(stepdir).exists())
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
                    yield steppath, wrap_step(stepfunc, stepargs, stepdir)
                else:
                    log.warn(
                        f"Skipping step {steppath}/{stepname}, using cache instead"
                    )
                    tablefile = os.path.join(str(stepdir), "*.jsonl")
                    yield steppath, TableSet(executor.load(tablefile))
            elif "split" in stepargs:
                # Persist pre-split tables
                tableset.tables.persist()
                for split, splitargs in enumerate(stepargs["split"]):
                    splitargs = dict(splitargs)
                    splitname = splitargs.pop("name", f"{si}-split-{split}")
                    splitpath = os.path.join(steppath, splitname)
                    yield from chain_step(tableset, splitpath, si, splitargs)
            else:
                raise Exception(f"Pipeline step {si} has no step type")

        # Prepend input tables to pipeline
        streams = [("", TableSet(executor.new([])))]
        if isinstance(input_tables, Path) or isinstance(input_tables, str):
            log.info(f"Getting input tabels from path: {input_tables}")
            streams = [("", TableSet.load(input_tables, executor=executor))]
        elif isinstance(input_tables, pages.PageSource):
            log.info(f"Getting input tabels from extraction: {input_tables}")
            pipeline.insert(0, {"step": "extract", "source": input_tables})
        elif isinstance(input_tables, HashBag):
            streams = [("", TableSet(input_tables))]
        elif input_tables is not None:
            log.info(f"Getting input tabels from spec: {input_tables}")
            if not isinstance(input_tables.get("path"), list):
                input_tables["path"] = [input_tables["path"]]
            tables = TableSet.load(
                *input_tables.pop("path"), **input_tables, executor=executor
            )
            streams = [("", tables)]

        if cache and workdir:
            if force:
                try:
                    Storage(workdir).rmtree()
                except Exception as e:
                    log.error(e)
            Storage(workdir).mkdir()

        log.info(
            f"Running pipeline {getattr(pipeline, 'name', '')} of {len(pipeline)} steps in {workdir} using {executor}"
        )

        for si, args in enumerate(pipeline):
            streams = [c for wd, ts in streams for c in chain_step(ts, wd, si, args)]

        if cache:
            for _, tableset in streams:
                for _ in tableset:
                    pass
        else:
            _, tablesets = zip(*streams)
            if all(isinstance(t, TableSet) for t in tablesets):
                return tablesets[0].tables.__class__.concat(
                    [t.tables for t in tablesets]
                )
            else:
                return streams
