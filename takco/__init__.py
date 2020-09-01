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


def _download(ent_abouturl, encoding=None):
    import requests

    for e, url in ent_abouturl:
        result = requests.get(url)
        if encoding:
            if encoding == "guess":
                result.encoding = result.apparent_encoding
            else:
                result.encoding = encoding
        if result.status_code == 200:
            yield {
                "url": url,
                "about": e,
                "html": result.text,
            }


def wiki(
    dbconfig: Config = {"class": "GraphDB", "store": {"class": "SparqlStore"}},
    pred: str = None,
    obj: str = None,
    urlprefix: str = "https://en.wikipedia.org/wiki/",
    localurlprefix: str = None,
    sample: int = None,
    encoding: str = None,
    executor: str = None,
):
    """
    Download Wikipedia articles
    
    Downloads HTML files for entities that have a certain predicate/object in a DB.
    
    Args:
        dbconfig: DB configuration (default: Wikidata)
        pred: Predicate URI
        obj: Object URI
        urlprefix: Wikipedia URL prefix
        localprefix: Wikipedia URL prefix for locally hosted pages
        encoding: Force encoding (use "guess" for guessing)
    """
    executor = (executor or HashBag) if type(executor) != str else globals()[executor]

    from . import link

    db = Config(dbconfig).init_class(**link.__dict__)

    ent_abouturl = []
    for e, ps in db.pages_about([None, pred, obj]).items():
        for pageurl in ps:
            if localurlprefix:
                pageurl = pageurl.replace(urlprefix, localurlprefix)
            ent_abouturl.append((e, pageurl))
    ent_abouturl = ent_abouturl[:sample]

    log.info(f"Downloading {len(ent_abouturl)} URLs with executor {executor}")
    return executor(ent_abouturl)._pipe(_download, encoding=encoding)


class TableSet(HashBag):
    """A set of tables that can be clustered and linked.
    """

    @classmethod
    def _from_csvs(cls, *fnames):
        log.info(f"Reading csv tables")
        import csv

        return cls(
            {
                "tableData": [
                    [{"text": c} for c in row] for row in csv.reader(Path(fname).open())
                ],
                "_id": fname,
            }
            for fname in fnames
        )

    @classmethod
    def _load(cls, *vals):
        if all(str(val).endswith("csv") for val in vals):
            return cls._from_csvs(*vals)
        else:
            return super()._load(*vals)

    @classmethod
    def dataset(
        cls,
        params: Config,
        datadir: Path = None,
        resourcedir: Path = None,
        assets: typing.List[Config] = (),
        sample: int = None,
        executor: str = None,
    ):
        """Load tables from a dataset
        
        See also: :mod:`takco.evaluate.dataset`
        
        Args:
            params: Dataset parameters
            datadir: Data directory
            resourcedir: Resource directory
            assets: Asset specifications
            sample: Sample N tables
        """
        executor = (
            (executor or HashBag) if type(executor) != str else globals()[executor]
        )
        assets = {a.get("name", a.get("class")): Config(a) for a in assets}
        import itertools
        from . import evaluate

        params = Config(params, **assets)
        log.info(f"Loading dataset {params}")
        ds = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **params)
        return executor(itertools.islice(ds.get_unannotated_tables(), sample))

    @classmethod
    def extract(
        cls, source: typing.Union[Config, HashBag], executor: str = None,
    ):
        """Collect tables from HTML files
        
        See also: :mod:`takco.extract`

        Args:
            source: Source of HTML files
            workdir: Working directory for caching
            kbdir: Directory of Trident KB
        """
        if isinstance(source, dict):
            for k, v in source.items():
                htmlpages = globals()[k](**v, executor=executor)
                break
        elif isinstance(source, HashBag):
            htmlpages = source
        else:
            htmlpages = HashBag(source)

        from .extract import extract_tables, restructure

        tables = htmlpages._pipe(extract_tables, desc="Extract")
        tables = tables._pipe(restructure, desc="Restructure")
        return tables

    def reshape(
        tables: TableSet,
        workdir: Path = None,
        kbdir: Path = None,
        heuristics: typing.List[Config] = ({"class": "NumSuffix"},),
        load_user: str = None,
        split_compound_columns: bool = False,
    ):
        """Reshape tables
        
        See also: :mod:`takco.reshape`

        Args:
            workdir: Working Directory
            kbdir: Directory of Trident KB

            heuristics: Use a subset of available heuristics
            load_user: Load into db as user

        """
        heuristics = {h.get("name", h.get("class")): Config(h) for h in heuristics}
        from . import reshape

        log.info(f"Reshaping with heuristics {heuristics}")

        tables.persist()
        headers = tables._fold(reshape.table_get_headerId, lambda x, y: x)
        headers = headers._pipe(reshape.get_headerobjs)
        pivots = headers._pipe(reshape.yield_pivots, use_heuristics=heuristics)
        headerId_pivot = {p["headerId"]: p for p in pivots}
        log.info(f"Found {len(headerId_pivot)} pivots")

        log.info(f"Reshaping with heuristics {heuristics}")

        tables = tables._pipe(
            reshape.unpivot_tables,
            headerId_pivot,
            use_heuristics=heuristics,
            desc="Unpivoting",
        )

        if split_compound_columns:
            tables = tables._pipe(reshape.split_compound_columns, desc="Uncompounding")

        return tables

    def cluster(
        tables: TableSet,
        workdir: Path = None,
        addcontext: typing.List[str] = (),
        headerunions: bool = True,
        matchers: typing.List[Config] = ({"class": "CellJaccMatcher"}),
        agg_func: str = "mean",
        agg_threshold: float = 0,
        edge_exp: float = 1,
    ):
        """Cluster tables
        
        See also: :mod:`takco.cluster`

        Args:
            tables: Tables
            
            workdir: Working Directory

            addcontext: Add these context types
            headerunions: make header unions
            matchers: Use only these matchers
            agg_func: Matcher aggregation function
            agg_threshold: Matcher aggregation threshold
            edge_exp : Exponent of edge weight for Louvain modularity
        """
        matchers = {m.get("name", m.get("class")): Config(m) for m in matchers}

        import sqlite3
        import pandas as pd
        from . import cluster

        if addcontext:
            tables = tables._pipe(cluster.tables_add_context_rows)

        if headerunions:
            tables = tables._fold(
                cluster.table_get_headerId, cluster.combine_by_first_header
            )

        if matchers:

            tables = tables._offset("tableIndex", "tableIndex", default=1)
            tables = tables._offset("numCols", "columnIndexOffset")

            # Collect index
            tables.persist()
            alltables = list(tables)
            index = pd.concat(tables._pipe(cluster.make_column_index_df))
            Path(workdir).mkdir(exist_ok=True, parents=True)
            fpath = Path(workdir) / Path("indices.sqlite")
            log.info(f"Opening sqlitedb {fpath}")
            with sqlite3.connect(fpath) as con:
                index.to_sql("indices", con, index_label="i", if_exists="replace")
                con.execute("create index colOffset on indices(columnIndexOffset)")

            clusters = cluster.cluster(
                tables=alltables,
                dirpath=workdir,
                matcher_kwargs=matchers,
                agg_func=agg_func,
                agg_threshold=agg_threshold,
                edge_exp=edge_exp,
            )
            tables = tables.__class__(list(clusters))

        return tables

    def integrate(
        tables: TableSet,
        searcher_config: Config = {
            "class": "RDFSearcher",
            "statementURIprefix": "http://www.wikidata.org/entity/statement/",
            "store": {"class": "SparqlStore"},
        },
        kbs: typing.List[Config] = (),
        pfd_threshold: float = 0.9,
    ):
        """Integrate tables with a KB
        
        See also: :meth:`takco.link.integrate`
        
        Args:
            tables: Tables to integrate
            workdir: Working directory
            kbdir: Knowledge Base directory
            pfd_threshold:
        """
        kbs = {k.get("name", k.get("class")): Config(k) for k in kbs}
        from . import link

        searcher_config = Config(searcher_config, **kbs)
        log.info(f"Integrating with config {searcher_config}")

        return tables._pipe(
            link.integrate, searcher_config, pfd_threshold=pfd_threshold
        )

    def link(
        tables: TableSet,
        linker_config: Config = {
            "class": "First",
            "searcher": {"class": "MediaWikiAPI"},
        },
        lookup_config: Config = {"class": "MediaWikiAPI"},
        lookup_cells: bool = False,
        kbs: typing.List[Config] = (),
        usecols: str = [],
    ):
        """Link table entities to KB
        
        See also: :meth:`takco.link.link`
        
        Args:
            tables: Tables to link
            linker: Entity Linker config
            usecols: Columns to use
        """
        kbs = {k.get("name", k.get("class")): Config(k) for k in kbs}
        from . import link

        if lookup_config:
            lookup_config = Config(lookup_config, **kbs)
            log.info(f"Looking up hyperlinks with config {lookup_config}")
            tables = tables._pipe(
                link.lookup_hyperlinks,
                lookup_config=lookup_config,
                lookup_cells=lookup_cells,
            )

        if linker_config:
            linker_config = Config(linker_config, **kbs)
            log.info(f"Linking with config {linker_config}")
            tables = tables._pipe(link.link, linker_config, usecols=usecols,)

        return tables

    def score(
        tables: TableSet,
        labels: Config,
        assets: typing.List[Config] = (),
        datadir: Path = None,
        resourcedir: Path = None,
        report: typing.Dict = None,
        keycol_only: bool = False,
    ):
        """Calculate evaluation scores
        
        See also: :mod:`takco.evaluate.score`
        
        Args:
            tables: Table with predictions to score
            labels: Annotated data config
            assets: Asset specifications
            datadir: Data directory
            resourcedir: Resource directory
            report: Report config
            keycol_only: Only report results for key column
        """
        assets = {a.get("name", a.get("class")): Config(a) for a in assets}

        from . import evaluate

        flatten = {
            "entities": evaluate.score.flatten_entity_annotations,
            "properties": evaluate.score.flatten_property_annotations,
            "classes": evaluate.score.flatten_class_annotations,
        }
        tasks = list(flatten)

        annot = Config(labels, **assets)
        dset = evaluate.dataset.load(resourcedir=resourcedir, datadir=datadir, **annot)

        all_gold = {}
        all_pred = {}
        table_annot = dset.get_annotated_tables()
        for _id, goldtable in table_annot.items():
            key = [_id]
            for task in tasks:
                gold = goldtable.get(task, {})
                if gold:
                    log.debug(f"Scoring {task} of {_id}")
                    all_gold.setdefault(task, {}).update(
                        dict(flatten[task](gold, key=key))
                    )

        for table in tables:
            _id = table["_id"]
            key = [_id]
            goldtable = table_annot.get(_id, {})
            for task in tasks:
                gold = goldtable.get(task, {})
                pred = table.get(task, {})
                if keycol_only and pred.get(str(table.get("keycol"))):
                    keycol = str(table.get("keycol"))
                    pred = {keycol: pred.get(keycol)}

                if gold:
                    all_pred.setdefault(task, {}).update(
                        dict(flatten[task](pred, key=key))
                    )
                else:
                    log.debug(f"No {task} annotations for {_id}")

        scores = {}
        for task in tasks:
            gold, pred = all_gold.get(task, {}), all_pred.get(task, {})
            log.info(f"Collected {len(gold)} gold and {len(pred)} pred for task {task}")

            task_scores = evaluate.score.classification(gold, pred)
            task_scores["predictions"] = len(pred)
            scores[task] = task_scores

        return tables.__class__([{"score": scores}])
