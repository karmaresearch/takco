import sys
from typing import List, Dict, Union
import logging as log

from .base import *
from .external import *
from .trident import *
from .sqlite import *
from .pickle import *
from .elasticsearch import *
from .rdf import *

from .linkers import *

from .integrate import *
from .profile import *
from .types import *


from rdflib import Graph, URIRef

try:
    from rdflib_hdt import HDTStore
except:
    log.info(f"Library rdflib_hdt is not available")


def get_hrefs(datarows, lookup_cells=False):
    def cell_ok(c):
        return bool(c.get("text") and not c.get("text", "").isnumeric())

    return [
        [
            [
                target.get("href", target.get("title", "")).rsplit("/", 1)[-1]
                for l in c.get("surfaceLinks", [])
                for target in [l.get("target", {})]
            ]
            + ([c.get("text").strip()] if lookup_cells and cell_ok(c) else [])
            for c in row
        ]
        for row in datarows
    ]


def lookup_hyperlinks(tables: List[dict], lookup: Lookup, lookup_cells=False):
    """Lookup the (Wikipedia) hyperlinks inside cells for entity links

    Args:
        tables: Tables to link
        lookup_config: Configuration hash for a :mod:`takco.link.base.Lookup`
            object
    """
    assert isinstance(lookup, Lookup)
    with lookup:
        for table in tables:
            log.debug(f"Looking up hyperlinks of {table.get('_id')} using {lookup}")
            hrefs = get_hrefs(table.get("tableData", []), lookup_cells=lookup_cells)
            ents = table.setdefault("entities", {})
            for ci, ri_ents in lookup.lookup_cells(hrefs).items():
                for ri, es in ri_ents.items():
                    ents.setdefault(ci, {}).setdefault(ri, {}).update(es)
            yield table

            if hasattr(lookup, "flush"):
                lookup.flush()


def link(
    tables: List[dict], linker: Linker, usecols: Union[str, List[int]] = [],
):
    """Link table entities to KB

    Args:
        tables: Tables to link
        linker_config: Entity Linker config
        usecols: Columns to use (table attribute name or list of column indexes)
    """
    assert isinstance(linker, Linker)
    with linker:
        for table in tables:
            rows = [[c.get("text", "") for c in row] for row in table["tableData"]]
            if not rows:
                log.debug(f"No rows in table {table.get('_id')}")

            # Restrict columns to link (e.g. 'keycol', or 'entcols')
            def isnum(col):
                num = lambda x: x.translate(str.maketrans("", "", "-.,%")).isnumeric()
                return sum(int(num(c)) for c in col) / len(col) > 0.5

            table["non_numeric_cols"] = [
                i for i, c in enumerate(zip(*rows)) if not isnum(c)
            ]

            def heur(col):
                isnum = lambda x: x.translate(str.maketrans("", "", "-.,%")).isnumeric()
                numscore = sum(int(isnum(c)) for c in col) / len(col)
                uniqscore = len(set(col)) / len(col)
                return (numscore < 0.5) and (uniqscore > 0.9)

            heuristic_keys = [i for i, c in enumerate(zip(*rows)) if heur(c)]
            table["heuristic_key"] = heuristic_keys[0] if heuristic_keys else []

            table_usecols = table.get(str(usecols), [])
            if type(table_usecols) != list:
                table_usecols = [table_usecols]
            if not all(type(c) == int for c in table_usecols):
                log.debug(
                    f"Skipping table {table.get('_id')}, usecols = {table_usecols}"
                )
                continue

            if table_usecols:
                log.debug(
                    f"Linking columns {table_usecols} of table {table.get('_id')}"
                )
            else:
                log.debug(f"Linking table {table.get('_id')}")

            links = linker.link(
                rows, usecols=table_usecols, existing=table
            )
            table.update(links)
            yield table

            if hasattr(linker, "flush"):
                linker.flush()


def get_col_cell_ents(table):
    ents = table.get("entities", {})
    for ci, col in enumerate(zip(*table.get("tableData", []))):
        ri_ents = ents.get(str(ci), {})
        yield [
            (cell.get("text", ""), ri_ents.get(str(ri), {}))
            for ri, cell in enumerate(col)
        ]


def coltypes(tables: List[dict], typer: Typer):
    """Detect column types
    
    Args:
        tables: Tables to link
    """
    assert isinstance(typer, Typer)
    with typer:
        for table in tables:

            # Find column types
            ci_classes = table.setdefault("classes", {})
            for ci, cell_ents in enumerate(get_col_cell_ents(table)):
                cell_ents = list(dict(cell_ents).items())

                cls_score = typer.coltype(cell_ents)

                ci_classes.setdefault(str(ci), {}).update(cls_score)

            yield table


def integrate(
    tables: List[dict], db: NaryDB, pfd_threshold=0.9, typer: Typer = None,
):
    """Integrate tables with n-ary relations

    Args:
        tables: Tables to link
        kbdir: KB directory (TODO config)
        pfd_threshold: Probabilistic Functional Dependency Threshold
    """
    assert isinstance(db, NaryDB)
    with db:

        if typer is not None:
            tables = coltypes(tables, typer=typer)

        for table in tables:
            log.debug(
                "Integrating table %s (%d rows)",
                table.get("_id"),
                len(table.get("tableData", [])),
            )

            # Find key column
            profiler = PFDProfiler()
            ci_literal = {
                int(ci): any(SimpleTyper().is_literal_type(t) for t in ts)
                for ci, ts in table.get("classes", {}).items()
            }
            usecols = [ci for ci in range(table["numCols"]) if not ci_literal.get(ci)]
            rows = [[c.get("text") for c in row] for row in table.get("tableData", [])]
            keycol = profiler.get_keycol(rows, usecols)
            table["keycol"] = keycol
            log.debug(f"Got keycol {keycol}")

            ents = table.get("entities", {})
            row_entsets = [
                [
                    set(URIRef(s) for s in ents.get(str(ci), {}).get(str(ri), {}) if s)
                    for ci, _ in enumerate(row)
                ]
                for ri, row in enumerate(rows)
            ]
            tocol_fromcolprop = db.integrate(rows, row_entsets)
            log.debug(f"Got tocol_fromcolprop {tocol_fromcolprop}")
            properties = {}
            for tocol, fromcolprop in tocol_fromcolprop.items():
                for fromcol, prop in fromcolprop.items():
                    properties.setdefault(str(fromcol), {}).setdefault(str(tocol), prop)
            table.update({"properties": properties})

            yield table
