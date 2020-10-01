import logging as log

from .headers import table_get_headerId, get_headerId, get_headerobjs
from .compound import SpacyCompoundSplitter
from . import findpivot
from .clean import (
    init_captions,
    remove_empty_columns,
    deduplicate_header_rows,
    remove_empty_header_rows,
    process_rowspanning_head_cells,
    restack_horizontal_schema_repeats,
    remove_empty_rows,
    process_rowspanning_body_cells,
    heuristic_transpose,
)


from typing import Dict, List, Iterator, Any, Tuple

from collections import Counter
import json
import copy

L_COLHEADER = "_Variable"
R_COLHEADER = "_Value"


def unpivot(
    header: List[List[Any]],
    body: List[List[Any]],
    level: int,
    colfrom: int,
    colto: int,
    leftcolheader=L_COLHEADER,
    emptycell="",
    rightcolheader=R_COLHEADER,
    merge_header_func=lambda x: x[0],
    wrap_funcs=(json.dumps, json.loads),
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Unpivot a table.

    Header and body should both be a matrix of values for which ``x == eval(str(x))``,
    where ``eval`` and ``str`` can be set using ``wrap_funcs``.

    Args:
        header: Table header rows
        body: Table body rows
        level: Index of header row to pivot
        colfrom: Index of leftmost pivot column
        colto: Index of rightmost pivot column

        leftcolheader: New header for left column
        emptycell: New value for empty cells
        rightcolheader: New header for right column (if single header row)
        merge_header_func: Function for merging a collection of header cells
        wrap_funcs: Tuple of ``(str, eval)`` functions for encoding values to hashable
    """
    import pandas as pd

    nhead = len(header)

    if all(all(all(type(c) == str for c in r) for r in p) for p in (header, body)):
        wrap_funcs = (lambda x: x, lambda x: x)
    enc, dec_ = wrap_funcs
    dec = lambda x: dec_(x) if x else copy.deepcopy(emptycell)  # must be a better way..

    allrows = [[enc(c) for c in row] for row in header + body]

    df = pd.DataFrame(allrows[nhead:])
    df.columns = pd.MultiIndex.from_arrays(allrows[:nhead])

    # Make a table that is indexed by non-pivot columns
    colrange = range(colfrom, colto + 1)
    id_cols = [df.columns[i] for i in range(len(df.columns)) if i not in colrange]
    value_cols = [df.columns[i] for i in colrange]
    df = df[value_cols].set_index(pd.MultiIndex.from_frame(df[id_cols]))
    df.index.names = [merge_header_func(headers) for headers in df.index.names]

    if nhead > 1:
        # For tables with multiple header rows, the right columns get their own headers
        df = df.stack(level)
        df.index.names = df.index.names[:-1] + [enc(leftcolheader)]
        df = df.reset_index()
    else:
        # For tables with a single header row, the right column needs to be given
        df.columns = [c[0] for c in df.columns]
        df = df.stack()
        df.index.names = df.index.names[:-1] + [(enc(leftcolheader),)]
        df = df.to_frame((enc(rightcolheader),)).reset_index()

    head = df.columns.to_frame().applymap(dec).T.values
    body = df.fillna(enc(emptycell)).applymap(dec).values
    return [list(row) for row in head], [list(row) for row in body]


def yield_pivots(headerobjs, use_heuristics: Dict[str, Dict] = None, heuristics=None):
    """Detect headers that should be unpivoted using heuristics."""
    if not heuristics:
        heuristics = {
            hname: h.init_class(**findpivot.__dict__)
            for hname, h in use_heuristics.items()
        }

    for headerobj in headerobjs:

        headertext = [[c.get("text", "") for c in hrow] for hrow in headerobj]

        if headertext:

            pivot_size = Counter()
            for hname, h in heuristics.items():
                for level, colfrom, colto in h.find_longest_pivots(headerobj):
                    pivot_size[(level, colfrom, colto, hname)] = colto - colfrom

            # Get longest pivot
            for (level, colfrom, colto, hname), _ in pivot_size.most_common(1):
                log.debug(f"Found pivot {(level, colfrom, colto)} using {hname}")

                old_headerId = get_headerId(headertext)

                try:
                    dummy = [[str(ci) for ci in range(len(headertext[0]))]]
                    unpivot(headertext, dummy, level, colfrom, colto)

                    yield {
                        "headerId": old_headerId,
                        "level": level,
                        "colfrom": colfrom,
                        "colto": colto,
                        "heuristic": hname,
                    }
                except Exception as e:
                    log.debug(f"Failed to unpivot header {headertext} due to {e}")


def unpivot_tables(
    tables: Iterator[Dict],
    headerId_pivot: Dict[str, Dict] = None,
    use_heuristics: Dict[str, Dict] = (),
):
    """Unpivot tables."""

    heuristics = {
        hname: h.init_class(**findpivot.__dict__) for hname, h in use_heuristics.items()
    }

    for table in tables:

        headerText = [
            [c.get("text", "") for c in hrow] for hrow in table["tableHeaders"]
        ]
        if "headerId" not in table:
            table["headerId"] = get_headerId(headerText)

        pivot = None
        if headerId_pivot is not None:
            pivot = headerId_pivot.get(table["headerId"])
        else:
            headers = [table.get("tableHeaders", [])]
            for p in yield_pivots(headers, heuristics=heuristics):
                pivot = p

        if pivot and headerText:

            log.debug(f"Unpivoting {table.get('_id')}")
            try:

                level, colfrom, colto = pivot["level"], pivot["colfrom"], pivot["colto"]
                leftcolheader = L_COLHEADER
                rightcolheader = R_COLHEADER
                if level >= len(headerText):
                    log.debug(f"Unpivot level is too big! ({level}, {headerText})")
                    return table

                # Allow heuristics to split the colheader
                heuristic = heuristics.get(pivot["heuristic"])
                if heuristic:
                    splits = heuristic.split_header(headerText[level], colfrom, colto)
                    splitheaders = []
                    for ci, (head, cell) in enumerate(splits):
                        links = table["tableHeaders"][level][ci]["surfaceLinks"]
                        splitheaders.append(
                            (
                                {
                                    "text": head or "",
                                    "tdHtmlString": f"<td>{head}</td>",
                                    "surfaceLinks": links,  # TODO adjust link offsets
                                },
                                {
                                    "text": cell or "",
                                    "tdHtmlString": f"<td>{cell}</td>",
                                    "surfaceLinks": links,  # TODO adjust link offsets
                                },
                            )
                        )
                    if splitheaders:
                        log.debug(f"Splitting pivot header {table['headerId']}")
                        above, below = zip(*splitheaders)
                        table["tableHeaders"][level] = below
                        table["tableHeaders"].insert(level, above)
                        pivot["level"] = level = level + 1

                # If pivot spans entire header, discard table!
                if (level == colfrom == 0) and (colto == len(headerText[0]) - 1):
                    log.info(f"Discarded table {pgId}-{tbNr}")
                    return

                leftcolheader = {
                    "text": L_COLHEADER,
                    "tdHtmlString": f"<th>{L_COLHEADER}</th>",
                }
                rightcolheader = {
                    "text": R_COLHEADER,
                    "tdHtmlString": f"<th>{R_COLHEADER}</th>",
                }
                emptycell = {"text": ""}

                headrows, bodyrows = table["tableHeaders"], table["tableData"]

                headrows, bodyrows = unpivot(
                    headrows,
                    bodyrows,
                    level,
                    colfrom,
                    colto,
                    leftcolheader=leftcolheader,
                    emptycell=emptycell,
                    rightcolheader=rightcolheader,
                )
                if not bodyrows:
                    log.info(f"Pivoting {pgId}-{tbNr} resulted in no data")
                    return

                oldHeaderId = table["headerId"]
                headerText = [[c.get("text", "") for c in r] for r in headrows]
                newHeaderId = get_headerId(headerText)
                table["headerId"] = newHeaderId

                table["pivot"] = pivot
                table["tableHeaders"] = headrows
                table["tableData"] = bodyrows
                table["numCols"] = len(bodyrows[0])
                table["numDataRows"] = len(bodyrows)
                table["numHeaderRows"] = len(headrows)

                m = min(table["numCols"] - 1, colto + 1)
                table.setdefault("numericColumns", [])
                table["numericColumns"] = (
                    table["numericColumns"][:colfrom] + table["numericColumns"][m:]
                )
                log.debug(f"Pivoted table {table.get('_id')}")
            except Exception as e:
                log.debug(f"Cannot pivot table {table.get('_id')} due to {e}")

        yield table


def split_compound_columns(tables, **kwargs):
    """Using an NLP pipeline, detect and split compound columns"""

    splitter = SpacyCompoundSplitter(**kwargs)
    log.info(f"Splitting compound columns using {splitter}")

    for table in tables:

        newcols = []
        headcols = list(zip(*table.get("tableHeaders", [])))
        datacols = list(zip(*table.get("tableData", [])))
        numheaders = len(headcols[0]) if headcols else 0

        for ci, (hcol, dcol) in enumerate(zip(headcols, datacols)):
            splits = list(splitter.find_splits(dcol))
            if splits:
                log.debug(
                    f"Found {len(splits)} splits in column {ci} of {table.get('_id')}: {list(zip(*splits))[:2]}"
                )
                for part, typ, newcol in splits:
                    newhcol = list(hcol)
                    if newhcol:
                        newhcol[-1] = dict(newhcol[-1])
                        newhcol[-1]["text"] = newhcol[-1].get("text", "") + " " + part
                    newcols.append((newhcol, newcol))
            else:
                newcols.append((hcol, dcol))

        if newcols:
            headcols, datacols = zip(*newcols)
            table["tableHeaders"] = list(zip(*headcols))
            table["tableData"] = list(zip(*datacols))

        yield table


def restructure(tables: Iterator[Dict]) -> Iterator[Dict]:
    """Restructure tables.

    Performs all sorts of heuristic cleaning operations, including:

        - Remove empty columns (:meth:`takco.extract.clean.remove_empty_columns`)
        - Deduplicate header rows (:meth:`takco.extract.clean.deduplicate_header_rows`)
        - Remove empty header rows (:meth:`takco.extract.clean.remove_empty_header_rows`)
        - Process rowspanning head cells (:meth:`takco.extract.clean.process_rowspanning_head_cells`)
        - Restack horizontal schema repeats (:meth:`takco.extract.clean.restack_horizontal_schema_repeats`)
        - Remove empty rows (:meth:`takco.extract.clean.remove_empty_rows`)
        - Process rowspanning body cells (:meth:`takco.extract.clean.process_rowspanning_body_cells`)

    """
    for table in tables:
        init_captions(table)

        # Analyze headers & data together
        remove_empty_columns(table)
        deduplicate_header_rows(table)

        # Analyze header
        remove_empty_header_rows(table)
        process_rowspanning_head_cells(table)
        restack_horizontal_schema_repeats(table)
        table["tableHeaders"] = [h for h in table["tableHeaders"] if h]

        # Analyze body
        remove_empty_rows(table)
        process_rowspanning_body_cells(table)
        heuristic_transpose(table)

        if table["tableData"]:
            yield table
