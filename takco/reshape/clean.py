from collections import Counter
from itertools import combinations
import logging as log
import typing

from takco import Table

EMPTY_CELL = {
    "cellID": -1,
    "textTokens": [],
    "text": "",
    "surfaceLinks": [],
    "subtableID": -1,
    "isNumeric": False,
}


def restructure(
    tables: typing.Iterable[Table], prefix_header_rules=(), max_cols=100
) -> typing.Iterable[Table]:
    """Restructure tables.

    Performs all sorts of heuristic cleaning operations, including:

        - Remove empty columns (:meth:`takco.reshape.clean.remove_empty_columns`)
        - Deduplicate header rows (:meth:`takco.reshape.clean.deduplicate_header_rows`)
        - Remove empty header rows (:meth:`takco.reshape.clean.remove_empty_header_rows`)
        - Process rowspanning head cells (:meth:`takco.reshape.clean.process_rowspanning_head_cells`)
        - Restack horizontal schema repeats (:meth:`takco.reshape.clean.restack_horizontal_schema_repeats`)
        - Remove empty rows (:meth:`takco.reshape.clean.remove_empty_rows`)
        - Process rowspanning body cells (:meth:`takco.reshape.clean.process_rowspanning_body_cells`)

    """

    for table in tables:
        try:
            table = Table(table).to_dict()

            if table.get("numCols", 0) >= max_cols:
                continue

            if any("tdHtmlString" in c for r in table.get("tableHeaders") for c in r):
                hs = table.get("tableHeaders", [])
                if all(c.get("tdHtmlString", "")[:3] == "<td" for r in hs for c in r):
                    table["tableData"] = hs + table.get("tableData", [])
                    table["tableHeaders"] = []

            init_captions(table)

            # Analyze headers & data together
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
            remove_empty_columns(table)

            apply_prefix_header_rules(table, prefix_header_rules)

            if table["tableData"]:
                yield Table(table)
        except Exception as e:
            log.error(e)


def init_captions(table):
    table["tableCaptions"] = []
    if "tableCaption" in table and table["tableCaption"] != table["sectionTitle"]:
        caption = dict(EMPTY_CELL)
        caption["text"] = table["tableCaption"]
        caption["cellClass"] = "original"
        table["tableCaptions"].append(caption)


def remove_empty_columns(table):
    col_empty = {}
    for i, bodycol in enumerate(zip(*table["tableData"])):
        col_empty[i] = not any(c.get("text", "").strip() for c in bodycol)

    if any(col_empty.values()):
        table["tableHeaders"] = [
            [cell for c, cell in enumerate(row) if not col_empty.get(c, False)]
            for row in table["tableHeaders"]
        ]
        table["tableData"] = [
            [cell for c, cell in enumerate(row) if not col_empty.get(c, False)]
            for row in table["tableData"]
        ]
        table["numCols"] = len(table["tableData"][0])


def deduplicate_header_rows(table):
    if table["tableHeaders"] and table["tableData"]:
        lastHeaderRow = [cell.get("text", "") for cell in table["tableHeaders"][-1]]
        firstDataRow = [cell.get("text", "") for cell in table["tableData"][-1]]
        if lastHeaderRow == firstDataRow:
            table["tableData"] = table["tableData"][1:]


def process_rowspanning_head_cells(table):
    # Find header rows that are actually captions or inner-table rows
    allspanhead = []
    for row in table["tableHeaders"]:
        cells = [c for cell in row for c in [cell.get("text", "")] if c]
        s = (len(set(cells)) == 1) and all(cells)
        allspanhead.append(s)

    try:
        newTableHeaders = []
        newTableCaptions = []
        inner = False
        for r in list(range(len(table["tableHeaders"])))[::-1]:  # iterate backwards
            row = table["tableHeaders"][r]
            if allspanhead[r]:
                # If the allspan row occurs at the bottom, it's an inner-header.
                # Otherwise it's a caption
                if not inner:
                    table["tableData"].insert(0, row)
                    inner = True
                else:
                    cell = dict(row[0])
                    cell["cellClass"] = "fromheader"
                    newTableCaptions.insert(0, cell)
            else:
                newTableHeaders.insert(0, row)

        table["tableHeaders"] = newTableHeaders
        table["numHeaderRows"] = len(table["tableHeaders"])
        tableCaptions = table.setdefault("tableCaptions", [])
        tableCaptions += newTableCaptions
    except:
        pass


def restack_horizontal_schema_repeats(table):
    # Find horizontally stacked headers
    table["wasStackedHor"] = 0
    if table["tableHeaders"]:
        for i in range(2, table["numCols"]):
            a = [
                [c for cell in row[:i] for c in [cell.get("text", "")]]
                for row in table["tableHeaders"]
            ]
            b = [
                [c for cell in row[i:] for c in [cell.get("text", "")]]
                for row in table["tableHeaders"]
            ]
            a, b = list(zip(*a)), list(zip(*b))
            if all(all(ac) for ac in a) and all(all(bc) for bc in b):
                l = len(b) // len(a)
                if b == a * l:
                    table["wasStackedHor"] = l
                    table["tableHeaders"] = [row[:i] for row in table["tableHeaders"]]
                    table["numCols"] = i
                    newTableData = []
                    for j in range(l):
                        for row in table["tableData"]:
                            newTableData.append(row[j * i : (j + 1) * i])
                    table["tableData"] = newTableData
                    table["numDataRows"] = len(table["tableData"])
                    break


def remove_empty_rows(table):
    # Remove empty rows
    table["tableData"] = [
        row for row in table["tableData"] if any(cell.get("text", "") for cell in row)
    ]
    table["numDataRows"] = len(table["tableData"])


def remove_empty_header_rows(table):
    # Remove empty rows
    table["tableHeaders"] = [
        row
        for row in table["tableHeaders"]
        if any(cell.get("text", "") for cell in row)
    ]
    table["numDataRows"] = len(table["tableHeaders"])


def process_rowspanning_body_cells(table):
    # Heuristics:
    # - if normal rows and all-column-spanning rows alternate every row,
    #   the all-span rows are extra cells for the row above
    # - if there are a few all-column-spanning rows,
    #   those are extra cells for the rows below until next

    table.setdefault("tableCaptions", [])

    allspan = []
    for row in table["tableData"]:
        cells = [cell.get("text", "").lower() for cell in row]
        s = (len(set(cells)) == 1) and all(cells)
        allspan.append(s)

    if (
        table["numCols"] >= 2
        and any(allspan)
        and (not all(allspan))
        and (len(table["tableData"]) >= 2)
    ):
        row_class = [(None if r else "N") for r in allspan]

        # Wrapped cells (occur before or after every non-all-spanning row)
        if table["tableHeaders"] and sum(1 for c in row_class if c == "N") > 1:
            j = 1  # Only look one row above / below for now
            # After: move right
            allspan_after = all(
                (allspan[i + j] if not allspan[i] else True)
                for i in range(len(allspan) - j)
            )
            if allspan[-1] and allspan_after:
                for i in range(len(allspan) - j):
                    if not allspan[i] and allspan[i + j]:
                        row_class[i + j] = "R"

            # Before: move left
            allspan_before = all(
                (allspan[i - j] if not allspan[i] else True)
                for i in range(j, len(allspan))
            )
            if allspan[0] and allspan_before:
                for i in range(j, len(allspan)):
                    if not allspan[i] and allspan[i - j]:
                        row_class[i - j] = "L"

        # Footnotes (occur at bottom)
        for i in range(1, len(row_class) - 1):
            if row_class[-i] == None:
                row_class[-i] = "F"
            else:
                break

        # Subsection headers occur just above non-allspan row
        for r in range(len(row_class) - 1):
            if (not allspan[r + 1]) and (row_class[r] is None):
                row_class[r] = "H"

        # Delete all other allspan rows
        for r in range(len(row_class)):
            if row_class[r] is None:
                row_class[r] = "X"

        #         if 'N' not in row_class:
        #             print('No normal row in http://localhost:5000/table/{pgId}-{tbNr}'.format(**table), ''.join(row_class))

        # Transform table
        # - R: move to new column on the right
        # - L: move to new column on the left
        # - F: store as footnote
        # - H: copy to all rows below in a new column on the right
        # - X: discard row

        R, L, H, F = {}, {}, {}, {}
        current_H = None
        for i, row in enumerate(table["tableData"]):
            allspan_cell = dict(row[0])

            # keep track of cell origins
            allspan_cell["cellClass"] = row_class[i]
            if row_class[i] == "R":
                R[i + 1] = allspan_cell
            if row_class[i] == "L":
                L[i - 1] = allspan_cell
            if row_class[i] == "H":
                current_H = allspan_cell
            if current_H:
                H[i] = current_H
            if row_class[i] == "F":
                F[i] = allspan_cell

        if L:
            # Add to left
            table["tableHeaders"] = [
                [dict(EMPTY_CELL)] + row for row in table["tableHeaders"]
            ]
            table["tableData"] = [
                [L.get(r, dict(EMPTY_CELL))] + row
                for r, row in enumerate(table["tableData"])
            ]
            table["numCols"] += 1

        if H:
            # Add header to left
            hcell = dict(EMPTY_CELL)
            hcell["text"] = "SUBHEADER"
            table["tableHeaders"] = [
                [dict(EMPTY_CELL)] + row for row in table["tableHeaders"]
            ]
            table["tableData"] = [
                [H.get(r, dict(EMPTY_CELL))] + row
                for r, row in enumerate(table["tableData"])
            ]
            table["numCols"] += 1
        if R:
            # Add to right
            table["tableHeaders"] = [
                row + [dict(EMPTY_CELL)] for row in table["tableHeaders"]
            ]
            table["tableData"] = [
                row + [R.get(r, dict(EMPTY_CELL))]
                for r, row in enumerate(table["tableData"])
            ]
            table["numCols"] += 1

        if F:
            # Make a footnote
            for footer in F.values():
                footer["cellClass"] = "footer"
                table["tableCaptions"].append(footer)

        table["tableData"] = [
            row for r, row in enumerate(table["tableData"]) if row_class[r] == "N"
        ]

        table["numDataRows"] = len(table["tableData"])


def transpose(table, n_header_rows=0):

    transposed = list(zip(*(table["tableHeaders"] + table["tableData"])))
    n = min(n_header_rows, len(transposed))
    table["tableHeaders"], table["tableData"] = transposed[:n], transposed[n:]

    table["numDataRows"] = len(table["tableData"])
    table["numCols"] = len(table["tableData"][0])


def heuristic_transpose(table):
    if (table["numDataRows"] == 1) and (table["numCols"] == 2):
        transpose(table)


def apply_prefix_header_rules(table, prefix_header_rules):
    prefixes = {p["find"]: p["header"] for p in prefix_header_rules}

    if not any(h for hr in table["tableHeaders"] for h in hr):
        hcols = dict(enumerate(zip(*table["tableHeaders"])))
        cols = list(zip(*table["tableData"]))
        for ci, col in enumerate(cols):
            for p, headertext in prefixes.items():
                if any(c.get("text", "").startswith(p) for c in col):
                    l = len(p)
                    for c in col:
                        c["text"] = c["text"][l:]
                        for link in c.get("surfaceLinks", []):
                            link["offset"] -= l
                            link["endOffset"] -= l
                    head = hcols.setdefault(ci, [{}])
                    head[0]["text"] = headertext
                    break

        if hcols:
            for ci in range(len(cols)):
                hcols.setdefault(ci, [{"text": ""}])
            _, hcols = zip(*sorted(hcols.items()))
            table["tableHeaders"] = list(zip(*hcols))
            table["tableData"] = list(zip(*cols))
            table["numHeaderRows"] = len(table["tableHeaders"])
