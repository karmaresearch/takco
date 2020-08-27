from itertools import combinations
from collections import Counter
import hashlib


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def get_colspan_repeats(rows):
    header_colspan, header_repeats = [], []
    try:
        for row in rows:
            colspan = [1 for _ in row]
            repeats = {}
            c, span = None, 1
            for ci, cell in enumerate(list(row) + [None]):
                cell = str(cell)
                if cell == c:
                    span += 1
                else:
                    for j in range(1, span + 1):
                        colspan[ci - j] = span
                    span = 1
                    repeats[c] = repeats.get(c, 0) + 1
                c = cell
            repeats = [repeats.get(str(cell), 0) for cell in row]
            header_colspan.append(colspan)
            header_repeats.append(repeats)
    except:
        print(rows)
    return header_colspan, header_repeats


def yield_table_csv(table):
    f = io.StringIO()
    cw = csv.writer(f)
    for r in table["tableHeaders"]:
        cw.writerow([cell.get("text", "") for cell in r])
    for r in table["tableData"]:
        cw.writerow([cell.get("text", "") for cell in r])
    yield dict(
        pgId=table["pgId"],
        tbNr=table["tbNr"],
        csv=f.getvalue().encode("unicode-escape").decode(),
    )


def yield_table_headers(table):
    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r])
        for r in table["tableHeaders"]
    )
    header_colspan, header_repeats = get_colspan_repeats(headerText)
    for ri, header_row in enumerate(table["tableHeaders"]):
        for ci, cell in enumerate(header_row):
            yield {
                "headerId": table["headerId"],
                "row": ri,
                "col": ci,
                "cellText": headerText[ri][ci],
                "colspan": header_colspan[ri][ci],
                "repeats": header_repeats[ri][ci],
            }


def yield_table_linkshead(table):
    for ri, header_row in enumerate(table["tableHeaders"]):
        for ci, cell in enumerate(header_row):
            for link in cell.get("surfaceLinks", []):
                if link.get("linkType", None) == "INTERNAL":
                    target = link.get("target", {})
                    yield {
                        "headerId": table["headerId"],
                        "row": ri,
                        "col": ci,
                        "cellOffset": link.get("offset", -1),
                        "cellEndOffset": link.get("endOffset", -1),
                        "linkId": target.get("id", -1),
                    }


def yield_table_tablecaptions(table):
    for ci, cell in enumerate(table.get("tableCaptions", [])):
        yield {
            "pgId": table["pgId"],
            "tbNr": table["tbNr"],
            "nr": ci,
            "cellText": cell.get("text", ""),
            "cellClass": cell.get("cellClass", ""),
        }


import csv, io


def yield_tuples(table):
    # Yield (database-table, tuple) pairs for analysis db

    if "tbNr" not in table:
        table["tbNr"] = table["tableId"]
    pgId, tbNr = table["pgId"], table["tbNr"]

    for csvdb in yield_table_csv(table):
        yield "csvdb", csvdb

    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r])
        for r in table["tableHeaders"]
    )
    table["headerId"] = get_headerId(headerText)

    ## HEADERS ##
    for headers in yield_table_headers(table):
        yield "headers", headers
    for linkshead in yield_table_linkshead(table):
        yield "linkshead", linkshead

    # Captions table
    for tablecaptions in yield_table_tablecaptions(table):
        yield "tablecaptions", tablecaptions

    ## CELLS ##
    dataText = tuple(
        tuple([cell.get("text", "").lower() for cell in r]) for r in table["tableData"]
    )
    table["nRowsIdentical"] = len(dataText) - len(set(dataText))

    table.setdefault("numericColumns", [])
    pcount = Counter()
    for ri, row in enumerate(table["tableData"]):
        for ci, cell in enumerate(row):
            spans_l = ci != 0 and dataText[ri][ci] == dataText[ri][ci - 1]
            spans_r = ci + 1 < len(row) and dataText[ri][ci] == dataText[ri][ci + 1]

            # Write cell and link table rows
            if not (spans_l or spans_r) and ci not in table["numericColumns"]:
                yield "cells", {
                    "pgId": pgId,
                    "tbNr": tbNr,
                    "row": ri,
                    "col": ci,
                    "cellText": cell.get("text", "").lower(),
                    "cellClass": cell.get("cellClass", ""),
                }

                for link in cell.get("surfaceLinks", []):
                    if link.get("linkType", None) == "INTERNAL":
                        target = link.get("target", {})
                        yield "links", {
                            "pgId": pgId,
                            "tbNr": tbNr,
                            "row": ri,
                            "col": ci,
                            "cellOffset": link.get("offset", -1),
                            "cellEndOffset": link.get("endOffset", -1),
                            "linkId": target.get("id", -1),
                        }

    yield "tables", table
