import copy
import hashlib
import logging as log


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def tables_add_context_rows(tables, fields=None):
    for table in tables:
        table = copy.deepcopy(table)

        if (not fields) or ("pgTitle" in fields):
            empty_header = {
                "tdHtmlString": "<th>_Page</th>",
                "text": "_Page",
                "surfaceLinks": [],
            }
            table["tableHeaders"] = [
                [empty_header] + list(hrow) for hrow in table["tableHeaders"]
            ]
            tableHeaders = table["tableHeaders"]
            headerText = tuple(
                tuple([cell.get("text", "").lower() for cell in r])
                for r in tableHeaders
            )
            table["headerId"] = get_headerId(headerText)

            pgId = table.get("pgId")
            context_cells = [
                {
                    "text": table["pgTitle"],
                    "tdHtmlString": f"<td><a href='/wiki/{table['pgTitle']}'>{table['pgTitle']}</a></td>",
                    "surfaceLinks": [
                        {
                            "offset": 0,
                            "endOffset": len(table["pgTitle"]),
                            "linkType": "INTERNAL",
                            "target": {"id": pgId, "href": table["pgTitle"]},
                        }
                    ],
                }
            ]
            table["tableData"] = [
                context_cells + list(drow) for drow in table["tableData"]
            ]
            table["numCols"] = len(table["tableData"][0])

        yield table
