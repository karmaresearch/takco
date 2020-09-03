import copy
import hashlib
import logging as log


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def tables_add_context_rows(tables, fields=()):
    """Add context to table depending on table dict fields"""
    for table in tables:
        table = copy.deepcopy(table)

        for field in fields:
            empty_header = {
                "tdHtmlString": f"<th>_{field}</th>",
                "text": f"_{field}",
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

            fieldtext = table.get(field, "")
            context_cells = [
                {
                    "text": fieldtext,
                    "tdHtmlString": f"<td><a href='/wiki/{fieldtext}'>{fieldtext}</a></td>",
                    "surfaceLinks": [
                        {
                            "offset": 0,
                            "endOffset": len(fieldtext),
                            "linkType": "INTERNAL",
                            "target": {"href": fieldtext},
                        }
                    ],
                }
            ]
            table["tableData"] = [
                context_cells + list(drow) for drow in table["tableData"]
            ]
            table["numCols"] = len(table["tableData"][0])

        yield table
