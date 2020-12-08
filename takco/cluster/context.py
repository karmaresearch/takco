import hashlib
import logging as log
import copy

from takco import Table

def tables_add_context_rows(tables, fields=()):
    """Add context to table depending on table dict fields"""
    for table in tables:
        table = Table(table).to_dict()

        for field in list(fields)[::-1]:
            empty_header = {
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
            table["headerId"] = Table.get_headerId(headerText)

            fieldtext = table.get(field, "")
            context_cells = [
                {
                    "text": fieldtext,
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
                copy.deepcopy(context_cells) + list(drow) for drow in table["tableData"]
            ]
            table["numCols"] = len(table["tableData"][0]) if table["tableData"] else 0

        n = len(fields)
        if "entities" in table:
            table["entities"] = {
                str(int(ci) + n): x for ci, x in table["entities"].items()
            }
        if "classes" in table:
            table["classes"] = {
                str(int(ci) + n): x for ci, x in table["classes"].items()
            }
        if "properties" in table:
            table["properties"] = {
                str(int(fci) + n): {str(int(tci) + n): e for tci, e in te.items()}
                for fci, te in table["properties"].items()
            }

        yield Table(table)
