import hashlib


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def combine_by_first_header(table1, table2):
    tableHeaders = table1["tableHeaders"]

    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
    )
    if "headerId" not in table1:
        table1["headerId"] = get_headerId(headerText)

    headerId = table1["headerId"]

    tableData = table1["tableData"] + table2["tableData"]

    return {
        "_id": f"{headerId}-0",
        "pgId": headerId,
        "tbNr": 0,
        "type": "headerunion",
        "pgTitle": f"Header {headerId}",
        "sectionTitle": "",
        "headerId": headerId,
        "numCols": len(tableData[0]),
        "numDataRows": len(tableData),
        "numHeaderRows": len(tableHeaders),
        "numericColumns": [],
        "numTables": table1.get("numTables", 1) + table2.get("numTables", 1),
        "tableHeaders": tableHeaders,
        "tableData": tableData,
        "pivots": table1.get("pivots", [table1.get("pivot")])
        + table2.get("pivots", [table2.get("pivot")]),
    }
