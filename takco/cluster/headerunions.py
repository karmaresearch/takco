import hashlib

from ..table import Table

def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def combine_by_first_header(table1, table2):
    if not isinstance(table1, Table):
        table1 = Table(table1)
    if not isinstance(table2, Table):
        table1 = Table(table2)

    tableHeaders = table1["tableHeaders"]

    headerText = tuple(
        tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
    )
    if "headerId" not in table1:
        table1["headerId"] = get_headerId(headerText)

    headerId = table1["headerId"]

    row_offset2 = len(table1["rows"])

    annotations = {}
    if "entities" in table1:
        annotations["entities"] = table1["entities"]
    if "entities" in table2:
        for ci, ri_ents in table2["entities"].items():
            annotations.setdefault("entities", {}).setdefault(ci, {}).update(
                {str(int(ri) + row_offset2): es for ri, es in ri_ents.items()}
            )

    for table in [table1, table2]:
        for ci, classes in table.get("classes", {}).items():
            annotations.setdefault("classes", {}).setdefault(ci, {}).update(classes)

        for fromci, toci_props in table.get("properties", {}).items():
            newprops = annotations.setdefault("properties", {}).setdefault(fromci, {})
            for toci, props in toci_props:
                newprops.setdefault(toci, {}).update(props)

    rows = table1["rows"] + table2["rows"]
    links = table1["links"] + table2["links"]
    return Table({
        "_id": f"{headerId}-0",
        "pgId": headerId,
        "tbNr": 0,
        "type": "headerunion",
        "pgTitle": f"Header {headerId}",
        "sectionTitle": "",
        "headerId": headerId,
        "numCols": len(rows[0]) if rows else 0,
        "numDataRows": len(rows),
        "numHeaderRows": len(tableHeaders),
        "numericColumns": [],
        "numTables": table1.get("numTables", 1) + table2.get("numTables", 1),
        "tableHeaders": tableHeaders,
        "rows": rows,
        "links": links,
        "pivots": table1.get("pivots", [table1.get("pivot")])
        + table2.get("pivots", [table2.get("pivot")]),
        **annotations,
    })
