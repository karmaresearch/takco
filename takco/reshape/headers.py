import hashlib
import logging as log


def get_headerId(header):
    # header is a tuple of tuples.
    header = tuple(tuple(c.lower() for c in r) for r in header)
    h = hashlib.sha224(str(header).encode()).hexdigest()
    return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL


def table_get_headerId(table):
    if "tableHeaders" in table:
        log.debug(f"Getting headerId from table {table.get('_id')}")
        headertext = [
            [c.get("text", "") for c in hrow] for hrow in table["tableHeaders"]
        ]
        return get_headerId(headertext)


def get_headerobjs(tables):
    for table in tables:
        yield table.get("tableHeaders", [])
