from pathlib import Path
import logging as log

from .headerunions import combine_by_first_header
from .context import tables_add_context_rows
from .cluster import *


def table_get_headerId(table):
    """Get the hash for a table header (create it if it isn't set)"""

    if not any(table["tableHeaders"]):
        return table["_id"]

    if "headerId" not in table:
        tableHeaders = table["tableHeaders"]

        headerText = tuple(
            tuple([cell.get("text", "").lower() for cell in r]) for r in tableHeaders
        )
        return util.get_headerId(headerText)
    else:
        return table["headerId"]
