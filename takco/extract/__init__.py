"""
This module extracts tables from html
"""
from typing import List, Dict, Iterator

from .htmltables import page_extract_tables
from .clean import (
    init_captions,
    remove_empty_columns,
    deduplicate_header_rows,
    remove_empty_header_rows,
    process_rowspanning_head_cells,
    restack_horizontal_schema_repeats,
    remove_empty_rows,
    process_rowspanning_body_cells,
)


def extract_tables(pages: Iterator[Dict]) -> Iterator[Dict]:
    """Extract tables from pages.
    
    Pages are dicts with an ``html`` field.
    Tables are dicts in Bhagavatula format.
    
    See also:
    
        - :meth:`takco.extract.htmltables.page_extract_tables`
    
    """
    for page in pages:
        yield from page_extract_tables(page.get("html"), aboutURI=page.get("about"))


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

        if table["tableData"]:
            yield table
