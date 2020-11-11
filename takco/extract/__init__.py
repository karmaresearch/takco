"""
This module extracts tables from html
"""
from typing import List, Dict, Iterator

from .htmltables import page_extract_tables


def extract_tables(pages: Iterator[Dict]) -> Iterator[Dict]:
    """Extract tables from pages.

    Pages are dicts with an ``html`` field.
    Tables are dicts in Bhagavatula format.

    See also:

        - :meth:`takco.extract.htmltables.page_extract_tables`

    """
    for page in pages:
        yield from page_extract_tables(page.get("html"), aboutURI=page.get("about"))
