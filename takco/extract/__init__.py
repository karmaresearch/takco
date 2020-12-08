"""
This module extracts tables from html
"""
from typing import List, Dict, Iterable

from .htmltables import page_extract_tables
from .pages import *


def extract_tables(pages: Iterable[Page]) -> Iterable[Dict]:
    """Extract tables from pages.

    Pages are dicts with an ``html`` field.
    Tables are dicts in Bhagavatula format.

    See also:

        - :meth:`takco.extract.htmltables.page_extract_tables`

    """
    for page in pages:
        yield from page_extract_tables(page.html, aboutURI=page.about, pgId=page.url)
