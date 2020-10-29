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

def from_wdc(docs):
    for doc in docs:
        if doc['headerPosition'] == 'FIRST_ROW':
            header, *body = zip(*doc['relation'])
            yield {
                '_id': 'wdc-' + str(abs(hash(str(doc)))),
                'tbNr': doc.get('tableNum', 0),
                'pgId': doc.get('url', ''),
                'pgTitle': doc.get('pageTitle', '').strip() or doc.get('url', ''),
                'tableCaption': doc.get('title', '').strip(),
                'tableHeaders': [[{'text':c} for c in header]],
                'tableData': [[{'text':c} for c in row] for row in body],
                'numHeaderRows': 1,
                'numCols': len(header),
                'numDataRows': len(body),
            }
