import re
import copy
import logging as log
import typing

from takco import Table

try:
    from bs4 import BeautifulSoup, Tag
except:
    log.error(f"Could not import BeautifulSoup")

class Extractor(object):
    """
    Extracts cells from tables with colspans and rowspans

    Based on `html-table-extractor <https://github.com/yuanxu-li/html-table-extractor>`_.

    """

    WHITESPACE = re.compile(r"\s+")

    def __init__(self, input, id_=None, **kwargs):
        # TODO: should divide this class into two subclasses
        # to deal with string and bs4.Tag separately

        # validate the input
        if not isinstance(input, str) and not isinstance(input, Tag):
            raise Exception("Unrecognized type. Valid input: str, bs4.element.Tag")

        soup = (
            BeautifulSoup(input, "html.parser").find()
            if isinstance(input, str)
            else input
        )

        # locate the target table
        if soup.name == "table":
            self._table = soup
        else:
            self._table = soup.find(id=id_)

        if "transformer" in kwargs:
            self._transformer = kwargs["transformer"]
        else:
            self._transformer = str

        self._output = []

        self._empty = BeautifulSoup("<td></td>", "html.parser")

    def parse(self):
        self._output = []
        row_ind = 0
        col_ind = 0
        for row in self._table.find_all("tr"):
            # record the smallest row_span, so that we know how many rows
            # we should skip
            smallest_row_span = 1

            for cell in row.children:
                if cell.name in ("td", "th"):
                    # check multiple rows
                    # pdb.set_trace()
                    try:
                        row_span = (
                            int(cell.get("rowspan")) if cell.get("rowspan") else 1
                        )
                    except:
                        row_span = 1

                    # try updating smallest_row_span
                    smallest_row_span = min(smallest_row_span, row_span)

                    # check multiple columns
                    try:
                        col_span = (
                            int(cell.get("colspan")) if cell.get("colspan") else 1
                        )
                    except:
                        col_span = 1

                    # find the right index
                    while True:
                        if self._check_cell_validity(row_ind, col_ind):
                            break
                        col_ind += 1

                    # insert into self._output
                    try:
                        self._insert(
                            row_ind,
                            col_ind,
                            row_span,
                            col_span,
                            self._transformer(cell),
                        )
                    except UnicodeEncodeError:
                        raise Exception(
                            "Failed to decode text; you might want to specify kwargs transformer=unicode"
                        )

                    # update col_ind
                    col_ind += col_span

            # update row_ind
            row_ind += smallest_row_span
            col_ind = 0
        return self

    def return_list(self):
        return [[(cell or self._empty) for cell in row] for row in self._output]


    def _check_validity(self, i, j, height, width):
        """
        check if a rectangle (i, j, height, width) can be put into self.output
        """
        return all(
            self._check_cell_validity(ii, jj)
            for ii in range(i, i + height)
            for jj in range(j, j + width)
        )

    def _check_cell_validity(self, i, j):
        """
        check if a cell (i, j) can be put into self._output
        """
        if i >= len(self._output):
            return True
        if j >= len(self._output[i]):
            return True
        if self._output[i][j] is None:
            return True
        return False

    def _insert(self, i, j, height, width, val):
        # pdb.set_trace()
        for ii in range(i, i + height):
            for jj in range(j, j + width):
                self._insert_cell(ii, jj, val)

    def _insert_cell(self, i, j, val):
        while i >= len(self._output):
            self._output.append([])
        while j >= len(self._output[i]):
            self._output[i].append(None)

        if self._output[i][j] is None:
            self._output[i][j] = val

    @classmethod
    def get_cell_text(cls, cell):
        if cell:
            return cls.WHITESPACE.sub(" ", str(cell.get_text(separator=" "))).strip()
        else:
            return ""

    @classmethod
    def cell_extract_wikilinks(cls, node):
        if not node:
            return

        curOffset = 0
        for a in node.find_all("a"):
            parent_text = Extractor.get_cell_text(node)
            surface = Extractor.get_cell_text(a)
            offset = parent_text[curOffset:].index(surface) + curOffset
            endOffset = offset + len(surface)
            curOffset = endOffset

            href = a.attrs.get("href")
            if href:
                target = {"href": href}
                if href.startswith("#"):
                    linkType = "PAGE"
                elif href.startswith(
                    "http://www.wikipedia.org/"
                ) or not href.startswith("http"):
                    linkType = "INTERNAL"
                    target["title"] = href.split("/")[-1]
                elif href.startswith("http"):
                    linkType = "EXTERNAL"
                else:
                    linkType = "UNKNOWN"

                yield dict(
                    offset=offset,
                    endOffset=endOffset,
                    surface=surface,
                    linkType=linkType,
                    target=target,
                )

    @classmethod
    def get_extra_links(cls, celltext, surface_pattern, surface_links):
        for m in surface_pattern.finditer(celltext):
            surface = m.group(0)
            href = surface_links.get(surface)
            target = {"href": href, "title": href.split("/")[-1]}
            yield dict(
                offset=m.start(),
                endOffset=m.end(),
                surface=surface,
                linkType="INTERNAL",
                target=target,
            )

    @classmethod
    def get_cell_dict(cls, cell, surface_pattern=None, surface_links=()):
        text = cls.get_cell_text(cell)
        links = list(cls.cell_extract_wikilinks(cell))
        if not links and surface_pattern and surface_links:
            links = list(cls.get_extra_links(text, surface_pattern, surface_links))
        return {
            "text": text,
            "surfaceLinks": links,
        }


def clean_wikihtml(cell):
    if cell:
        for img in cell.find_all("span", attrs={"title": True}):
            img.replaceWith(img.attrs.get("title"))
        reftags = {"sup": "reference", "span": "mw-ref"}
        for tag, c in reftags.items():
            for ref in cell.find_all(tag, [c]):
                for a in ref.find_all("a"):
                    a.extract()
    return cell


def str2html(s):
    return BeautifulSoup(s, "html.parser")


def htmlrows_to_dataframe(htmlrows):
    import pandas as pd

    head, body = [], []
    for row in htmlrows:
        if all(c.name == "th" for c in row):
            head.append(row)
        else:
            body.append(row)
    return pd.DataFrame(body, columns=pd.MultiIndex.from_tuples(list(zip(*head))))


def indexify_header(df, level=1, cells=None):
    import pandas as pd

    body = pd.DataFrame(df)
    head = body.columns.to_frame().T
    head.columns = range(0, head.shape[1])
    body.columns = range(0, body.shape[1])

    if cells is None:
        cells = head.apply(lambda col: all(col[0] == cell for cell in col), axis=0)

    df = body[body.columns[~cells]]
    df.columns = pd.MultiIndex.from_tuples(
        list(head[head.columns[~cells]].T.to_numpy())
    )
    indexnames = head[head.columns[cells]].T[level]
    df.index = pd.MultiIndex.from_frame(body[head.columns[cells]], names=indexnames)
    return df


def vertically_split_tables_on_subheaders(htmlrows):
    subtable = []  # type: ignore
    prev_row_is_header = True

    for row in htmlrows:
        row_is_header = all(c.name == "th" for c in row)

        # make subheader
        if row_is_header and len(row) > 1 and len(set(row)) == 1:
            for c in row:
                c.name = "td"
            row_is_header = False

        if (not prev_row_is_header) and row_is_header:
            if len(subtable) > 1:
                yield subtable
            subtable = []
        subtable.append(row)
        prev_row_is_header = row_is_header
    yield subtable


def hack_annoying_layouts(all_htmlrows):

    # Stupid football tables
    all_text = set(cell.text.strip() for row in all_htmlrows for cell in row)
    if all(s in all_text for s in ["Season", "Club", "League", "Total"]):
        first_row = all_htmlrows[1]
        new_htmlrows = []
        for row in all_htmlrows[2:]:
            if len(set(cell.text.strip() for cell in row[0:3])) == 1:
                new_htmlrows += [first_row, (first_row[0:3] + row[3:])]
            else:
                new_htmlrows += [row]
        return new_htmlrows

    return all_htmlrows

def node_extract_tables(table_node, link_surface_pattern=None, surface_links=()):
    extractor = Extractor(table_node, transformer=lambda x: x)
    extractor.parse()
    all_htmlrows = [
        [clean_wikihtml(cell) for cell in row]
        for row in extractor.return_list()
    ]

    all_htmlrows = hack_annoying_layouts(all_htmlrows)

    for htmlrows in vertically_split_tables_on_subheaders(all_htmlrows):

        numCols = max((len(row) for row in htmlrows), default=0)
        td = BeautifulSoup("<td></td>", "html.parser")
        th = BeautifulSoup("<th></th>", "html.parser")

        tableHeaders = []
        tableData = []
        for row in htmlrows:
            h, e = (
                (tableHeaders, th)
                if all(c.name == "th" for c in row)
                else (tableData, td)
            )
            row = [(row[i] if i < len(row) else e) for i in range(numCols)]
            h.append(
                [
                    Extractor.get_cell_dict(
                        cell, link_surface_pattern, surface_links
                    )
                    for cell in row
                ]
            )

        if tableData:
            numDataRows = len(tableData)
            numHeaderRows = len(tableHeaders)
            yield Table(
                dict(
                    numCols=numCols,
                    numDataRows=numDataRows,
                    numHeaderRows=numHeaderRows,
                    tableData=tableData,
                    tableHeaders=tableHeaders,
                    originalHTML=str(table_node),
                )
            )


def page_extract_tables(
    htmlpage: str, aboutURI=None, pgTitle=None, pgId=None, link_pattern=None,
    class_restrict=()
):
    """Extract tables from html in Baghavatula's json format
    
    Also vertically splits tables on rows that are all ``th`` elements (subheaders).
    
    Args:
        htmlpage: HTML string
        aboutURI: URI of page subject entity
        pgTitle: Page title
        pgId: Page ID
    """
    from bs4 import BeautifulSoup

    if not htmlpage:
        return

    extract_links_pattern = re.compile(link_pattern) if link_pattern else None

    htmlpage = htmlpage.replace('<th>scope="row"</th>', "")  # Kiwix fix hack

    soup = BeautifulSoup(htmlpage, "html.parser")

    surface_links = {}
    link_surface_pattern = None
    if extract_links_pattern is not None:
        for a in soup.find_all("a"):
            href = a.attrs.get("href")
            if href and extract_links_pattern.match(href) and len(a.text.strip()) > 1:
                surface_links[a.text.strip()] = href
        link_surface_pattern = re.compile("|".join(map(re.escape, surface_links.keys())))
        log.debug(f"Got {len(surface_links)} extra surface links")

    for page in soup.find_all("html"):
        if pgTitle is None:
            pgTitle = ""
            h = page.find("h1")
            if h:
                pgTitle = h.text.strip()
        if pgId is None:
            pgId = 0

        tableId = 0
        for table_node in page.find_all("table", list(class_restrict)):

            if '>scope="row"<' in str(table_node):
                log.debug(f"Scope in {table_node}")

            if not table_node.text.strip():
                continue
            sectionTitle = table_node.find_previous_sibling("summary")
            sectionTitle = sectionTitle.text if sectionTitle else ""

            tableCaption = table_node.find("caption")
            tableCaption = tableCaption.text if tableCaption else sectionTitle
            tableCaption = tableCaption.strip()

            for table in node_extract_tables(
                table_node, 
                link_surface_pattern=link_surface_pattern,
                surface_links=surface_links,
            ):
                tableId += 1
                log.debug(f"Extracted table {tableId} from {pgTitle}")
                
                table.update(dict(
                    _id=f"{pgId}#{tableId}",
                    pgId=pgId,
                    tableId=tableId,
                    pgTitle=pgTitle,
                    aboutURI=aboutURI,
                    sectionTitle=sectionTitle,
                    tableCaption=tableCaption,
                ))
                yield table