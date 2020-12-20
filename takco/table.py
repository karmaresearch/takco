import logging as log
import typing

from takco.linkedstring import LinkedString

try:
    import pandas as pd
except:
    log.error(f"Cannot import pandas")


@pd.api.extensions.register_dataframe_accessor("takco")
class TakcoAccessor:
    def __init__(self, df):
        self._df = df
        self.provenance = {}

    @staticmethod
    def try_html(obj):
        return obj._repr_html_() if hasattr(obj, "_repr_html_") else obj

    @property
    def header(self):
        return tuple(map(tuple, self._df.columns.to_frame().values.T))

    @classmethod
    def from_header(cls, head):
        columns = pd.MultiIndex.from_frame(pd.DataFrame(head).T)
        return pd.DataFrame([], columns=columns)

    @property
    def style(self):
        head = self._df.columns.to_frame().applymap(self.try_html)
        head.insert(0, -1, range(len(self._df.columns)))
        return pd.DataFrame(
            self._df.applymap(self.try_html).values,
            columns=pd.MultiIndex.from_arrays(head.values.T),
        ).style.set_table_styles(
            [{"selector": f"thead tr:first-child", "props": [("display", "none")]}]
        )

    def highlight_cells(self, body=(), head=(), color=None, props=()):
        props = props or [("background-color", color or "#ff0")]
        st = self.style
        for ci, ri in body:
            st.table_styles.append(
                {"selector": f".data.row{ri}.col{ci}", "props": props}
            )
        for ci, li in head:
            st.table_styles.append(
                {"selector": f".col_heading.level{li+1}.col{ci}", "props": props}
            )
        return st

    def highlight_pivot(self, level, colfrom, colto, color=None, props=(), **kwargs):
        head = [(c, level) for c in range(colfrom, colto + 1)]
        return self.highlight_cells(head=head, color=color, props=props)

    def to_html(self):
        return self.style.render()

    def _repr_html_(self):
        return self.to_html()


TABEL_PROBLEMS = [
    "Error: This is not a valid number. Please refer to the documentation at for correct input.",
    "[[|]]",
]


def get_tabel_rows(matrix):
    urltemplate = "http://{lang}.wikipedia.org/wiki/{page}"
    newmatrix = []
    for row in matrix:
        newrow = []
        for cell in row:
            text = cell.get("text", "")
            for p in TABEL_PROBLEMS:
                text = text.replace(p, "")

            links = []
            for link in cell.get("surfaceLinks", []):
                target = link.get("target")
                if not target:
                    continue
                start = link.get("offset", 0)
                end = link.get("endOffset", len(text))
                if link.get("linkType") in ["INTERNAL", "INTERNAL_RED"]:
                    try:
                        lang = target.get("language", "en")
                        page = target.setdefault("href", target.get("title", ""))
                        page = page.replace(" ", "_")
                        url = urltemplate.format(lang=lang, page=page)
                        if target.get("id", 0) > 0:
                            url += "?curid=" + str(target.get("id"))
                    except:
                        raise Exception(f"bad target {target}")
                    if start == 0 and end == 1:
                        end = len(text)  # HACK
                    links.append((start, end, url))
            newrow.append(LinkedString(text, links))
        newmatrix.append(newrow)
    return newmatrix


def to_tabel_rows(matrix):
    return [
        [
            {
                "text": getattr(c, "text", str(c)),
                "surfaceLinks": [
                    {
                        "offset": start,
                        "endOffset": end,
                        "linkType": "INTERNAL",
                        "target": {"url": url, "title": url.split("/")[-1],},
                    }
                    for start, end, url in getattr(c, "links", [])
                ],
            }
            for c in row
        ]
        for row in matrix
    ]


def from_tabel(obj):
    body = get_tabel_rows(obj.get("tableData", []))
    head = get_tabel_rows(obj.get("tableHeaders", []))
    try:
        df = pd.DataFrame(body, columns=head or None)
    except:
        df = pd.DataFrame(body)
    provenance = {}
    for key in ["tableCaption", "sectionTitle", "pgTitle", "tableId", "pgId"]:
        provenance[key] = obj.get(key)
    df.attrs["provenance"] = provenance
    return df


class Table(dict):
    """ A takco table object

    >>> Table(head=[['foo','bar']], body=[['1','2']]).head
    (('foo', 'bar'),)

    """

    _id: str
    head: typing.Tuple[typing.Tuple[str, ...], ...]
    body: typing.Tuple[typing.Tuple[str, ...], ...]
    provenance: typing.Dict[str, typing.Any]
    annotations: typing.Dict[str, typing.Any]
    headerId: int

    _old_keys = {
        "_id": lambda self: self._id,
        "tableData": lambda self: to_tabel_rows(self.body),
        "tableHeaders": lambda self: to_tabel_rows(self.head),
        "headerId": lambda self: self.headerId,
        "numCols": lambda self: len(next(iter(self.body), ())),
        "numDataRows": lambda self: len(self.body),
        "numHeaderRows": lambda self: len(self.head),
        "numericColumns": lambda self: [],
    }
    _default_annotations = ["entities", "properties", "classes"]

    def __init__(
        self, obj=None, _id=None, head=(), body=(), provenance=(), annotations=()
    ):
        if isinstance(obj, Table):
            _id = obj._id
            head, body = obj.head, obj.body
            provenance = dict(obj.provenance)
            annotations = dict(obj.annotations)
            for key in self._default_annotations:
                if key in obj:
                    annotations[key] = obj.get(key)

            for key in list(obj.keys()):
                if (key not in self._old_keys) and (
                    key not in self._default_annotations
                ):
                    provenance[key] = obj.get(key)
        elif obj is not None:
            _id = obj.get("_id")
            body = get_tabel_rows(obj.get("tableData", []))
            head = get_tabel_rows(obj.get("tableHeaders", []))

            annotations = {}
            for key in self._default_annotations:
                if key in obj:
                    annotations[key] = obj.get(key)

            provenance = {}
            for key in list(obj.keys()):
                if (key not in self._old_keys) and (
                    key not in self._default_annotations
                ):
                    provenance[key] = obj.get(key)

        self.head, self.body = tuple(map(tuple, head)), tuple(map(tuple, body))
        self._id = _id or str(hash(self.head + self.body))

        self.provenance = dict(provenance)
        self.annotations = dict(annotations)
        self.headerId = self.get_headerId(self.head)

    @staticmethod
    def get_headerId(header):
        import hashlib

        # header is a tuple of tuples.
        header = tuple(map(tuple, header))
        h = hashlib.sha224(str(header).encode()).hexdigest()
        return int(h[:16], 16) // 2  # integer between 0 and SQLite MAX_VAL

    def to_dict(self):
        return {
            **{k: self[k] for k in self._old_keys},
            **self.provenance,
            **self.annotations,
        }

    def __repr__(self):
        return self.df.head(1).__repr__()

    def _repr_html_(self):
        return self.df.takco._repr_html_()

    def __bool__(self):
        return bool(self.body)

    @property
    def df(self):
        columns = pd.MultiIndex.from_arrays(self.head) if len(self.head) else None
        return pd.DataFrame(self.body, columns=columns)

    def __getitem__(self, k):
        if k in self._old_keys:
            return self._old_keys[k](self)
        if k in self.provenance:
            return self.provenance[k]
        if k in self.annotations:
            return self.annotations[k]
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __contains__(self, k):
        if k in list(self._old_keys) + list(self.provenance) + list(self.annotations):
            return True
        return k in self.keys()

    def append(self, other: "Table"):
        assert self.head == other.head
        import copy

        # Merge annotations
        row_offset = len(self.body)
        annotations = copy.deepcopy(self.annotations)
        if "entities" in other.annotations:
            for ci, ri_ents in other.annotations["entities"].items():
                annotations.setdefault("entities", {}).setdefault(ci, {}).update(
                    {str(int(ri) + row_offset): es for ri, es in ri_ents.items()}
                )
        for ci, classes in other.annotations.get("classes", {}).items():
            annotations.setdefault("classes", {}).setdefault(ci, {}).update(classes)
        for fromci, toci_props in other.annotations.get("properties", {}).items():
            newprops = annotations.setdefault("properties", {}).setdefault(fromci, {})
            for toci, props in toci_props:
                newprops.setdefault(toci, {}).update(props)

        # Make provenance
        provenance = {}
        if "concat" in self.provenance:
            provenance["concat"] = self.provenance["concat"] + [other.provenance]
        else:
            provenance["concat"] = [self.provenance, other.provenance]

        return Table(
            head=self.head,
            body=self.body + other.body,
            annotations=annotations,
            provenance=provenance,
        )
