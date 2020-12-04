import pandas as pd
from takco.linkedstring import LinkedString

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
        return pd.DataFrame([], columns = columns)
    
    @property
    def style(self):
        head = self._df.columns.to_frame().applymap(self.try_html)
        head.insert(0, -1, range(len(self._df.columns)))
        return pd.DataFrame(
            self._df.applymap(self.try_html).values,
            columns=pd.MultiIndex.from_arrays(head.values.T)
        ).style.set_table_styles([
            {'selector': f'.col_heading.level0','props':[('display','none')]}
        ])
    
    def highlight_cells(self, body=(), head=(), color=None, props=()):
        props = props or [('background-color', color or '#ff0')]
        st = self.style
        for ci, ri in body:
            st.table_styles.append({'selector': f'.data.row{ri}.col{ci}','props': props})
        for ci, li in head:
            st.table_styles.append({'selector': f'.col_heading.level{li+1}.col{ci}','props': props})
        return st
    
    def highlight_pivot(self, level, colfrom, colto, color=None, props=(), **kwargs):
        head = [(c, level) for c in range(colfrom, colto+1)]
        return self.highlight_cells(head=head, color=color, props=props)

    def to_html(self):
        return self.style.render()
        
    def _repr_html_(self):
        return self.to_html()

    
TABEL_PROBLEMS = [
    'Error: This is not a valid number. Please refer to the documentation at for correct input.',
    '[[|]]',
]
def get_tabel_linkedstrings(matrix):
    urltemplate = "http://{language}.wikipedia.org/wiki/{title}"
    newmatrix = []
    for row in matrix:
        newrow = []
        for cell in row:
            text = cell.get("text", "")
            for p in TABEL_PROBLEMS:
                text = text.replace(p, '')
            
            links = []
            for link in cell.get('surfaceLinks', []):
                target = link.get('target')
                if not target:
                    continue
                start = link.get('offset', 0)
                end = link.get('endOffset', len(text))
                if link.get('linkType') in ['INTERNAL', 'INTERNAL_RED']:
                    url = urltemplate.format(**target)
                    if target.get('id', 0) > 0:
                        url += '?curid=' + str(target.get('id'))
                    if start == 0 and end == 1:
                        end = len(text)  # HACK
                    links.append((start, end, url))
            newrow.append( LinkedString(text, links) )
        newmatrix.append(newrow)
    return newmatrix

def from_tabel(obj):
    body = get_tabel_linkedstrings(obj.get('tableData', []))
    head = get_tabel_linkedstrings(obj.get('tableHeaders', []))
    try:
        df = pd.DataFrame(body, columns=head or None)
    except:
        df = pd.DataFrame(body)
    provenance = {}
    for key in ['tableCaption', 'sectionTitle', 'pgTitle', 'tableId', 'pgId']:
        provenance[key] = obj.get(key)
    df.attrs['provenance'] = provenance
    return df

class Table(dict):
    @classmethod
    def minimize(cls, obj):
        def minlink(ls):
            if ls:
                for l in ls:
                    for f in ["surface", "inTemplate", "isInTemplate", "locType"]:
                        if f in l:
                            l.pop(f)
            return ls

        if isinstance(obj, dict) and "tableData" in obj:
            obj["links"] = [
                [minlink(c.pop("surfaceLinks", None) or None) for c in row]
                for row in obj["tableData"]
            ]
            obj["rows"] = [
                [c.get("text", "") for c in row] for row in obj.pop("tableData")
            ]
        return obj

    def __init__(self, obj):
        if not isinstance(obj, Table):
            self.minimize(obj)
        self.update(obj)

    def to_dict(self):
        tableData = self["tableData"]
        obj = {**self, "tableData": tableData}
        del obj["rows"]
        del obj["links"]
        return obj

    def __getitem__(self, k):
        if k == "tableData":
            rows = dict.__getitem__(self, "rows")
            links = dict.__getitem__(self, "links")
            return [
                [
                    {"text": c, "surfaceLinks": lc} if lc else {"text": c}
                    for c, lc in zip(row, lrow)
                ]
                for row, lrow in zip(rows, links)
            ]
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        dict.__delitem__(self, k)

    def __contains__(self, k):
        if k == "tableData":
            return True
        return dict.__contains__(self, k)

    def keys(self):
        return dict.keys(self)


from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class TableAnnotation:
    """Table annotation

    >>> TableAnnotation([[{'foo': 0.5}]])
    TableAnnotation(entities=[[Annotation({'foo': 0.5})]], classes=None, properties=None)

    """

    class Annotation(Dict[str, float]):
        def __repr__(self):
            return self.__class__.__name__ + f"({dict(self)})"

    entities: Optional[List[List[Annotation]]] = None
    classes: Optional[List[Annotation]] = None
    properties: Optional[List[List[Annotation]]] = None
    keycol: Optional[int] = None

    def __post_init__(self):
        if self.entities is not None:
            self.entities = [
                [self.Annotation(anno) for anno in col] for col in self.entities
            ]
        if self.classes is not None:
            self.classes = [self.Annotation(anno) for anno in self.classes]
        if self.properties is not None:
            self.properties = [
                [self.Annotation(anno) for anno in col] for col in self.properties
            ]
