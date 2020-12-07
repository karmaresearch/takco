import pandas as pd
from takco.linkedstring import LinkedString
import typing

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
def get_tabel_rows(matrix):
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
                    target.setdefault('language', 'en')
                    url = urltemplate.format(**target)
                    if target.get('id', 0) > 0:
                        url += '?curid=' + str(target.get('id'))
                    if start == 0 and end == 1:
                        end = len(text)  # HACK
                    links.append((start, end, url))
            newrow.append( LinkedString(text, links) )
        newmatrix.append(newrow)
    return newmatrix

def to_tabel_rows(matrix):
    return [
            [
                {
                    "text": getattr(c, "text", str(c)),
                    "surfaceLinks": [
                        {
                            'offset': start,
                            'endOffset': end,
                            'target': {
                                'url': url,
                            }
                        }
                        for start, end, url in getattr(c, "links", [])
                    ]
                }
                for c in row
            ]
            for row in matrix
        ]

def from_tabel(obj):
    body = get_tabel_rows(obj.get('tableData', []))
    head = get_tabel_rows(obj.get('tableHeaders', []))
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
    head: typing.Collection[typing.Collection[str]]
    body: typing.Collection[typing.Collection[str]]
    provenance: typing.Dict[str, str]

    def __init__(self, obj=None, head=(), body=(), provenance=()):
        if obj is not None and not isinstance(obj, Table):
            body = get_tabel_rows(obj.get('tableData', []))
            head = get_tabel_rows(obj.get('tableHeaders', []))
            provenance = {}
            for key in ['tableCaption', 'sectionTitle', 'pgTitle', 'tableId', 'pgId']:
                provenance[key] = obj.get(key)
        self.head, self.body = head, body
        self.provenance = dict(provenance)

    def to_dict(self):
        return {
            "tableData": self["tableData"],
            "tableHeaders": self["tableHeaders"],
            **self.provenance
        }
    
    def __repr__(self):
        return self.to_dict().__repr__()

    @property
    def df(self):
        return pd.DataFrame(self.body, columns=pd.MultiIndex.from_arrays(self.head))
        

    def __getitem__(self, k):
        if k == "tableData":
            return to_tabel_rows(self.body)
        if k == "tableHeaders":
            return to_tabel_rows(self.head)
        if k in self.provenance:
            return self.provenance[k]
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def __contains__(self, k):
        if k in ["tableData", "tableHeaders"] + list(self.provenance):
            return True
        return False

    def unpivot(self, level, colfrom, colto, var_name='variable', value_name='value'):
        df = self.df
        nhead = df.columns.nlevels
        
        colrange = range(colfrom, colto + 1)
        id_cols = [df.columns[i] for i in range(len(df.columns)) if i not in colrange]
        value_cols = [df.columns[i] for i in colrange]
        df = df[value_cols].set_index(pd.MultiIndex.from_frame(df[id_cols]))
        df.index.names = [LinkedString(' ').join(set(hs)) for hs in df.index.names]
        
        
        if nhead > 1:
            # For tables with multiple header rows, the right columns get their own headers
            df = df.stack(level)
            df.index.names = df.index.names[:-1] + [var_name]
            df = df.reset_index()
        else:
            # For tables with a single header row, the right column needs to be given
            df.columns = [c[0] for c in df.columns]
            df = df.stack()
            df.index.names = df.index.names[:-1] + [(var_name,)]
            df = df.to_frame((value_name,)).reset_index()
        
        head = df.columns.to_frame().T.values
        body = df.values
        return Table(head=head, body=body, provenance=self.provenance)

