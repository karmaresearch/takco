import pandas as pd


@pd.api.extensions.register_dataframe_accessor("takco")
class TakcoAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.links = pd.DataFrame()
        self.classes = pd.Series(dtype="O")
        self.properties = pd.DataFrame()

    @staticmethod
    def _validate(obj):
        pass

    @classmethod
    def from_obj(cls, obj):
        obj = dict(obj)
        df = pd.DataFrame()
        for ci, col in enumerate(zip(*obj.pop("tableData"))):
            df[ci] = [c.get("text", "") for c in col]
            df.takco.links[ci] = [c.get("surfaceLinks", "") for c in col]

        df.takco.header = pd.DataFrame()
        for ci, col in enumerate(zip(*obj.pop("tableHeaders"))):
            df.takco.header[ci] = [c.get("text", "") for c in col]
            df.takco.header.takco.links[ci] = [c.get("surfaceLinks", "") for c in col]

        df.columns = pd.MultiIndex.from_frame(df.takco.header.T)
        df.attrs.update(obj)
        return df

    @property
    def entities(self):
        return self.links.to_dict()

class Table(dict):
    
    @classmethod
    def minimize(cls, obj):
        def minlink(ls):
            if ls:
                for l in ls:
                    for f in ['surface', 'inTemplate', 'isInTemplate', 'locType']:
                        if f in l:
                            l.pop(f)
            return ls

        if isinstance(obj, dict) and 'tableData' in obj:
            obj['links'] = [
                [minlink(c.pop('surfaceLinks', None) or None) for c in row]
                for row in obj['tableData']
            ]
            obj['rows'] = [[c.get('text', '') for c in row] for row in obj.pop('tableData')]
        return obj

    def __init__(self, obj):
        self.minimize(obj)
        self.update(obj)


    def __getitem__(self, k):
        if k == 'tableData':
            rows = dict.__getitem__(self, 'rows')
            links = dict.__getitem__(self, 'links')
            return [
                [
                    {'text':c, 'surfaceLinks': lc} if lc else {'text':c}
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
        if k == 'tableData':
            return True
        return dict.__contains__(self, k)

    def keys(self):
        return dict.keys(self)

    