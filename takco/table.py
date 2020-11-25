import pandas as pd 


@pd.api.extensions.register_dataframe_accessor("takco")
class TakcoAccessor:
    def __init__(self, df):
        self._df = df

    @property
    def entities(self):
        pass

    @staticmethod
    def try_html(obj):
        return obj._repr_html_() if hasattr(obj, "_repr_html_") else obj

    def to_html(self):
        return self._repr_html_().to_html()

    def _repr_html_(self):
        return self._df.style.format(self.try_html)

    def show(self):
        return self._repr_html_()


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
