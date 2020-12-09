from typing import (
    List,
    Tuple,
    NamedTuple,
    Iterable,
    Dict,
    Optional,
    Set,
    Collection,
    Pattern,
)
from collections import defaultdict, Counter
import re
import logging as log
import contextlib
from dataclasses import dataclass, field

from takco import Table, link

def table_get_headerId(table: Table):
    return Table(table).headerId

def get_header(table1, table2):
    return Table(table1).head

class Pivot(NamedTuple):
    """The sequence of cells in a header row that looks more like it should be a column"""

    level: int  #: Header row index
    colfrom: int  #: Leftmost column index
    colto: int  #: Rightmost column index
    source: str #: Provenance

@dataclass
class PivotFinder:
    name: str = 'PivotFinder'
    min_len: int = 1
    allow_gap: int = 0

    def build(self, tables):
        return self

    def merge(self, heuristic):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def find_pivot_cells(
        self, headerrows: List[List[str]]
    ) -> Iterable[Tuple[int, int]]:
        """Yield positions of pivoted cells"""
        pass
    
    @staticmethod
    def longest_seq(numbers: Collection[int], allow_gap: int = 0) -> List[int]:
        """Find the longest sequence in a set of numbers"""
        if not numbers:
            return []
        numbers = set(numbers)
        i = min(numbers)
        longest: List[int] = []
        seq: List[int] = []
        while numbers:
            seq.append(i)
            numbers -= set([i])
            j = next((j for j in range(1, 2 + allow_gap) if i + j in numbers), None)
            if j is not None:
                i += j
            elif len(seq) > len(longest):
                longest, seq = seq, []
                i = min(numbers) if numbers else None
        return longest
    
    def find_longest_pivots(self, headerrows: List[List[str]]) -> Iterable[Pivot]:
        """Yield longest pivots"""
        row_cols = defaultdict(set)
        for ri, ci in self.find_pivot_cells(headerrows):
            row_cols[ri].add(ci)

        for level, cols in row_cols.items():
            pivot_cols = self.longest_seq(cols, allow_gap=self.allow_gap)

            colfrom, colto = pivot_cols[0], pivot_cols[-1]
            if colfrom <= colto and (colto-colfrom >= self.min_len-1):
                yield Pivot(level, colfrom, colto, self.name)

    def split_header(self, headrows: List[List[str]], level: int, colfrom: int, colto: int):
        """Split the header containing the pivot"""
        return headrows

    def unpivot(self, head, body, pivot, var_name: str='_Variable', value_name: str='_Value'):
        """Unpivot a table

        Args:
            head: Header
            body: Table body
            pivot: Pivot tuple
            var_name: Name to use for the ‘variable’ column. Defaults to '_Variable'.
            value_name: Name to use for the ‘value’ column.. Defaults to '_Value'.

        Raises:
            UnpivotException: Unable to unpivot

        Returns:
            Tuple of new head and body
        """
        import pandas as pd

        # Split header
        head = self.split_header(head, pivot.level, pivot.colfrom, pivot.colto)

        # Make dataframe
        df = pd.DataFrame(body, columns=pd.MultiIndex.from_arrays(head))
        level = pivot.level
        colrange = range(pivot.colfrom, pivot.colto + 1)

        if level >= len(head):
            raise UnpivotException(f"Unpivot level is too big: ({level}, {head})")

        if (pivot.colfrom == 0) and (pivot.colto == len(head[0]) - 1):
            raise UnpivotException(f"Pivot spans entire head: ({colrange}, {head})")

        # Set dataframe index to id columns
        nhead = df.columns.nlevels
        id_cols = [df.columns[i] for i in range(len(df.columns)) if i not in colrange]
        value_cols = [df.columns[i] for i in colrange]
        df = df[value_cols].set_index(pd.MultiIndex.from_frame(df[id_cols]))

        def merge_head_cells(hs):
            cls = hs[0].__class__
            return cls(' ').join(set(h for h in hs if h))
        df.index.names = [merge_head_cells(hs) for hs in df.index.names]
        
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
        
        return df.columns.to_frame().T.values, df.fillna('').values

class UnpivotException(Exception):
    pass

def yield_pivots(headertexts: Iterable[Collection[Collection[str]]], heuristics: List[PivotFinder]):
    """Detect headers that should be unpivoted using heuristics."""
    import copy

    with contextlib.ExitStack() as hstack:
        heuristics = [hstack.enter_context(copy.deepcopy(h)) for h in heuristics]
        named_heuristics = {h.name: h for h in heuristics}
        for headertext in headertexts:
            if headertext:
                pivot = find_longest_pivot(headertext, heuristics)
                if pivot is not None:
                    try:
                        dummy = [[str(ci) for ci in range(len(next(iter(headertext))))]]
                        heuristic = named_heuristics[pivot.source]
                        heuristic.unpivot(headertext, dummy, pivot)
                        yield Table.get_headerId(headertext), pivot
                    except Exception as e:
                        log.debug(f"Failed to unpivot header {headertext} with {pivot.source} due to {e}")

def try_unpivot(table, pivot, named_heuristics):
    try:
        pivotmeta = {
            "headerId": table.headerId,
            "level": pivot.level,
            "colfrom": pivot.colfrom,
            "colto": pivot.colto,
            "heuristic": pivot.source
        }
        heuristic = named_heuristics[pivot.source]
        head, body = heuristic.unpivot(table.head, table.body, pivot)
        return Table(
            head=head,
            body=body,
            provenance = {**table.provenance, 'pivot': pivotmeta}
        )
    except Exception as e:
        log.debug(f"Cannot pivot table {table.get('_id')} due to {e}")


def build_heuristics(
    tables: Iterable[Table], heuristics: List[PivotFinder],
):
    for heuristic in heuristics:
        yield heuristic.build(tables)


def unpivot_tables(
    tables: Iterable[Dict],
    headerId_pivot: Optional[Dict[int, Pivot]],
    heuristics: List[PivotFinder],
):
    """Unpivot tables."""
    tablelist = [Table(t) for t in tables]

    if headerId_pivot is None:
        headertexts = [table.head for table in tablelist]
        headerId_pivot = dict(yield_pivots(headertexts, heuristics=heuristics))
    log.debug(f"Using {len(headerId_pivot)} detected pivots")

    named_heuristics = {h.name: h for h in heuristics}
    for table in tablelist:
        pivot = headerId_pivot.get(table.headerId)
        if pivot and table.head:
            table = try_unpivot(table, pivot, named_heuristics)

        if table is not None:
            yield table

def get_colspan_fromto(rows: List[List[str]]) -> List[List[Tuple[int, int]]]:
    """Gets colspan (from,to) index of every cell"""
    fromto = []
    for row in rows:
        fr, to = [], [] # type: ignore
        _cell = None
        for ci, cell in enumerate(row):
            if fr and cell == _cell:
                fr.append(fr[ci - 1])
                to.append(ci)
                for cj, t in enumerate(to):
                    if t == ci - 1:
                        to[cj] = ci
            else:
                fr.append(ci)
                to.append(ci)
            _cell = cell
        fromto.append(list(zip(fr, to)))
    return fromto

def find_longest_pivot(headertext, heuristics):
    pivot_size: Counter = Counter()
    for h in heuristics:
        for p in h.find_longest_pivots(headertext):
            pivot_size[p] = p.colto - p.colfrom

    # Get longest pivot
    for p, _ in pivot_size.most_common(1):
        
        # If spanned, lengthen pivot to spanning cell width
        if p.colfrom != p.colto and p.level > 0 and len(headertext) > 1:
            colspans = get_colspan_fromto(headertext)
            spanning = set(colspans[0][ci] for ci in range(p.colfrom, p.colto+1))
            if len(spanning) == 1:
                colfrom, colto = list(spanning)[0]
                p = Pivot(p.level, colfrom, colto, p.source)

        log.debug(f"Found pivot {p}")
        return p




@dataclass
class RegexFinder(PivotFinder):
    """Find pivots based on a regex

    >>> h = RegexFinder(pattern=r'(?P<var>.*?)\s*(?P<val>\d+)')
    >>> list(h.find_pivot_cells([['a 4']]))
    [(0, 0)]

    >>> h.split_header([['a 4']], 0, 0, 0)
    [('a',), ('4',)]
    
    Args:
        pattern: Unpivot cell if this regex matches. 
            Optional: group for extracting value
    """
    name: str = 'RegexFinder'
    pattern: Pattern = re.compile('.*')

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def find_pivot_cells(self, headerrows):
        for ri, hrow in enumerate(headerrows):
            for ci, cell in enumerate(hrow):
                if cell and self.pattern.match(cell.strip()):
                    yield ri, ci

    def split_header(self, headrows, level, colfrom, colto):
        splitrow = []
        for ci, cell in enumerate(headrows[level]):
            if ci in range(colfrom, colto + 1):
                m = self.pattern.match(cell)
                if m:
                    if m.groups() and set(['val','var']) == set(m.groupdict()):
                        val_start, val_end = m.span('val')
                        val = cell[val_start:val_end]
                        var_start, var_end = m.span('var')
                        var = cell[var_start:var_end]
                        splitrow.append( (val, var) )
                        continue
            splitrow.append( (cell, cell) )
        
        splitrow1, splitrow2 = zip(*splitrow)
        if splitrow1 != splitrow2:
            return tuple(headrows[:0]) + (splitrow1, splitrow2) + tuple(headrows[1:])
        else:
            return headrows

@dataclass
class NumSuffix(RegexFinder):
    """Find cells with a numeric suffix"""
    name: str = 'NumSuffix'
    pattern: Pattern = re.compile(r"(?P<var>.*)(?:^|\s)[\W\s]*(?P<val>\d[\W\d]*?)[\W\s]*$")

@dataclass
class NumPrefix(RegexFinder):
    """Find cells with a numeric prefix"""
    name: str = 'NumPrefix'
    pattern: Pattern = re.compile(r"[\W\s]*(?P<val>\d[\W\d]*)(?:$|\s)(?P<var>.*)")

@dataclass
class SeqPrefix(PivotFinder):
    """Find cells with a shared prefix"""
    name: str = 'SeqPrefix'
    

    def find_pivot_cells(self, headerrows):
        from collections import Counter

        for ri, row in enumerate(headerrows):
            prefixes = []
            for cell in row:
                p = (cell or "").strip().split()
                if p:
                    prefixes.append(p[0])

            for p, pcount in Counter(prefixes).most_common(1):
                if pcount > 1:
                    for ci, cell in enumerate(row):
                        if str(cell or "").startswith(p) and str(cell) != str(p):
                            yield ri, ci

    def split_header(self, headrows, level, colfrom, colto):
        prefixes = []
        for cell in headrows[level]:
            p = (cell or "").strip().split()
            if p:
                prefixes.append(p[0])

        for p, _ in Counter(prefixes).most_common(1):
            splitrow = []
            for ci, cell in enumerate(headrows[level]):
                if ci in range(colfrom, colto + 1):
                    cell = cell.strip()
                    if cell.startswith(p):
                        splitrow.append( (cell[len(p) :].strip(), p) )
                        continue
                splitrow.append( (cell, cell) )
            splitrow1, splitrow2 = zip(*splitrow)
            if splitrow1 != splitrow2:
                return headrows[:0] + [splitrow1, splitrow2] + headrows[1:]
        
        return headrows

@dataclass
class SpannedRepeat(PivotFinder):
    """Find cells that span repeating cells"""
    name: str = 'SpannedRepeat'

    @staticmethod
    def get_colspan_repeats(
        rows: List[List[str]],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Find cells that span multiple columns and cells that repeat

        Args:
            rows: A matrix of cells

        Returns:
            - A matrix of colspans
            - A matrix of repeats (excluding colspans)
        """
        header_colspan, header_repeats = [], []
        for row in rows:
            colspan = [1 for _ in row]
            repeats: Dict[Optional[str], int] = {}
            c, span = None, 1
            for ci, cell in enumerate(list(row) + [None]):  # type: ignore
                cell = str(cell)
                if cell == c:
                    span += 1
                elif c:
                    for j in range(1, span + 1):
                        colspan[ci - j] = span
                    span = 1
                    repeats[c] = repeats.get(c, 0) + 1
                c = cell
            header_colspan.append(colspan)
            header_repeats.append([repeats.get(str(cell), 0) for cell in row])
        return header_colspan, header_repeats

    def find_pivot_cells(self, headerrows):
        header_colspan, header_repeats = self.get_colspan_repeats(headerrows)
        header_fromto = get_colspan_fromto(headerrows)
        for ri, row in enumerate(headerrows):
            colspan = header_colspan[ri]
            fromto = header_fromto[ri]
            for ci, cell in enumerate(row):
                f, t = fromto[ci]
                if cell and (colspan[ci] > 1):
                    # This cell is spanning
                    for rj in range(len(headerrows)):
                        for cspan in range(f, t + 1):
                            if ri != rj:
                                if header_repeats[rj][cspan] > 1:
                                    # There's a repeating cell in another row
                                    yield ri, ci

@dataclass
class AgentLikeHyperlink(PivotFinder):
    """Find cells with links to entities that seem agent-like
    
    Rules for agent-like entities:
    
    - Not used as class
    - Don't have attribute-like types (e.g. disambiguation pages, list pages, units)
    - Don't have attribute-like properties (e.g. has associated property)
    
    Args:
        lookup_config: Entity lookup object
        kb_config: Knowledge base
        id_types: URIs for attribute-like types
        id_props: URIs for attribute-like props
    """
    name: str = 'AgentLikeHyperlink'
    lookup: Optional[link.Lookup] = None
    kb: Optional[link.GraphDB] = None
    id_types: List[str] = field(default_factory=list)
    id_props: List[str] = field(default_factory=list)
    type_props: List[str] = field(default_factory=list)
    lookup_cells: bool = False

    def __post_init__(self):
        assert self.lookup is not None and self.kb is not None
        self.id_types = [link.URIRef(b) for b in self.id_types]
        self.id_props = [link.URIRef(b) for b in self.id_props]
        if hasattr(self.kb, "typeProperties"):
            self.typeProperties = self.kb.typeProperties
        else:
            self.typeProperties = self.type_props

    def __enter__(self):
        assert self.lookup is not None and self.kb is not None
        self.lookup.__enter__()
        self.kb.__enter__()
        return self

    def __exit__(self, *args):
        assert self.lookup is not None and self.kb is not None
        self.lookup.__exit__(*args)
        self.kb.__exit__(*args)

    def is_attribute(self, e):
        assert self.lookup is not None and self.kb is not None
        e = link.URIRef(e)

        if any(self.kb.count([None, tp, e]) for tp in self.typeProperties):
            return True  # is type

        if any(
            t in set(self.kb.get_prop_values(e, tp))
            for tp in self.typeProperties
            for t in self.id_types
        ):
            return True  # has attr type

        if any(self.kb.count([e, p, None]) for p in self.id_props):
            return True  # has attr prop
        
        return False


    def find_pivot_cells(self, headerrows):
        
        assert self.lookup is not None and self.kb is not None
        ents = self.lookup.lookup_cells(link.get_hrefs(headerrows, lookup_cells=self.lookup_cells))

        for ri, hrow in enumerate(headerrows):
            for ci, _ in enumerate(hrow):
                for e in ents.get(str(ci), {}).get(str(ri), {}):
                    if not self.is_attribute(e):
                        yield ri, ci
                        break


@dataclass
class AttributeContext(PivotFinder):
    name: str = 'AttributeContext'
    attname: Optional[str] = None
    values: Set[str] = field(default_factory=set)

    def build(self, tables):
        for t in tables:
            att = str(t.get(self.attname, "")).lower()
            if att:
                for hrow in t.get("tableHeaders"):
                    for cell in hrow:
                        if isinstance(cell, dict):
                            celltext = cell.get('text', '')
                        else:
                            celltext = cell
                            
                        if celltext and len(celltext) > 1 and att == celltext.lower():
                            self.values.add(celltext)
        return self

    def merge(self, heuristic):
        self.values.update(heuristic.values)
        return self

    def find_pivot_cells(self, headerrows):
        for ri, hrow in enumerate(headerrows):
            for ci, celltext in enumerate(hrow):
                if celltext in self.values:
                    yield ri, ci


@dataclass
class Rule(PivotFinder):
    name: str = 'Rule'
    id_vars: List[str] = field(default_factory=list)
    value_vars: List[str] = field(default_factory=list)
    value_name: Optional[str] = None

    def find_pivot_cells(self, headerrows):
        if self.id_vars or self.value_vars:
            for ri, hrow in enumerate(headerrows):
                id_match = all(v in hrow for v in self.id_vars)
                value_match = all(v in hrow for v in self.value_vars)
                if id_match and value_match:
                    for ci, hcell in enumerate(hrow):
                        if self.value_vars:
                            if hcell in self.value_vars:
                                yield ri, ci
                        else:
                            if hcell not in self.id_vars:
                                yield ri, ci
