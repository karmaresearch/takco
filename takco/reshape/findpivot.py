from typing import (
    List,
    Container,
    Tuple,
    NamedTuple,
    Iterator,
    Dict,
    Optional,
    Set,
    Collection,
    Pattern,
)
from collections import defaultdict, Counter
import re
import logging as log
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .. import link

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
            else:
                for j in range(1, span + 1):
                    colspan[ci - j] = span
                span = 1
                repeats[c] = repeats.get(c, 0) + 1
            c = cell
        header_colspan.append(colspan)
        header_repeats.append([repeats.get(str(cell), 0) for cell in row])
    return header_colspan, header_repeats


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


def longest_seq(numbers: Collection[int]) -> List[int]:
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
        if i + 1 in numbers:
            i += 1
        else:
            if len(seq) > len(longest):
                longest, seq = seq, []
            i = min(numbers) if numbers else None
    return longest


class Pivot(NamedTuple):
    """The sequence of cells in a header row that looks more like it should be a column"""

    level: int  #: Header row index
    colfrom: int  #: Leftmost column index
    colto: int  #: Rightmost column index

@dataclass
class PivotFinder():
    min_len: int = 1

    def build(self, tables):
        return self

    def merge(self, heuristic):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @abstractmethod
    def find_pivot_cells(
        self, headerrows: List[List[str]]
    ) -> Iterator[Tuple[int, int]]:
        """Yield positions of pivoted cells"""
        pass

    def find_longest_pivots(self, headerrows: List[List[Dict]]) -> Iterator[Pivot]:
        """Yield longest pivots"""
        row_cols = defaultdict(set)
        for ri, ci in self.find_pivot_cells(headerrows):
            row_cols[ri].add(ci)

        for level, cols in row_cols.items():
            pivot_cols = longest_seq(cols)
            colfrom, colto = pivot_cols[0], pivot_cols[-1]
            if colfrom <= colto and (colto-colfrom >= self.min_len-1):
                yield Pivot(level, colfrom, colto)

    def split_header(self, headrow: List[str], colfrom: int, colto: int):
        """Split the header containing the pivot"""
        return ()
    
def find_longest_pivot(headertext, heuristics):
    pivot_size = Counter()
    for h in heuristics:
        for level, colfrom, colto in h.find_longest_pivots(headertext):
            pivot_size[(level, colfrom, colto, h.name)] = colto - colfrom

    # Get longest pivot
    for (level, colfrom, colto, hname), _ in pivot_size.most_common(1):
        log.debug(f"Found pivot {(level, colfrom, colto)} using {hname}")
        return (level, colfrom, colto, hname)
        

@dataclass
class RegexFinder(PivotFinder):
    """Find pivots based on a regex
    
    Args:
        find_regex: Unpivot cell if this regex matches
        split_regex: Regex that returns groupdict with ``cell`` and ``head``
    """
    name: str = 'RegexFinder'
    find_regex: Optional[Pattern] = None
    split_regex: Optional[Pattern] = None

    def __post_init__(self):
        if self.find_regex:
            self.find_regex = re.compile(str(self.find_regex))
        if self.split_regex:
            self.split_regex = re.compile(str(self.split_regex))

    def find_pivot_cells(self, headerrows):
        for ri, hrow in enumerate(headerrows):
            for ci, cell in enumerate(hrow):
                if cell and self.find_regex and self.find_regex.match(cell.strip()):
                    yield ri, ci

    def split_header(self, headrow, colfrom, colto):
        for ci, cell in enumerate(headrow):
            if self.split_regex and (ci in range(colfrom, colto + 1)):
                m = self.split_regex.match(cell.strip())
                if m:
                    cell = m.groupdict().get("cell", cell)
                    head = m.groupdict().get("head", "")
                    yield head, cell
                else:
                    yield cell, cell
            else:
                yield cell, cell

@dataclass
class NumSuffix(RegexFinder):
    """Find cells with a numeric suffix"""
    name: str = 'NumSuffix'
    
    def __post_init__(self):
        self.find_regex = re.compile(r".*(^|\s)\d+[\W\s]*$")
        self.split_regex = re.compile(r"(?P<head>.*?)[\W\s]*(?P<cell>[\d\W]+?)[\W\s]*$")

@dataclass
class NumPrefix(RegexFinder):
    """Find cells with a numeric prefix"""
    name: str = 'NumPrefix'
    
    def __post_init__(self):
        self.find_regex = re.compile(r"[\W\s]*\d+($|\s)")
        self.split_regex = re.compile(r"(?P<cell>[\d\W]+?)[\W\s]*(?P<head>.*?)[\W\s]*$")

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

    def split_header(self, headrow, colfrom, colto):
        prefixes = []
        for cell in headrow:
            p = (cell or "").strip().split()
            if p:
                prefixes.append(p[0])

        for p, pcount in Counter(prefixes).most_common(1):
            for ci in range(colfrom, colto + 1):
                cell = headrow[ci].strip()
                if cell.startswith(p):
                    yield cell[len(p) :].strip(), p

@dataclass
class SpannedRepeat(PivotFinder):
    """Find cells that span repeating cells"""
    name: str = 'SpannedRepeat'

    def find_pivot_cells(self, headerrows):
        header_colspan, header_repeats = get_colspan_repeats(headerrows)
        header_fromto = get_colspan_fromto(headerrows)
        for ri, row in enumerate(headerrows):
            colspan, repeats, fromto = (
                header_colspan[ri],
                header_repeats[ri],
                header_fromto[ri],
            )
            cols = []
            for ci, cell in enumerate(row):
                f, t = fromto[ci]
                if colspan[ci] > 1:
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
    - Don't have bad types (e.g. disambiguation pages, list pages, units)
    - Don't have bad properties (e.g. has associated property)
    
    Args:
        lookup_config: Entity lookup object
        kb_config: Knowledge base
        bad_types: URIs for bad types
        bad_props: URIs for bad props
    """
    name: str = 'AgentLikeHyperlink'
    lookup: Optional[link.Lookup] = None
    kb: Optional[link.GraphDB] = None
    bad_types: List[str] = field(default_factory=list)
    bad_props: List[str] = field(default_factory=list)
    type_props: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert self.lookup is not None and self.kb is not None
        self.bad_types = [link.URIRef(b) for b in self.bad_types]
        self.bad_props = [link.URIRef(b) for b in self.bad_props]
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

    def find_pivot_cells(self, headerrows):
        
        assert self.lookup is not None and self.kb is not None
        kb = self.kb
        type_props = self.typeProperties
        ents = self.lookup.lookup_cells(link.get_hrefs(headerrows))

        for ri, hrow in enumerate(headerrows):
            for ci, _ in enumerate(hrow):
                for e in ents.get(str(ci), {}).get(str(ri), {}):
                    e = link.URIRef(e)

                    if any(kb.count([None, tp, e]) for tp in type_props):
                        continue  # is type

                    if any(
                        t in set(kb.get_prop_values(e, tp))
                        for tp in type_props
                        for t in self.bad_types
                    ):
                        continue  # has bad type

                    if any(kb.count([e, p, None]) for p in self.bad_props):
                        continue  # has bad prop

                    yield ri, ci


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
                    for celltext in hrow:
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
