from typing import List, Container, Tuple, NamedTuple, Iterator
from collections import defaultdict, Counter
import re


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
        repeats = {}
        c, span = None, 1
        for ci, cell in enumerate(list(row) + [None]):
            cell = str(cell)
            if cell == c:
                span += 1
            else:
                for j in range(1, span + 1):
                    colspan[ci - j] = span
                span = 1
                repeats[c] = repeats.get(c, 0) + 1
            c = cell
        repeats = [repeats.get(str(cell), 0) for cell in row]
        header_colspan.append(colspan)
        header_repeats.append(repeats)
    return header_colspan, header_repeats


def get_colspan_fromto(rows: List[List[str]]) -> List[List[Tuple[int, int]]]:
    """Gets colspan (from,to) index of every cell"""
    fromto = []
    for row in rows:
        fr, to = [], []
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


def longest_seq(numbers: Container[int]) -> Container[int]:
    """Find the longest sequence in a set of numbers"""
    if not numbers:
        return []
    numbers = set(numbers)
    i = min(numbers)
    longest, seq = [], []
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


class PivotFinder:
    def find_pivot_cells(
        self, headerrows: List[List[str]]
    ) -> Iterator[Tuple[int, int]]:
        """Yield positions of pivoted cells"""
        return

    def find_longest_pivots(self, headerrows: List[List[str]]) -> Iterator[Pivot]:
        """Yield longest pivots"""
        row_cols = defaultdict(set)
        for ri, ci in self.find_pivot_cells(headerrows):
            row_cols[ri].add(ci)

        for level, cols in row_cols.items():
            pivot_cols = longest_seq(cols)
            colfrom, colto = pivot_cols[0], pivot_cols[-1]
            if colfrom <= colto:
                yield Pivot(level, colfrom, colto)

    def split_header(self, headrow: List[str], colfrom: int, colto: int):
        """Split the header containing the pivot"""
        return


class RegexFinder(PivotFinder):
    def __init__(self, find_regex, split_regex):
        self.find_regex = re.compile(find_regex)
        self.split_regex = re.compile(split_regex)

    def find_pivot_cells(self, headerrows):
        for ri, hrow in enumerate(headerrows):
            for ci, cell in enumerate(hrow):
                if cell and self.find_regex.match(cell.strip()):
                    yield ri, ci

    def split_header(self, headrow, colfrom, colto):
        for ci, cell in enumerate(headrow):
            if ci in range(colfrom, colto + 1):
                m = self.split_regex.match(cell.strip())
                if m:
                    cell = m.groupdict().get("cell", cell)
                    head = m.groupdict().get("head", "")
                    yield head, cell
                else:
                    yield cell, cell
            else:
                yield cell, cell


class NumSuffix(RegexFinder):
    """Find cells with a numeric suffix"""

    def __init__(self):
        self.find_regex = re.compile(".*\d[\W\s]*$")
        self.split_regex = re.compile("(?P<head>.*?)[\W\s]*(?P<cell>[\d\W]+?)[\W\s]*$")


class NumPrefix(RegexFinder):
    """Find cells with a numeric prefix"""

    def __init__(self):
        self.find_regex = re.compile("[\W\s]*\d")
        self.split_regex = re.compile("(?P<cell>[\d\W]+?)[\W\s]*(?P<head>.*?)[\W\s]*$")


class SeqPrefix(PivotFinder):
    """Find cells with a shared prefix"""

    def find_pivot_cells(self, headerrows):
        from collections import Counter

        for ri, row in enumerate(headerrows):
            prefixes = []
            for cell in row:
                p = (cell or "").strip().split()[0]
                if p:
                    prefixes.append(p)

            for p, pcount in Counter(prefixes).most_common(1):
                if pcount > 1:
                    for ri, cell in enumerate(row):
                        if str(cell or "").startswith(p):
                            yield ri, ci

    def split_header(headrow, colfrom, colto):
        prefixes = []
        for cell in row:
            p = (cell or "").strip().split()[0]
            if p:
                prefixes.append(p)

        for p, pcount in Counter(prefixes).most_common(1):
            for ci in range(colfrom, colto + 1):
                cell = headrow[ci].strip()
                if cell.startswith(p):
                    yield ci, cell[len(p) :].strip(), p


class SpannedRepeat(PivotFinder):
    """Find cells that span repeating cells"""

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
