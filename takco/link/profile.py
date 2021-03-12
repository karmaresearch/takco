import sys, os, json, itertools
import logging as log
from collections import Counter

from typing import List, Dict, Mapping, Tuple, Iterable

from .tane import Tane

class Profiler:
    def get_keycol(self, rows: List[List[str]], classes: Dict[int, str]):
        """Predict the key column of a table. ``None`` means that the table is n-ary."""
        return 0


class HeuristicProfiler(Profiler):
    """Profiler based on heuristics"""

    def get_keycol(self, rows, usecols=None, threshold=0.9):
        """Return the leftmost most-unique column in ``usecols``"""
        ci_uniq = {
            ci: len(set(col)) / len(col)
            for ci, col in enumerate(zip(*rows))
            if (not usecols) or (ci in usecols)
        }
        if ci_uniq:
            maxuniq = max(ci_uniq.values())
            if maxuniq > threshold:
                for ci, uniq in sorted(ci_uniq.items()):
                    if uniq == maxuniq:
                        return ci


class PFDProfiler(Profiler):
    """Profiler based on Probabilistic Functional Dependencies"""

    def get_keycol(self, rows, usecols=None, threshold=0.9):
        """Return the column with highest harmonic mean of incoming pFD scores"""
        col_inpFDs: Dict[int, Dict] = {}
        rowdicts = [dict(enumerate(row)) for row in rows]
        for colset, ci, score in perTuple_pFDs(rowdicts, 1):
            inci = colset[0]
            if (not usecols) or (inci in usecols):
                col_inpFDs.setdefault(inci, Counter())[ci] = score

        def harm(ss):
            return (len(ss) / sum((1 / s) for s in ss if s)) if ss else 0

        col_harmpFDs = {}
        for ci, inpFDs in col_inpFDs.items():
            h = harm(inpFDs.values())
            if h > threshold:
                col_harmpFDs[ci] = h

        if col_harmpFDs:
            return max(col_harmpFDs, key=lambda ci: col_harmpFDs[ci])


def perTuple_pFDs(
    rowdicts: List[Dict[int, str]], depsize: int
) -> Iterable[Tuple[Tuple[int, ...], int, float]]:
    """Finds probablistic functional dependencies.

    A probabilistic functional dependency :math:`X \\rightarrow_p a` indicates that two
    tuples which share the same value for the column set :math:`X` also share the same
    value for the column :math:`a` with probability :math:`p`.

    See also:

        Wang, Daisy Zhe, Xin Luna Dong, Anish Das Sarma, Michael J. Franklin, and Alon
        Y. Halevy. `Functional Dependency Generation and Applications in Pay-As-You-Go
        Data Integration Systems. <https://api.semanticscholar.org/CorpusID:15960115>`_
        In WebDB (2009)

    Args:
        rowdicts: a list of ``{colnr:text}`` dicts
        depsize: is the maximum ``len(X)``

    Returns:
        Yields ``(X, a, score)`` where ``a`` is a column number and ``X`` is a list of them
    """

    cxs_count: Dict[Tuple, int] = Counter()  # how often a subtuple occurs
    cxs_c_v_count: Dict[Tuple, Dict[int, Dict[str, int]]] = {}  # how often a subtuple occurs with a value in a column
    for row in rowdicts:
        rowset = set(row.items())
        for n in range(1, depsize + 1):
            for cxs in itertools.combinations(rowset, n):
                cxs_count[cxs] += 1
                for colnr_val in rowset:
                    if colnr_val not in cxs:
                        colnr, val = colnr_val
                        cxs_c_v_count.setdefault(cxs, {}).setdefault(
                            colnr, Counter()
                        )[val] += 1

    # how often a subtuple occurs with its most frequent co-occurring value in a column
    cxs_c_count: Dict[Tuple, Dict[int, int]] = {}
    for cxs, c_v_count in cxs_c_v_count.items():
        for c, v_count in c_v_count.items():
            cxs_c_count.setdefault(cxs, Counter())[c] = max(v_count.values())

    ncols = max((max(row, default=0) for row in rowdicts), default=0) + 1
    for n in range(1, depsize + 1):
        for cs in itertools.combinations(range(ncols), n):
            distinct = set(tuple((c, row.get(c)) for c in cs) for row in rowdicts)
            distinct_sum = sum(cxs_count.get(cxs, 0) for cxs in distinct)
            if distinct_sum:
                for a in range(ncols):
                    if a not in cs:
                        pFD = (
                            sum(cxs_c_count.get(Vx, {}).get(a, 0) for Vx in distinct)
                            / distinct_sum
                        )
                        yield (cs, a, pFD)

def pfd_prob_pervalue(R: List[Tuple[str]]):
    """
        R is a sorted dataframe where the last column is the dependent attribute
        det_count is determinant count (Vx)
        tup_count is attribute value count (Vx,Va)
        max_tup_count is most-frequent tuple count (Vx,Va)
    """
    R = sorted(R)
    assert(R)

    row = R[0]
    det, det_count, max_tup_count = row[:-1], 1, 1
    val, tup_count = row[-1], 1
    ndistinct, total_prob = 1, 0.0
    for row in R[1:]:
        if row[:-1] == det: # same determinant
            det_count += 1
            if row[-1] == val: # same tuple
                tup_count += 1
                if max_tup_count < tup_count:
                    max_tup_count = tup_count
            else: # new tuple
                val, tup_count = row[-1], 1
        else: # new determinant = new tuple
            total_prob += max_tup_count / det_count
            ndistinct += 1
            det, det_count, max_tup_count = row[:-1], 1, 1
            val, tup_count = row[-1], 1
    total_prob += max_tup_count / det_count
    return total_prob / ndistinct