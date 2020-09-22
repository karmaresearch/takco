import sys, os, json, logging as log, collections, itertools

from typing import List, Dict, Tuple, Iterator


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
        col_inpFDs = {}
        rowdicts = [dict(enumerate(row)) for row in rows]
        for colset, ci, score in perTuple_pFDs(rowdicts, 1):
            inci = colset[0]
            if (not usecols) or (inci in usecols):
                col_inpFDs.setdefault(inci, collections.Counter())[ci] = score

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
) -> Iterator[Tuple[List[int], int, float]]:
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

    Vx_count = collections.Counter()  # how often a subtuple occurs
    Vx_a_Va_count = {}  # how often a subtuple occurs with a value in a column
    for row in rowdicts:
        row = set(row.items())
        for n in range(1, depsize + 1):
            for Vx in itertools.combinations(row, n):
                Vx_count[Vx] += 1
                for Va in row:
                    if Va not in Vx:
                        a, Va = Va
                        Vx_a_Va_count.setdefault(Vx, {}).setdefault(
                            a, collections.Counter()
                        )[Va] += 1

    # how often a subtuple occurs with its most frequent co-occurring value in a column
    VaVx_count = {}
    for Vx, a_Va_count in Vx_a_Va_count.items():
        for a, Va_count in a_Va_count.items():
            VaVx_count.setdefault(Vx, collections.Counter())[a] = max(Va_count.values())

    ncols = max((max(row, default=0) for row in rowdicts), default=0) + 1
    for n in range(1, depsize + 1):
        for X in itertools.combinations(range(ncols), n):
            distinct = set(tuple((x, row.get(x, None)) for x in X) for row in rowdicts)
            distinct_sum = sum(Vx_count.get(Vx, 0) for Vx in distinct)
            if distinct_sum:
                for a in range(ncols):
                    if a not in X:
                        pFD = (
                            sum(VaVx_count.get(Vx, {}).get(a, 0) for Vx in distinct)
                            / distinct_sum
                        )
                        yield (X, a, pFD)
