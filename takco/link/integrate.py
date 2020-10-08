from typing import List, Tuple, Dict, Container, Iterator, NamedTuple, Optional
import logging as log
from collections import defaultdict, Counter

from . import types
from .base import Database, LiteralMatchResult, Triple


class QualifierMatchResult(NamedTuple):
    """The result of matching a qualified statement in the Knowledge Base"""

    column: int  #: Table column where this match was found
    triple: Triple  #: Statement qualifier triple
    match: Optional[LiteralMatchResult]  #: Match metadata


class NaryMatchResult(NamedTuple):
    """The result of matching a Knowledge Base statement"""

    columns: Tuple[int, int]  #: Pair of (head, tail) table columns
    triple: Triple  #: Statement triple
    qualifiers: Container[QualifierMatchResult]  #: Statement qualifier matches


class NaryDB(Database):
    """For querying a KB with qualifiers."""

    def get_rowfacts(
        self, celltexts: List[str], entsets: List[Container[str]]
    ) -> Iterator[NaryMatchResult]:
        return

    def integrate(self, rows, row_entsets):
        """Find n-ary matches"""
        nrows = len(rows)

        colmatch_count = Counter()
        colmatch_qcolprop_count = defaultdict(Counter)
        for celltexts, entsets in zip(rows, row_entsets):

            for (c1, c2), (s, p, o), qs in self.get_rowfacts(celltexts, entsets):
                if c1 == c2:
                    continue
                colmatch_count[(c1, c2, p)] += 1
                for qcol, (q, qprop, qo), m in qs:
                    colmatch_qcolprop_count[(c1, c2, p)][(qcol, qprop)] += 1

        log.debug(
            f"colmatch_count = {colmatch_count}, colmatch_qcolprop_count = {colmatch_qcolprop_count}"
        )

        def by_qualifier_count(cm):
            return len(colmatch_qcolprop_count.get(cm, [])), colmatch_count[cm]

        tocol_fromcolprop = {}
        for c1, c2, p in sorted(colmatch_count, key=by_qualifier_count):
            qcolprop_count = colmatch_qcolprop_count[(c1, c2, p)]
            n = colmatch_count[(c1, c2, p)]
            tocol_fromcolprop[c2] = {c1: {p: n / nrows}}

            col_qprops = defaultdict(Counter)
            for (col, prop), count in qcolprop_count.items():
                col_qprops[col][prop] += count

            # Add qualifier props
            for ci, qprops in col_qprops.items():
                for qp, qn in qprops.most_common(1):
                    tocol_fromcolprop[ci] = {c1: {qp: qn / nrows}}

        # Add most frequent other relations
        for (c1, c2, p), n in colmatch_count.most_common():
            if c1 != c2 and c2 not in tocol_fromcolprop:
                tocol_fromcolprop[c2] = {c1: {p: n / nrows}}

        return tocol_fromcolprop
