from typing import List, Dict
import logging as log
from collections import defaultdict, Counter

from . import datatype


class NaryIntegrator:
    def __init__(self, db):
        self.db = db

    def integrate(self, rows, row_entsets):
        """Find n-ary matches"""
        nrows = len(rows)

        colmatch_count = Counter()
        colmatch_qcolprops = defaultdict(Counter)
        for celltexts, entsets in zip(rows, row_entsets):
            for (c1, c2), (s, p, o), qs in self.db.get_rowfacts(celltexts, entsets):
                if c1 == c2:
                    continue
                colmatch_count[(c1, c2, p)] += 1
                for qcol, (q, qprop, qo), m in qs:
                    colmatch_qcolprops[(c1, c2, p)][(qcol, qprop)] += 1

        def by_qualifier_count(cm):
            return len(colmatch_qcolprops.get(cm, [])), colmatch_count[cm]

        tocol_fromcolprop = {}
        for c1, c2, p in sorted(colmatch_count, key=by_qualifier_count):
            qcolprops = colmatch_qcolprops[(c1, c2, p)]
            n = colmatch_count[(c1, c2, p)]
            tocol_fromcolprop[c2] = {c1: {p: n / nrows}}

            col_qprops = defaultdict(Counter)
            for (col, prop), count in qcolprops:
                col_qprops[col][prop] += count

            # Add qualifier props
            for ci, qprops in col_qprops.items():
                for qp, qn in qprops.most_common(1):
                    tocol_fromcolprop[ci] = {c1: {p: qn / nrows}}

        # Add most frequent other relations
        for (c1, c2, p), n in colmatch_count.most_common():
            if c1 != c2 and c2 not in tocol_fromcolprop:
                tocol_fromcolprop[c2] = {c1: {p: n / nrows}}

        return tocol_fromcolprop
