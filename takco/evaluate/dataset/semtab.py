"""
This module is executable. Run ``python -m takco.evaluate.dataset.t2d -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import logging as log
import csv
import re
from typing import List

from .dataset import Dataset


class Semtab(Dataset):
    ISNUMBER = re.compile("^[\d.,\-\+\%]+$")

    def get_name(self, fpath):
        return Path(fpath).name.split(".")[0]

    def __init__(self, root: Path, **kwargs):
        self.root = Path(root)

    @property
    def tables(self):
        for fname in Path(self.root / Path("tables")).glob("*.csv"):
            tablefile = open(fname, "rb").read().decode("utf-8", errors="ignore")
            rows = list(csv.reader(tablefile.splitlines()))

            entcols = [
                ci
                for ci, col in enumerate(zip(*rows))
                if sum(bool(self.ISNUMBER.match(c)) for c in col) < len(col) / 2
            ]

            yield {
                "name": fname.name,
                "rows": rows[1:],
                "headers": rows[:1],
                "keycol": None,
                "entcols": entcols,
                "numheaderrows": 0,
            }

    def get_unannotated_tables(self, cache=True):
        import json

        cachefile = Path(self.root / Path("tables.jsonl"))
        if cache:
            if not cachefile.exists():
                with cachefile.open("w") as fw:
                    for table in self.tables:
                        print(json.dumps(table), file=fw)
            tables = (json.loads(line) for line in cachefile.open())
        else:
            tables = self.tables

        for table in tables:
            rows = [[{"text": c} for c in row] for row in table.get("rows")]
            headers = [[{"text": c} for c in row] for row in table.get("headers")]
            yield {
                "_id": table.get("name", ""),
                "tableData": rows,
                "tableHeaders": headers,
                "entcols": table["entcols"],
                "keycol": table["keycol"],
            }


if __name__ == "__main__":
    import defopt, json, csv, sys, tqdm
    from pathlib import Path
    from collections import Counter

    def cea(predfile: Path, targetfile: Path):
        """Convert predictions to Cell-Entity Annotation format given targets"""

        preds = {}
        with predfile.open() as f:
            tables = (json.loads(line) for line in f)
            preds = {t.get("_id"): t.get("entities", {}) for t in tables}
        if not preds:
            raise Exception("no predictions!")

        # The CEA dataset has 3 or 4 columns
        cw = csv.writer(sys.stdout)
        rows: List[str] = []
        n_total, n_annotated = 0, 0
        for line in tqdm.tqdm(targetfile.open()):
            n_total += 1
            row = line.strip().split(",")
            if len(row) < 4:
                row += [""]
            table, ri, ci, uri = row
            ents = preds.get(table + ".csv", {}).get(ci, {}).get(str(int(ri) - 1), {})
            for uri, _ in Counter(ents).most_common(1): # type: ignore
                if uri:
                    cw.writerow([table, ri, ci, uri])
                    n_annotated += 1

        print(f"Annotated {n_annotated} / {n_total} items", file=sys.stderr)

    def cta(predfile: Path, targetfile: Path):
        """Convert predictions to Column-Type Annotation format given targets"""

        preds = {}
        with predfile.open() as f:
            tables = (json.loads(line) for line in f)
            preds = {t.get("_id"): t.get("classes", {}) for t in tables}
        if not preds:
            raise Exception("no predictions!")

        # The CEA dataset has 3 columns
        cw = csv.writer(sys.stdout)
        rows: List[str] = []
        n_total, n_annotated = 0, 0
        for line in tqdm.tqdm(targetfile.open()):
            n_total += 1
            table, ci = line.strip().split(",")
            classes = preds.get(table + ".csv", {}).get(ci, {})
            for uri, _ in Counter(classes).most_common(1): # type: ignore
                if uri:
                    cw.writerow([table, ci, uri])
                    n_annotated += 1

        print(f"Annotated {n_annotated} / {n_total} items", file=sys.stderr)

    def cpa(predfile: Path, targetfile: Path):
        """Convert predictions to Column-Property Annotation format given targets"""

        preds = {}
        with predfile.open() as f:
            tables = (json.loads(line) for line in f)
            preds = {t.get("_id"): t.get("properties", {}) for t in tables}
        if not preds:
            raise Exception("no predictions!")

        # The CEA dataset has 3 columns
        cw = csv.writer(sys.stdout)
        rows: List[str] = []
        n_total, n_annotated = 0, 0
        for line in tqdm.tqdm(targetfile.open()):
            n_total += 1
            table, fromci, toci = line.strip().split(",")
            props = preds.get(table + ".csv", {}).get(fromci, {}).get(toci, {})
            for uri, _ in Counter(props).most_common(1): # type: ignore
                if uri:
                    cw.writerow([table, fromci, toci, uri])
                    n_annotated += 1

        print(f"Annotated {n_annotated} / {n_total} items", file=sys.stderr)

    defopt.run([cea, cta, cpa], strict_kwonly=False)
