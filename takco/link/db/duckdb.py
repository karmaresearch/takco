"""
This module is executable. Run ``python -m takco.link.db.duckdb -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import urllib.parse as ul
import logging as log

import pandas as pd

try:
    import duckdb
except:
    log.error("Could not import duckdb")


def read_tsv(fname, names, **kwargs):
    return pd.read_csv(
        fname,
        sep="\t",
        names=names,
        converters={
            "uri": ul.unquote_plus,
            "id": ul.unquote_plus,
            "src": ul.unquote_plus,
            "dst": ul.unquote_plus,
        },
        dtype={"count": "Int64"},
        error_bad_lines=False,
        **kwargs,
    )


def create(tsvpath: Path, dbpath: Path):
    """
        Unfortunately, far to slow to work in practice
    """
    assert not Path(dbpath).exists()
    tsvpath = Path(tsvpath)

    headers = {
        "uriCounts": ["uri", "uricount"],
        "pairCounts": ["form", "uri", "paircount"],
        "sfAndTotalCounts": ["form", "formcount", "tokencount"],
    }

    schema = """
    CREATE TABLE uriCounts(uri VARCHAR, uricount INTEGER);
    CREATE TABLE pairCounts(surfaceform VARCHAR, uri VARCHAR, paircount INTEGER);
    CREATE TABLE sfAndTotalCounts(surfaceform VARCHAR, surfaceformcount INTEGER, tokencount INTEGER);

    CREATE INDEX uriCounts_uri ON uriCounts (uri);
    CREATE INDEX pairCounts_uri ON pairCounts (uri);
    CREATE INDEX pairCounts_surfaceform ON pairCounts (surfaceform);
    CREATE INDEX sfAndTotalCounts_surfaceform ON sfAndTotalCounts (surfaceform);
    """

    con = duckdb.connect(database=str(dbpath), read_only=False)
    for line in schema.split(";"):
        if line.strip():
            con.execute(line + ";")

    for name, colnames in headers.items():
        fname = tsvpath.joinpath(f"spotlight-wikistats_type={name}_lang=en.tsv")
        df = read_tsv(fname, colnames)
        log.info(f"Loaded {name}, ({len(df)} rows, {dict(df.dtypes)}")
        con.register(f"{name}_df", df)
        con.execute(f'INSERT INTO {name} (SELECT * FROM "{name}_df")')


def query(dbpath: str, txt: str):
    con = duckdb.connect(database=dbpath, read_only=True)
    df = con.execute(
        """
  SELECT *
  FROM pairCounts p
  JOIN uriCounts u ON (p.uri = u.uri)
  JOIN sfAndTotalCounts s ON (s.surfaceform = p.surfaceform)
  WHERE p.surfaceform = ?
  """,
        [txt],
    ).fetchdf()
    print(df)


if __name__ == "__main__":
    import defopt, json, os, typing

    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL", "WARN")))

    r = defopt.run(
        [create, query],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
