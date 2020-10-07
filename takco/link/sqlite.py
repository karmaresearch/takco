"""
This module is executable. Run ``python -m takco.link.sqlite -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
from pathlib import Path
import glob
import rdflib
import logging as log
import sqlite3
import time

from .base import Searcher, SearchResult, Lookup, Database


class SQLiteCache:
    def __init__(self, files, fallback=None, sqlite_kwargs=None):
        if not isinstance(files, list):
            if "*" in files:
                if any(Path(f).exists() for f in glob.glob(files)):
                    files = glob.glob(files)
                else:
                    files = [files.replace("*", "index.sqlitedb")]
            else:
                files = [files]
        self.files = files
        self.cons = []
        self.cache = None
        self.sqlite_kwargs = sqlite_kwargs or {}
        self.fallback = fallback

    def __enter__(self):
        for file in self.files + [":memory:"]:
            if file != ":memory:":
                Path(file).parent.mkdir(parents=True, exist_ok=True)

            con = sqlite3.connect(file, **self.sqlite_kwargs).__enter__()

            if file == ":memory:":
                self.cache = con
            else:
                self.cons.append(con)

            try:
                con.execute(f"select count(*) from {self._DBNAME}").fetchall()
            except:
                try:
                    con.executescript(self._INITDB)
                    con.commit()
                except:
                    pass
        if self.fallback is not None:
            self.fallback.__enter__()

        return self

    def _write_cache(self, end=False):
        l = len(self.cons)
        cache = self.cache.execute(self._SELECTDB).fetchall()
        for ci, con in enumerate(self.cons):
            if cache:
                log.debug(f"Writing cache of size {len(cache)} to {con} ({ci+1}/{l})")

                con.executemany(self._INSERTDB, cache)
                con.commit()
        
        if not end:
            self.cache.execute(f"DROP TABLE {self._DBNAME}")
            self.cache.executescript(self._INITDB)
            self.cache.commit()

    def _addcache(self, new):
        if self.cache is None:
            raise Exception(f"no cache for {self}! Have you used __enter__?")

        self.cache.executemany(self._INSERTDB, new)
        self.cache.commit()

    def __exit__(self, *args):
        self._write_cache(end=True)

        for con in self.cons:
            con.__exit__(*args)
        self.cons = []

        self.cache.__exit__(*args)
        if self.fallback is not None:
            self.fallback.__exit__(*args)


class SQLiteLookup(SQLiteCache, Lookup):
    _DBNAME = "Lookup"
    _INITDB = """
        CREATE TABLE Lookup(
            title TEXT,
            uri TEXT
        );
        CREATE INDEX IF NOT EXISTS Lookup_uri ON Lookup(uri);
        CREATE INDEX IF NOT EXISTS Lookup_title ON Lookup(title);
    """
    _INSERTDB = "insert or replace into Lookup(title, uri) values (?,?)"
    _SELECTDB = "select title, uri from Lookup"

    def __init__(
        self,
        sqlitedb: typing.List[Path],
        baseuri="",
        cache_often=False,
        fallback: Lookup = None,
        sqlite_kwargs=None,
    ):
        self.baseuri = baseuri
        self.cache_often = cache_often
        SQLiteCache.__init__(
            self, files=sqlitedb, fallback=fallback, sqlite_kwargs=sqlite_kwargs
        )

    def __enter__(self):
        return SQLiteCache.__enter__(self)

    def __exit__(self, *args):
        return SQLiteCache.__exit__(self, *args)
    
    def flush(self):
        if self.cache_often:
            self._write_cache()
    
    def lookup_title(self, title: str) -> str:
        t = title.replace("_", " ").lower()
        """Gets the URI for a DBpedia entity based on wikipedia title."""
        try:
            for con in self.cons:
                q = "select uri from Lookup where title=:t"
                for uri in con.execute(q, {"t": t}).fetchone() or []:
                    if str(uri) == "-1":
                        continue
                    if not uri:
                        return
                    return self.baseuri + str(uri)

            if self.fallback is not None:
                uri = self.fallback.lookup_title(title)
                if str(uri) == "-1":
                    uri = None

                new = (t, uri.replace(self.baseuri, "") if uri else None)
                self._addcache([new])
                log.debug(f"Found Wikititle fallback for {title}: {uri}")
                return uri
        except Exception as e:
            log.error(e)


class SQLiteDB(SQLiteCache, Database):
    _DBNAME = "Triples"
    _INITDB = """
        CREATE TABLE Triples(
            s TEXT,
            p TEXT,
            o TEXT
        );
        CREATE INDEX IF NOT EXISTS Triples_s ON Triples(s);
        CREATE INDEX IF NOT EXISTS Triples_po ON Triples(p,o);
    """
    _INSERTDB = "insert or replace into Triples(s,p,o) values (?,?,?)"
    _SELECTDB = "select s,p,o from Triples"

    def __init__(
        self,
        sqlitedb: Path,
        baseuri="",
        propbaseuri="",
        fallback: Database = None,
        sqlite_kwargs=None,
    ):
        self.baseuri = baseuri
        self.propbaseuri = propbaseuri or baseuri
        SQLiteCache.__init__(
            self, files=sqlitedb, fallback=fallback, sqlite_kwargs=sqlite_kwargs
        )

    def __enter__(self):
        return SQLiteCache.__enter__(self)

    def __exit__(self, *args):
        return SQLiteCache.__exit__(self, *args)

    def get_prop_values(self, e, p):
        eid = str(e).replace(self.baseuri, "")
        pid = str(p).replace(self.propbaseuri, "")
        vals = []
        try:
            for con in self.cons:
                q = "select o from Triples where s=:s"
                for (o,) in con.execute(q, {"s": eid}).fetchall() or []:
                    if o is None:
                        return []
                    o = (self.baseuri or "") + o
                    vals.append(o)

            if (not vals) and (self.fallback is not None):
                vals = self.fallback.get_prop_values(e, p) or [None]
                triples = [
                    (
                        eid,
                        pid,
                        str(o).replace(self.baseuri, "") if o is not None else None,
                    )
                    for o in vals
                ]
                self._addcache(triples)
                log.debug(f"Found SQLiteDB fallback for {e}")

        except Exception as e:
            log.error(e)

        return vals

    def about(self, uri):
        s = uri.replace(self.baseuri, "")
        doc = {}
        try:
            for con in self.cons:
                q = "select p,o from Triples where s=:s"
                for (p, o) in con.execute(q, {"s": s}).fetchall() or []:
                    p = (self.propbaseuri or "") + p
                    o = (self.baseuri or "") + o
                    doc.setdefault(p, []).append(o)

            if (not doc) and (self.fallback is not None):
                doc = self.fallback.about(uri) or {}

                q = "insert or replace into Triples(s,p,o) values (?,?,?)"
                triples = [
                    (
                        s,
                        str(p).replace(self.propbaseuri, ""),
                        str(o).replace(self.baseuri, ""),
                    )
                    for p, os in doc.items()
                    for o in os
                ]
                self._addcache(triples)
                log.debug(f"Found SQLiteDB fallback for {uri}")

        except Exception as e:
            log.error(e)

        return doc


class SQLiteSearcher(SQLiteCache, Searcher):
    _DBNAME = "label"
    _INITDB = """
        CREATE TABLE label(
            uri TEXT,
            txt TEXT,
            score REAL
        );
        CREATE INDEX IF NOT EXISTS label_uri ON label(uri);
        CREATE INDEX IF NOT EXISTS label_txt ON label(txt);
    """
    _INSERTDB = "insert or replace into label(uri, txt, score) values (?,?,?)"
    _SELECTDB = "select uri, txt, score from label"

    _DEFAULT_SCORES = {
        "http://www.w3.org/2000/01/rdf-schema#label": 1,
        "http://schema.org/name": 1,
        "http://www.w3.org/2004/02/skos/core#prefLabel": 1,
        "http://www.w3.org/2004/02/skos/core#altLabel": 0.5,
    }
    _LIKEQUERY = """
        SELECT uri, txt, score FROM label WHERE txt LIKE :query LIMIT :limit
    """
    _EXACTQUERY = """
        SELECT uri, txt, score FROM label WHERE txt == :query LIMIT :limit
    """

    def __init__(
        self,
        files,
        graph=None,
        exact=True,
        lower=True,
        parts=True,
        baseuri=None,
        fallback=None,
        refsort=True,
        sqlite_kwargs=None,
        **_,
    ):
        self.graph = graph
        self.baseuri = baseuri or ""
        self.exact = exact
        self.lower = lower
        self.parts = parts
        self.refsort = refsort

        SQLiteCache.__init__(
            self, files=files, fallback=fallback, sqlite_kwargs=sqlite_kwargs
        )

    def __enter__(self):
        return SQLiteCache.__enter__(self)

    def __exit__(self, *args):
        return SQLiteCache.__exit__(self, *args)

    def search_entities(self, query: str, context=(), limit=1, add_about=False):
        if not query:
            return []

        if self.lower:
            query = query.lower()

        queries = [query]

        all_results = []
        knownempty = False
        for con in self.cons:
            cur = con.cursor()
            params = {"query": query, "limit": limit}
            q = self._EXACTQUERY if self.exact else self._LIKEQUERY

            results = cur.execute(q, params).fetchall()
            for uri, txt, score in results:
                if uri == "-1":
                    query_knownempty = True
                else:
                    if self.baseuri:
                        uri = self.baseuri + uri
                    if self.refsort:
                        c = self.graph.count([None, None, rdflib.URIRef(uri)])
                        log.debug(f"{uri} scored {score} * {1-1/(5+c):.4f}")
                        score *= 1 - 1 / (5 + c)
                    sr = SearchResult(uri, score=score)
                    all_results.append(sr)

        n = len(all_results)
        log.debug(f"{self.__class__.__name__} got {n} results for {query}")

        q = "insert or replace into label(uri, txt, score) values (?,?,?)"
        if (self.fallback is not None) and not (knownempty or all_results):
            all_results += list(self.fallback.search_entities(query, limit=None))
            if all_results:
                new = []
                for sr in all_results:
                    uri = sr.uri.replace(self.baseuri, "") if sr.uri else "-1"
                    new.append((uri, query, sr.score))
                self._addcache(new)
            else:
                self._addcache([("-1", query, 1)])
                log.debug(f"Added SQLiteSearcher empty-result-value for '{query}' ")

        if add_about and self.graph:
            all_results = [
                SearchResult(sr.uri, self.graph.about(sr.uri), sr.score)
                for sr in all_results
            ]

        if not all_results:
            if self.parts:
                for char in "([,:":
                    for qpart in query.split(char):
                        qpart = qpart.translate(str.maketrans("", "", ")]")).strip()
                        if qpart != query:
                            all_results += self.search_entities(
                                qpart, limit=limit, add_about=add_about
                            )

        all_results = sorted(all_results, key=lambda x: -x.score)

        return all_results[:limit]

    @staticmethod
    def createdb(
        triplefile: Path,
        outdir: Path,
        baseuri: str = None,
        langmatch: str = "en",
        chunk: int = 10 ** 4,
        scoredict: typing.Dict = None,
        limit: int = None,
    ):
        import re

        scoredict = scoredict or SQLiteSearcher._DEFAULT_SCORES
        nbaseuri = len(baseuri) if baseuri else 0

        b = baseuri or ""
        l = f"@{langmatch}" if langmatch else ""
        LINE = f'<{b}(?P<s>[^>]*)> <(?P<p>[^>]*)> "(?P<v>.*)"{l} .$'
        LINE = re.compile(LINE)

        log.debug(f"Creating a db for {triplefile} in {outdir}")
        fname = Path(outdir) / Path(Path(triplefile).name.split(".")[0] + ".sqlitedb")
        with sqlite3.connect(fname) as con:
            cur = con.cursor()
            cur.executescript(SQLiteSearcher._INITDB)
            con.commit()

            tuples = []
            for li, line in enumerate(triplefile.open()):
                if tuples and not (li % chunk):
                    cur.executemany("INSERT INTO label VALUES (?,?,?)", tuples)
                    tuples = []

                try:
                    m = LINE.match(line.strip())
                    if m:
                        m = m.groupdict()
                        s, p, val = m["s"], m["p"], m["v"]
                        score = scoredict.get(p, 0)
                        if score and s:
                            val = val.encode().decode("unicode-escape").lower()
                            tuples.append((s, val, score))
                        else:
                            log.debug(f"Bad triple: {(s, p, o)}")
                except Exception as e:
                    log.debug(f"Bad line: {line}")

                if limit and (li > limit):
                    break
            if tuples:
                cur.executemany("INSERT INTO label VALUES (?,?,?)", tuples)

            con.commit()
        return fname

    @classmethod
    def create(
        cls,
        triplefiles: typing.List[Path],
        outdir: Path = Path("."),
        baseuri: str = None,
        langmatch: str = "en",
        nlimit: int = None,
        chunk: int = 10 ** 5,
        scoredict: typing.Dict = {},
        max_workers: int = 2,
    ):
        """Create SQLite DBs from triples"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm

        Path(outdir).mkdir(exist_ok=True, parents=True)

        scoredict["http://www.w3.org/2000/01/rdf-schema#label"] = 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            kwargs = dict(
                baseuri=baseuri,
                langmatch=langmatch,
                scoredict=scoredict,
                chunk=chunk,
                limit=nlimit,
            )
            futures = {
                executor.submit(cls.createdb, f, outdir, **kwargs): f
                for f in triplefiles
            }
            pbar = tqdm.tqdm(as_completed(futures), total=len(futures))
            pbar.set_description(f"Indexing")
            nrows = 0
            for future in pbar:
                fname = future.result()
                with sqlite3.connect(fname) as con:
                    cur = con.cursor()
                    nrows += cur.execute("select count(*) from label").fetchone()[0]
                    pbar.set_description(f"Indexed {nrows} labels")

    @classmethod
    def test(cls, sqlitedir: Path, query: str, limit: int = 1):
        """Search a set of sqlite label DBs for a query string """
        import json

        s = cls(files=sqlitedir.glob("*.sqlitedb"))
        return json.dumps(s.search_entities(query, limit=limit))


if __name__ == "__main__":
    import defopt, json, os

    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL", "WARN")))

    r = defopt.run(
        [SQLiteSearcher.create, SQLiteSearcher.test],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
