from pathlib import Path
import glob

import rdflib
import logging as log
import sqlite3

from .base import Searcher, SearchResult, WikiLookup


class SQLiteWikiLookup(WikiLookup):
    _INITDB = """
        CREATE TABLE WikiLookup(
            title TEXT,
            uri TEXT
        );
        CREATE INDEX IF NOT EXISTS WikiLookup_uri ON WikiLookup(uri);
        CREATE INDEX IF NOT EXISTS WikiLookup_title ON WikiLookup(title);
    """

    def __init__(self, sqlitedb: Path, baseuri="", fallback: WikiLookup = None):
        self.sqlitedb = sqlitedb
        self.baseuri = baseuri
        self.fallback = fallback

    def lookup_wikititle(self, title: str) -> str:
        """Gets the URI for a DBpedia entity based on wikipedia title."""
        title = title.replace(" ", "_")
        with sqlite3.connect(self.sqlitedb) as con:
            q = "select uri from WikiLookup where title=:q"
            for uri in con.execute(q, {"q": title}).fetchone() or []:
                if str(uri) == "-1":
                    continue
                if not uri:
                    return
                return self.baseuri + str(uri)

            if self.fallback:
                uri = self.fallback.lookup_wikititle(title)
                if str(uri) != "-1":
                    uri = None
                q = "insert or replace into WikiLookup(title, uri) values (?,?)"
                con.execute(q, [title, uri.replace(self.baseuri, "") if uri else None])
                con.commit()
                return uri


class SQLiteSearcher(Searcher):
    _INITDB = """
        CREATE TABLE label(
            uri TEXT,
            txt TEXT,
            score REAL
        );
        CREATE INDEX IF NOT EXISTS label_uri ON label(uri);
        CREATE INDEX IF NOT EXISTS label_txt ON label(txt);
    """
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
        self, files, graph=None, exact=True, lower=True, parts=True,
        baseuri=None, fallback=None,
    ):
        self.graph = graph
        if "*" in files:
            files = glob.glob(files)
        self.files = files
        self.baseuri = baseuri
        self.fallback = fallback
        self.exact = exact
        self.lower = lower
        self.parts = parts

        sizes = []
        for fname in self.files:
            with sqlite3.connect(fname) as con:
                sizes.append(con.execute("SELECT COUNT(*) FROM label").fetchone())
        log.debug(f"Using SQlite DBs with sizes {sizes}")

    def search_entities(self, query: str, limit=1, add_about=False):
        if not query:
            return []

        if self.lower:
            query = query.lower()
            
        queries = [query]
        if self.parts:
            for char in '([,:':
                queries += query.split(char)
            queries = set(queries)
        
        all_results = []
        for query in queries:
            query = query.strip()
            
            for fname in self.files:
                with sqlite3.connect(fname) as con:
                    cur = con.cursor()
                    params = {"query": query, "limit": limit}
                    q = self._EXACTQUERY if self.exact else self._LIKEQUERY

                    results = cur.execute(q, params).fetchall()
                    for uri, txt, score in results:
                        if self.baseuri:
                            uri = self.baseuri + uri
                        sr = SearchResult(uri, score=score)
                        all_results.append(sr)

        n = len(all_results)
        log.debug(f"{self.__class__.__name__} got {n} results for {query}")
        
        for query in queries:
            q = "insert or replace into label(uri, text, score) values (?,?,?)"
            if self.fallback and (not all_results or str(all_results[0].uri) == "-1"):
                all_results += list(self.fallback.search_entities(query, limit=limit))
                if all_results:
                    for fname in self.files:
                        with sqlite3.connect(fname) as con:
                            for sr in all_results:
                                uri = sr.uri.replace(self.baseuri, "") if uri else "-1"
                                con.execute(q, [uri, query.lower(), sr.score])
                else:
                    for fname in self.files:
                        with sqlite3.connect(fname) as con:
                            for sr in all_results:
                                con.execute(q, ["-1", query.lower(), 1])

        if add_about and self.graph:
            all_results = [
                SearchResult(sr.uri, self.graph.about(sr.uri), sr.score)
                for sr in all_results
            ]

        return all_results[:limit]


if __name__ == "__main__":
    import defopt, typing, tqdm, re, json
    from pathlib import Path
    import logging as log
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def createdb(
        triplefile,
        outdir,
        baseuri=None,
        langmatch="en",
        chunk=10 ** 4,
        scoredict=None,
        limit=None,
    ):
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

    def create(
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
                executor.submit(createdb, f, outdir, **kwargs): f for f in triplefiles
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

    def search(sqlitedir: Path, query: str, limit: int = 1):
        """Search a set of sqlite label DBs for a query string """

        s = SQLiteSearcher(files=sqlitedir.glob("*.sqlitedb"))
        return json.dumps(s.search_entities(query, limit=limit))

    r = defopt.run(
        [create, search], strict_kwonly=False, parsers={typing.Dict: json.loads}
    )
    print(r)
