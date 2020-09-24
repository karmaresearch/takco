"""
This module is executable. Run ``python -m takco.link.elasticsearch -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
import logging as log
from .base import Searcher, SearchResult
from pathlib import Path
import string

ONLY_IDF = (
    "double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0;"
    "double norm = 1/Math.sqrt(doc.length);"
    "return query.boost * idf * norm;"
)
SETTINGS = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "score": {"type": "float"},
            "surface": {
                "analyzer": "standard",
                "similarity": "only_idf",
                "type": "text",
            },
        }
    },
    "settings": {
        "index": {
            "similarity": {
                "only_idf": {"script": {"source": ONLY_IDF}, "type": "scripted"}
            },
        }
    },
}


NOPUNCT = str.maketrans("", "", string.punctuation)


def make_query_body(surface, limit=10):
    surface = surface.lower()  # .translate(NOPUNCT)
    return {
        "size": limit,
        "query": {
            "function_score": {
                "query": {
                    "dis_max": {
                        "queries": [
                            {
                                "match_phrase": {
                                    "surface": {
                                        "query": surface,
                                        "analyzer": "standard",
                                    }
                                }
                            },
                            {
                                "match": {
                                    "surface": {
                                        "query": surface,
                                        "analyzer": "standard",
                                        "boost": 0.5,
                                    }
                                }
                            },
                        ],
                    }
                },
                "functions": [
                    {"field_value_factor": {"field": "score", "modifier": "log1p",}}
                ],
                "boost_mode": "sum",
            }
        },
    }


class ElasticSearcher(Searcher):
    def __init__(self, index, baseuri=None, es_kwargs=None, parts=True, **_):
        from elasticsearch import Elasticsearch

        es_kwargs = es_kwargs or {}
        self.es = Elasticsearch(**es_kwargs)
        self.index = index
        self.baseuri = baseuri
        self.parts = parts

    def search_entities(self, query: str, limit=1, add_about=False):
        body = make_query_body(query, limit=limit)
        esresults = self.es.search(index=self.index, body=body)

        results = []
        for hit in esresults.get("hits", []).get("hits", []):
            uri = hit.get("_source", {}).get("id")
            label = hit.get("_source", {}).get("surface")
            if self.baseuri:
                uri = self.baseuri + uri
            score = hit.get("_score", 0)
            sr = SearchResult(uri, {"label": [label]}, score=score)
            results.append(sr)

        if not results:
            if self.parts:
                for char in "([,:":
                    for qpart in query.split(char):
                        qpart = qpart.translate(str.maketrans("", "", ")]")).strip()
                        if qpart != query:
                            results += self.search_entities(
                                qpart, limit=limit, add_about=add_about
                            )

        return results

    def labels(self, uri: str):
        id = str(uri).replace(self.baseuri, "")
        esresults = self.es.search(
            index="test-1", body={"query": {"match": {"id": id}}}
        )
        hits = esresults.get("hits", []).get("hits", [])
        return set([h.get("_source", {}).get("surface") for h in hits])

    @classmethod
    def load_synonyms(cls, redirects_label: Path):
        import urllib.parse as ul

        for line in Path(redirects_label).open():
            line = ul.unquote_plus(line)
            try:
                a, b = line.split("\t", 1)
                a, b = a.strip(), b.strip()
                if a and b and (a != b):
                    yield b, a
            except:
                pass

    @classmethod
    def tables2surface(
        cls,
        *tables: Path,
        baseuri: str = "http://dbpedia.org/resource/",
        skiprows: int = 4,
        uricol: int = 0,
        surfacecol: int = 1,
        score: float = 1,
    ):
        """Make SurfaceFormScores from kb tables """
        import gzip, csv, tqdm

        for fname in tqdm.tqdm(tables):
            zipped = fname.name.endswith("gz")
            with (gzip.open(fname, "rt") if zipped else fname.open()) as o:
                for ri, row in enumerate(csv.reader(o)):
                    if ri < skiprows:
                        continue
                    uri, surface = row[uricol], row[surfacecol]
                    uri = uri.replace(baseuri, "")
                    surface = surface.lower()
                    if surface and (surface != "null"):
                        print(uri, surface, score, sep="\t")

    @classmethod
    def create(
        cls,
        surfaceFormsScores: Path,
        index: str,
        synonym_file: Path = None,
        recreate: bool = True,
        chunksize: int = 10 ** 4,
        limit: int = None,
        thread_count: int = 8,
        es_kwargs: typing.Dict = None,
    ):
        """Create Elasticsearch index from surfaceFormsScores file"""
        from elasticsearch import Elasticsearch, helpers
        import tqdm, time, sys

        es_kwargs = es_kwargs or {}
        es = Elasticsearch(timeout=1000, **es_kwargs)

        if recreate:
            es.indices.delete(index=index, ignore=[400, 404])
            print("Creating index...", file=sys.stderr)
            es.indices.create(index=index, body=SETTINGS)

        action = {"_index": index}

        syn, check_syn = {}, ()
        if synonym_file:
            print("Loading synonyms from", synonym_file, file=sys.stderr)
            t = 0
            if Path(synonym_file).is_file():
                try:
                    from subprocess import run

                    wc = run(["wc", "-l", synonym_file], capture_output=True)
                    t = int(wc.stdout.split()[0])
                except:
                    pass
            syn = dict(tqdm.tqdm(cls.load_synonyms(synonym_file), total=t))

            try:
                from pybloomfilter import BloomFilter

                print(f"Making Bloom filter", file=sys.stderr)
                check_syn = BloomFilter(len(syn), 0.1, "/tmp/filter.bloom")
                check_syn.update(syn)
            except:
                check_syn = syn

        print(f"Using {len(syn)} synonyms", file=sys.stderr)

        def get_synonyms(s, path=()):
            if s and (s not in path):
                yield s
                if s in check_syn:
                    yield from get_synonyms(syn.get(s), path + (s,))

        def stream(chunksize=10 ** 4, limit=None):
            with tqdm.tqdm(total=limit) as bar:
                with Path(surfaceFormsScores).open() as fo:
                    for i, line in enumerate(fo):
                        try:
                            id, surface, score = line.split("\t")
                            score = float(score)
                            surface = surface.encode().decode("unicode-escape")

                            for s in get_synonyms(surface):
                                yield {
                                    **action,
                                    "id": id,
                                    "surface": s,
                                    "score": score,
                                }
                        except Exception as e:
                            pass

                        if not i % chunksize:
                            bar.update(chunksize)

                        if limit and (i > limit):
                            break

        if (limit is None) and Path(surfaceFormsScores).is_file():
            try:
                from subprocess import run

                wc = run(["wc", "-l", surfaceFormsScores], capture_output=True)
                limit = int(wc.stdout.split()[0])
            except:
                pass

        results = helpers.parallel_bulk(
            es, stream(limit=limit), thread_count=thread_count
        )
        for i, (status, r) in enumerate(results):
            if not status:
                print("ERROR", r, file=sys.stderr)

        time.sleep(1)
        print(
            f"Indexed {es.count(index=index).get('count')} documents", file=sys.stderr
        )

    @classmethod
    def test(
        cls, index: str, query: str, limit: int = 1, es_kwargs: typing.Dict = None
    ):
        """Search an Elasticsearch index for a query string """
        import json

        es_kwargs = es_kwargs or {}
        es = cls(index, es_kwargs=es_kwargs)
        return json.dumps(es.search_entities(query, limit=limit))


if __name__ == "__main__":
    import defopt, json, os

    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL", "WARN")))

    r = defopt.run(
        [ElasticSearcher.tables2surface, ElasticSearcher.create, ElasticSearcher.test,],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
