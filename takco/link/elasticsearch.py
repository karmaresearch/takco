import logging as log
from .base import Searcher, SearchResult

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
                "search_analyzer": "my_synonyms",
                "similarity": "only_idf",
                "type": "text",
            },
        }
    },
    "settings": {
        "index": {
            "analysis": {
                "analyzer": {
                    "my_synonyms": {
                        "filter": ["my_synonyms"],
                        "tokenizer": "whitespace",
                    }
                },
                "filter": {
                    "my_synonyms": {
                        "lenient": True,
                        "synonyms_path": "synonyms.txt",
                        "type": "synonym",
                    }
                },
            },
            "similarity": {
                "only_idf": {"script": {"source": ONLY_IDF}, "type": "scripted"}
            },
        }
    },
}
def make_query_body(surface, limit=10):
    return {
        "size": limit,
        "query": {
            "function_score": {
                "query": {"match": {"surface": {"query": surface}}},
                "functions": [{
                    "field_value_factor": {
                        "field": "score",
                        "modifier": "sqrt",
                    }
                }],
                "boost_mode": "sum"
            }
        }
    }



class ElasticSearcher(Searcher):
    def __init__(self, index, uriprefix = None, es_kwargs = None):
        from elasticsearch import Elasticsearch
        
        es_kwargs = es_kwargs or {}
        self.es = Elasticsearch(**es_kwargs)
        self.index = index
        self.uriprefix = uriprefix
        
    def search_entities(self, query: str, limit=1, add_about=False):
        body = make_query_body(query, limit=limit)
        esresults = self.es.search(index=self.index, body=body)
        
        results = []
        for hit in esresults.get('hits', []).get('hits', []):
            uri = hit.get('_source', {}).get('id')
            if self.uriprefix:
                uri = self.uriprefix + uri
            score = - hit.get('_score', 0)
            sr = SearchResult(uri, score=score)
            results.append(sr)
        return results


if __name__ == "__main__":
    import defopt, typing, tqdm, re, json, time
    from pathlib import Path

    def synonyms(redirects_label: Path):
        import urllib.parse as ul

        trans = str.maketrans("", "", ",")
        for line in Path(redirects_label).open():
            line = line.translate(trans)
            line = ul.unquote_plus(line)
            try:
                a, b = line.split("\t", 1)
                a, b = a.strip(), b.strip()
                if a and b:
                    print(a, "=>", b)
            except:
                pass

    def create(
        surfaceFormsScores: Path,
        index: str,
        chunksize: int = 10 ** 4,
        limit: int = None,
        es_kwargs: typing.Dict = None,
    ):
        from elasticsearch import Elasticsearch, helpers
        
        es_kwargs = es_kwargs or {}
        es = Elasticsearch(timeout=1000, **es_kwargs)

        es.indices.delete(index=index, ignore=[400, 404])
        es.indices.create(index=index, body=SETTINGS)

        action = {"_index": index}

        def stream(chunksize=10 ** 4, limit=None):
            with tqdm.tqdm(total=limit) as bar:
                with Path(surfaceFormsScores).open() as fo:
                    for i, line in enumerate(fo):
                        try:
                            id, surface, score = line.split("\t")
                            score = float(score)

                            yield {
                                **action,
                                "id": id,
                                "surface": surface,
                                "score": score,
                            }
                        except Exception as e:
                            pass

                        if not i % chunksize:
                            bar.update(chunksize)

                        if limit and (i > limit):
                            break

        results = helpers.parallel_bulk(es, stream(), thread_count=8)
        for i, (status, r) in enumerate(results):
            if not status:
                print("ERROR", r)

        time.sleep(1)
        print(f"Indexed {es.count(index=index).get('count')} documents")

    def search(index: str, query: str, limit: int = 1, es_kwargs: typing.Dict = None):
        es_kwargs = es_kwargs or {}
        es =  ElasticSearcher(index, es_kwargs=es_kwargs)
        return json.dumps(es.search_entities(query, limit=limit))

    r = defopt.run(
        [synonyms, create, search],
        strict_kwonly=False,
        parsers={typing.Dict: json.loads},
    )
    print(r)
