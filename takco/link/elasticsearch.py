"""
This module is executable. Run ``python -m takco.link.elasticsearch -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
import logging as log
from pathlib import Path
import re
import datetime
import itertools
from dataclasses import dataclass, field

from .base import (
    Searcher,
    SearchResult,
    Typer,
    Lookup,
)
from .rdf import GraphDB, URIRef, Literal
from .types import SimpleTyper
from .integrate import NaryDB, NaryMatchResult, QualifierMatchResult

try:
    from elasticsearch import Elasticsearch
except:
    log.warn(f"Failed to load Elasticsearch")

ONLY_IDF = (
    "double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0;"
    "double norm = 1/Math.sqrt(doc.length);"
    "return query.boost * idf * norm;"
)
SETTINGS = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "type": {"type": "keyword"},
            "context.value": {"type": "text", "norms": False},
            "refs": {"type": "integer"},
            "surface": {
                "type": "nested",
                "properties": {
                    "value": {
                        "type": "text",
                        "analyzer": "standard",
                        "similarity": "only_idf",
                    },
                    "score": {"type": "float"},
                },
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
QUERY_SCRIPT = """
{
    "query": {
        "function_score": {
            "query": {
                "bool": {
                    "must": {
                        "nested": {
                            "path": "surface",
                            "score_mode": "max",
                            "query": {
                                "function_score": {
                                    "query": {
                                        "dis_max": {
                                            "queries": [
                                                {
                                                    "match_phrase": {
                                                        "surface.value": {
                                                            "query": "{{query}}"
                                                        }
                                                    }
                                                },
                                                {
                                                    "match_phrase": {
                                                        "surface.value": {
                                                            "query": "{{query}}",
                                                            "boost": 0.5,
                                                            "slop": 2
                                                        }
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    "functions": [
                                        {
                                            "field_value_factor": {
                                                "field": "surface.score",
                                                "modifier": "sqrt"
                                            }
                                        }
                                    ],
                                    "boost_mode": "multiply",
                                    "boost": "{{query_boost}}{{^query_boost}}2{{/query_boost}}"
                                }
                            }
                        }
                    },
                    "should": [
                        {{#context}}
                            {"match_phrase": {"context.value": {"query": "{{value}}", "boost": 0.2}}},
                            {"match": {"context.value": {"query": "{{value}}", "boost": 0.2}}},
                        {{/context}}
                        {{#classes}}
                            {"term": {"type": {"value":"{{value}}", "boost":4}}},
                        {{/classes}}
                        {"match_all":{}}
                    ]
                }
            },
            "functions": [
                {"field_value_factor": {"field": "refs", "modifier": "log1p", "missing": 0}}
            ],
            "boost_mode": "sum"
        }
    },
    "size": "{{limit}}{{^limit}}1{{/limit}}"
}
"""


@dataclass
class ElasticSearcher(Searcher):
    index: str
    baseuri: typing.Dict = field(default_factory=dict)
    propbaseuri: typing.Dict = field(default_factory=dict)
    es_kwargs: typing.Dict = field(default_factory=dict)
    parts: bool = False
    prop_uri: typing.Dict[str, typing.Dict] = field(default_factory=dict)
    prop_baseuri: typing.Dict = field(default_factory=dict)
    typer: Typer = SimpleTyper()
    stringmatch: str = "jaccard"
    NUM = re.compile("^[\d\W]+$")

    def __enter__(self):
        self.es = Elasticsearch(**self.es_kwargs)
        return self

    def _about(self, source):
        about = {}
        for k, vs in list(source.items()):
            baseuri = self.prop_baseuri.get(k, "")
            k = self.prop_uri.get(k, k)
            if isinstance(vs, list):
                about[k] = [(baseuri + v if isinstance(v, str) else v) for v in vs]
        return about

    def get_parts(self, query):
        for char in "([,:":
            for qpart in query.split(char):
                qpart = qpart.translate(str.maketrans("", "", ")]")).strip()
                if qpart != query and not qpart.isnumeric():
                    yield qpart

    def make_query_body(self, query, **kwargs):
        return {"id": "query", "params": {"query": query, **kwargs}}

    def search_entities(
        self, query_params, limit=1, add_about=False, ispart=False,
    ):
        # Simplify classes
        query_params = tuple(query_params)
        if not query_params:
            return

        it = iter(query_params)
        while True:
            query_chunk = tuple(itertools.islice(it, 10 ** 3))  # chunk per 1000
            if not query_chunk:
                break
        
            log.debug(f"Submitting ES multiquery of size {len(query_chunk)}")

            bodies = []
            for query, params in query_chunk:
                context, classes = params.get("context", []), params.get("classes", [])
                context = [{"value": c} for c in (context or []) if not self.NUM.match(c)]
                classes = [{"value": c.split("/")[-1]} for c in (classes or [])]

                body = self.make_query_body(
                    query, context=context, classes=classes, limit=limit
                )
                bodies.append(())
                bodies.append(body)

            esresponses = self.es.msearch_template(index=self.index, body=bodies).get(
                "responses", []
            )
            for (query, params), esresponse in zip(query_chunk, esresponses):
                context, classes = params.get("context", []), params.get("classes", [])
                results = []
                for hit in esresponse.get("hits", {}).get("hits", []):
                    uri = hit.get("_source", {}).get("id")
                    if self.baseuri:
                        uri = self.baseuri + uri
                    score = hit.get("_score", 0)
                    about = self._about(hit.get("_source", {}))

                    context_matches = {}
                    if ("context" in about) and isinstance(context, dict):
                        for ec in about["context"]:
                            prop = ec.get("prop")
                            if self.propbaseuri:
                                prop = self.propbaseuri + prop
                            vals = ec.get("value", [])
                            for v in vals if isinstance(vals, list) else [vals]:
                                for c, csource in context.items():
                                    for m in self.typer.literal_match(
                                        v, c, self.stringmatch
                                    ):
                                        pms = context_matches.setdefault(csource, {})
                                        pms.setdefault(prop, []).append(m)

                    sr = SearchResult(
                        uri, about, context_matches=context_matches, score=score
                    )
                    results.append(sr)

                if not results:
                    # log.debug(f"No {self} results for {query}")
                    if self.parts and (not ispart):
                        partqueries = [(p, context) for p in self.get_parts(query)]
                        more = self.search_entities(
                            partqueries, limit=limit, add_about=add_about, ispart=True,
                        )
                        for srs in more:
                            results += srs

                yield results

    @classmethod
    def create(
        cls,
        input: Path = None,
        format: str = "ttl",
        surfaces: Path = None,
        refcounts: Path = None,
        es_index: str = None,
        es_kwargs: typing.Dict = None,
        recreate: bool = True,
        thread_count: int = 8,
        baseuris: typing.List[str] = (),
        uri_prefLabel: str = "http://www.w3.org/2004/02/skos/core#prefLabel",
        uri_altLabel: str = "http://www.w3.org/2004/02/skos/core#altLabel",
        uri_type: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        lang: str = "en",
        extract_surface: bool = False,
        uri_ignore: str = None,
        surface_norm: str = "[^\s\w]",
        context_label_threshold: float = 1,
        output_statements: bool = True,
    ):
        import re

        baseuris_re = [re.compile(b) for b in baseuris]
        uri_ignore_re = re.compile(uri_ignore) if uri_ignore else None
        ignore_uri = lambda uri: uri_ignore_re and uri_ignore_re.match(uri)

        surface_norm_re = re.compile(surface_norm)

        def normalize_surface(s):
            if isinstance(s, dict):
                s = "".join(s.values())
            return surface_norm_re.sub(" ", s.replace("_", " ").lower()).strip()

        def debase(uri):
            for baseuri in baseuris_re:
                uri = baseuri.sub("", uri)
            return uri

        id_surfaceformscores = {}
        if surfaces and surfaces.exists():
            import tqdm, json, sys
            import urllib.parse as ul

            for line in tqdm.tqdm(open(surfaces), desc="Loading surface forms"):
                try:
                    id, surf = line.split("\t", 1)
                    id = debase(ul.unquote_plus(id))
                    id_surfaceformscores[id] = json.loads(surf)
                except Exception as e:
                    print(e, file=sys.stderr)

        id_refcount = {}
        if refcounts and refcounts.exists():
            import tqdm, json, sys
            import urllib.parse as ul

            for line in tqdm.tqdm(open(refcounts), desc="Loading refcounts"):
                try:
                    id, n = line.split("\t", 1)
                    id = debase(ul.unquote_plus(id))
                    id_refcount[id] = int(n)
                except Exception as e:
                    print(e, file=sys.stderr)

        def parse_n3(node, base=False):
            import urllib.parse as ul

            if node:
                if node[0] == "<" and node[-1] == ">":
                    uri = ul.unquote_plus(node[1:-1])
                    return {"id": uri if base else debase(uri)}
                elif '"^^' in node:
                    val, dtype = node.rsplit("^^", 1)
                    if dtype.endswith(">"):
                        dtype = ul.unquote_plus(dtype[1:-1])
                    return {(dtype if base else debase(dtype)): val[1:-1]}
                elif '"@' in node:
                    val, l = node.rsplit("@", 1)
                    if l == lang:
                        return {"str": val[1:-1]}
                elif node[0] == '"' and node[-1] == '"':
                    return {"str": node[1:-1]}

        def parse_ttl(fname):
            for line in open(fname):
                try:
                    s_n3, claims = line.split(None, 1)
                    id = parse_n3(s_n3)["id"]
                    statements = []
                    surface_score = {}
                    for claim in set(claims.rstrip().rstrip(".").rstrip().split(" ; ")):
                        p, o = claim.split(None, 1)
                        p, o = parse_n3(p)["id"], parse_n3(o)
                        if " " in p:
                            continue
                        if any(o.values()):
                            if p == uri_prefLabel and ("str" in o):
                                surface_score[normalize_surface(o["str"])] = 1
                            elif p == uri_altLabel and ("str" in o):
                                surface_score[normalize_surface(o["str"])] = 0.5
                            else:
                                statements.append({"prop": p, **o})
                    yield id, surface_score, statements
                except Exception as e:
                    pass  # raise e

        def parse_json(fname):
            for line in open(fname):
                try:
                    if line[0] == "[":
                        continue
                    doc = json.loads(line if line[-2] != "," else line[:-2])
                    label = doc.get("labels", {}).get(lang, {}).get("value")
                    surface_score = {}
                    if label:
                        surface_score[normalize_surface(label)] = 1
                    for alias in doc.get("aliases", {}).get(lang, []):
                        if alias.get("value"):
                            surface_score[normalize_surface(alias["value"])] = 0.5
                    yield doc.get("id"), surface_score, ElasticDB._wd_statements(doc)
                except Exception as e:
                    raise e

        def stream():
            lines = parse_ttl(input) if format == "ttl" else parse_json(input)
            for id, surface_score, statements in lines:
                if ignore_uri(id):
                    continue

                if extract_surface:
                    surface_score[normalize_surface(id)] = 1

                types = set()
                prop_context = {}
                filtered_statements = []
                for st in statements:
                    if ignore_uri(st.get("id", "")):
                        continue

                    if st.get("prop") == uri_type and ("id" in st):
                        types.add(st["id"])
                    else:
                        filtered_statements.append(st)
                        if "id" in st and "prop" in st:
                            vals = set()
                            if extract_surface:
                                vals.add(normalize_surface(st["id"]))
                            for l, ls in id_surfaceformscores.get(st["id"], {}).items():
                                if ls >= context_label_threshold:
                                    vals.add(l)
                            if vals:
                                prop_context.setdefault(st["prop"], set()).update(vals)
                context = [
                    {"prop": p, "value": list(vs)} for p, vs in prop_context.items()
                ]

                surface_score.update(id_surfaceformscores.get(id, {}))

                if not surface_score:
                    continue

                yield {
                    "id": id,
                    "type": list(types),
                    "surface": [
                        {"value": l, "score": c} for l, c in surface_score.items()
                    ],
                    **(
                        {"statements": filtered_statements} if output_statements else {}
                    ),
                    "context": context,
                    **({"refs": id_refcount[id]} if id in id_refcount else {}),
                }

        if es_index:
            from elasticsearch import Elasticsearch, helpers
            import time, sys

            es_kwargs = es_kwargs or {}
            es = Elasticsearch(timeout=1000, **es_kwargs)

            if recreate:
                cls.init_index(es_index=es_index, es_kwargs=es_kwargs)

            results = helpers.parallel_bulk(
                es,
                ({"_index": es_index, **d} for d in stream()),
                thread_count=thread_count,
            )
            for i, (status, r) in enumerate(results):
                if not status:
                    print("ERROR", r, file=sys.stderr)

            time.sleep(1)
            print(
                f"Indexed {es.count(index=es_index).get('count')} documents",
                file=sys.stderr,
            )
        else:
            import json

            for doc in stream():
                print(json.dumps(doc))

    @classmethod
    def test(
        cls, index: str, *query: str, limit: int = 1, es_kwargs: typing.Dict = None
    ):
        """Search an Elasticsearch index for a query string """
        import json

        es_kwargs = es_kwargs or {}
        with cls(index, es_kwargs=es_kwargs) as es:
            queries = [(q, ()) for q in query]
            for es in es.search_entities(queries, limit=limit):
                for e in es:
                    print(json.dumps(e))

    @classmethod
    def store_template(cls, es_kwargs: typing.Dict = None):
        es_kwargs = es_kwargs or {}
        body = {"script": {"lang": "mustache", "source": QUERY_SCRIPT}}
        host = es_kwargs.get("host", "localhost")
        port = es_kwargs.get("port", "9200")
        import requests

        return requests.post(f"http://{host}:{port}/_scripts/query", json=body).text

    @classmethod
    def init_index(cls, es_index: str, es_kwargs: typing.Dict = ()):
        import sys

        es_kwargs = dict(es_kwargs)
        es = Elasticsearch(timeout=1000, **es_kwargs)
        es.indices.delete(index=es_index, ignore=[400, 404])
        print("Creating index...", file=sys.stderr)
        es.indices.create(index=es_index, body=SETTINGS)
        print(cls.store_template(es_kwargs=es_kwargs))


class ElasticDB(ElasticSearcher, GraphDB, NaryDB, Lookup):
    INIT = {
        "mappings": {
            "properties": {
                "statements": {
                    "type": "nested",
                    "properties": {"qualifiers": {"type": "nested"}},
                }
            }
        }
    }

    def __init__(self, typer: Typer = SimpleTyper(), cache=False, *args, **kwargs):
        self.typer = typer
        self.cache = {} if cache else ()
        ElasticSearcher.__init__(self, *args, **kwargs)
        GraphDB.__init__(self)

    def _hit_triples(self, hit):
        s = self._tonode(hit.get("_source", {}).get("id"), self.baseuri)
        for st in hit.get("_source", {}).get("statements", []):
            p = self._tonode(st.pop("prop"), self.propbaseuri)
            if "id" in st:
                o = self._tonode(st.pop("id"), self.baseuri)
                yield s, p, o
            # todo literals

    def _tonode(self, n, baseuri):
        if baseuri and (not str(n).startswith(baseuri)):
            return URIRef(baseuri + str(n))
        else:
            return URIRef(str(n))

    def _fromnode(self, n, baseuri):
        if baseuri and str(n).startswith(baseuri):
            return str(n).replace(baseuri, "")
        else:
            return str(n)

    def get_prop_values(self, e, p):
        # it's faster for ES to ignore p first
        return self.about(e).get(self._tonode(p, self.propbaseuri), [])

    def _triple_body(self, pattern):
        s, p, o = pattern
        must = []
        nest = []
        if s:
            s_id = self._fromnode(s, self.baseuri)
            must.append({"match": {"id": s_id}})
        if p:
            p_id = self._fromnode(p, self.propbaseuri)
            nest.append({"match": {"statements.prop": p_id}})
        if o:
            o_id = self._fromnode(o, self.baseuri)
            nest.append({"match": {"statements.id": o_id}})

        if nest:
            must.append(
                {"nested": {"path": "statements", "query": {"bool": {"must": nest}}}}
            )
        if must:
            return {"query": {"bool": {"must": must}}}

    def _nary_body(self, e1, e2):
        return {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"id": self._fromnode(e1, self.baseuri)}},
                        {
                            "nested": {
                                "path": "statements",
                                "query": {
                                    "match": {
                                        "statements.id": self._fromnode(
                                            e2, self.baseuri
                                        )
                                    }
                                },
                            }
                        },
                    ]
                }
            }
        }

    def triples(self, triplepattern, **kwargs):
        triplepattern = tuple(triplepattern)
        if triplepattern in self.cache:
            return self.cache[triplepattern]
        s, p, o = triplepattern
        body = self._triple_body(triplepattern)
        try:
            result = self.es.search(index=self.index, body=body)
        except Exception as e:
            result = {}
            log.error(e)

        for hit in result.get("hits", {}).get("hits", []):
            for s_, p_, o_ in self._hit_triples(hit):
                s_ok = (not s) or (str(s) == str(s_))
                p_ok = (not p) or (str(p) == str(p_))
                o_ok = (not o) or (str(o) == str(o_))
                if s_ok and p_ok and o_ok:
                    triple = s_, p_, o_
                    yield triple
                    if self.cache is not ():
                        self.cache.setdefault(triplepattern, set()).add(triple)

    def count(self, triplepattern):
        body = self._triple_body(triplepattern)
        try:
            result = self.es.count(index=self.index, body=body)
        except Exception as e:
            result = {}
            log.error(e)
        return result.get("count", 0)

    def __len__(self):
        return 0

    def lookup_title(self, title):
        title = title.replace(" ", "_")
        body = {"query": {"match_phrase": {"wiki": title}}}
        res = self.es.search(index=self.index, body=body)
        for hit in res.get("hits", {}).get("hits", []):
            return self._tonode(hit.get("_source", {}).get("id"), self.baseuri)

    def get_rowfacts(self, celltexts, entsets):
        for ci1, ents1 in enumerate(entsets):
            for ci2, ents2 in enumerate(entsets):
                if ci1 == ci2:
                    pass

                for e1, e2 in ((e1, e2) for e1 in ents1 for e2 in ents2 if e1 != e2):
                    body = self._nary_body(e1, e2)
                    try:
                        result = self.es.search(index=self.index, body=body)
                    except Exception as e:
                        result = {}
                        log.error(e)

                    for hit in result.get("hits", {}).get("hits", []):
                        for st in hit.get("_source", {}).get("statements", []):
                            if not st.get("id"):
                                continue

                            id = self._tonode(st.get("id"), self.baseuri)
                            if str(id) != str(e2):
                                continue

                            mainprop = st.get("prop")
                            mainprop = self._tonode(mainprop, self.propbaseuri)
                            qmatches = []
                            for q in st.get("qualifiers", []):
                                p = self._tonode(q.get("prop"), self.propbaseuri)

                                if "id" in q:
                                    o = q.get("id")
                                    for ci, es in enumerate(entsets):
                                        if o in es:
                                            qm = QualifierMatchResult(
                                                ci, (None, p, o), None
                                            )
                                            qmatches.append(qm)
                                else:
                                    for dt in ["str", "decimal", "dateTime"]:
                                        if dt not in q:
                                            continue
                                        o = q[dt]
                                        if dt == "decimal":
                                            o = float(o)
                                        if dt == "dateTime":
                                            o = datetime.datetime.fromisoformat(o[:-1])

                                        o = Literal(o)
                                        for ci, txt in enumerate(celltexts):
                                            for lm in self.typer.literal_match(o, txt):
                                                qm = QualifierMatchResult(
                                                    ci, (None, p, o), lm
                                                )
                                                qmatches.append(qm)

                            yield NaryMatchResult(
                                (ci1, ci2), (e1, mainprop, e2), qmatches
                            )

    @staticmethod
    def _wd_att(snak):
        att = {}
        val = snak.get("datavalue", {}).get("value", {})
        if isinstance(val, dict):
            if "id" in val:
                att["id"] = val.get("id")
            if "time" in val:
                att["dateTime"] = val.get("time")
            if "amount" in val:
                att["decimal"] = val.get("amount")
        else:
            att["str"] = val
        if att:
            att["prop"] = snak.get("property")
            return att

    @classmethod
    def _wd_statements(cls, doc):
        for claims in doc.get("claims", {}).values():
            for claim in claims:
                st = cls._wd_att(claim.get("mainsnak", {}))
                if st:
                    qualifiers = []
                    for qs in claim.get("qualifiers", {}).values():
                        for q in qs:
                            qualifiers.append(cls._wd_att(q))
                    if qualifiers:
                        st["qualifiers"] = qualifiers
                    yield st

    @classmethod
    def db_docs(cls, file: Path):
        """Convert Wikidata entity documents to minimalist ES documents"""
        import json

        for line in Path(file).open():
            line = line.strip()
            if line[0] == "[":
                continue
            try:
                if line[-1] == ",":
                    line = line[:-1]

                indoc = json.loads(line)
                doc = {"id": indoc.get("id")}
                statements = [cls._wd_statements(indoc)]

                if statements:
                    doc["statements"] = statements
                print(json.dumps(doc))
            except Exception as e:
                raise e


if __name__ == "__main__":
    import defopt, json, os

    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL", "WARN")))

    r = defopt.run(
        [
            ElasticSearcher.create,
            ElasticSearcher.test,
            ElasticSearcher.store_template,
            ElasticSearcher.init_index,
            ElasticDB.db_docs,
        ],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
