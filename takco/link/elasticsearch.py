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

from .base import (
    Searcher,
    SearchResult,
    NaryDB,
    CellType,
    MatchResult,
    QualifierMatchResult,
    WikiLookup,
)
from .rdf import GraphDB, URIRef, Literal
from .types import SimpleCellType

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
            "context": {"type": "text", "norms": False},
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


def make_query_body(q, context=(), classes=(), limit=1, lenient=False):

    classes = [c.split("/")[-1] for c in classes]

    return {
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
                                        "query": (
                                            {"match": {"surface.value": q}}
                                            if lenient
                                            else {"match_phrase": {"surface.value": q}}
                                        ),
                                        "functions": [
                                            {
                                                "field_value_factor": {
                                                    "field": "surface.score",
                                                    "modifier": "sqrt",
                                                },
                                            },
                                        ],
                                        "boost_mode": "multiply",
                                        "boost": 2,
                                    },
                                },
                            },
                        },
                        "should": [
                            {"match_phrase": {"context": {"query": c, "boost": 0.2}}}
                            for c in context
                        ]
                        + [
                            {"match": {"context": {"query": c, "boost": 0.2}}}
                            for c in context
                        ]
                        + [{"term": {"type": t}} for t in classes],
                    },
                },
                "functions": [
                    {"field_value_factor": {"field": "refs", "modifier": "log1p",}}
                ],
                "boost_mode": "sum",
            },
        },
        "size": limit,
    }


class ElasticSearcher(Searcher):
    NUM = re.compile("^[\d\W]+$")

    def __init__(
        self,
        index,
        baseuri=None,
        propbaseuri=None,
        es_kwargs=None,
        parts=True,
        prop_uri=None,
        prop_baseuri=None,
        **_,
    ):
        from elasticsearch import Elasticsearch

        es_kwargs = es_kwargs or {}
        self.es = Elasticsearch(**es_kwargs)
        self.index = index
        self.baseuri = baseuri
        self.propbaseuri = propbaseuri or baseuri
        self.parts = parts
        self.prop_uri = prop_uri or {}
        self.prop_baseuri = prop_baseuri or {}

    def _about(self, source):
        about = {}
        for k, vs in list(source.items()):
            baseuri = self.prop_baseuri.get(k, "")
            k = self.prop_uri.get(k, k)
            if isinstance(vs, list):
                about[k] = [(baseuri + v if isinstance(v, str) else v) for v in vs]
        return about

    def search_entities(
        self,
        query: str,
        context=(),
        classes=(),
        limit=1,
        add_about=False,
        lenient=False,
        ispart=False,
    ):

        context = [c for c in context if not self.NUM.match(c)]
        body = make_query_body(
            query, context=context, classes=classes, limit=limit, lenient=lenient
        )
        esresults = self.es.search(index=self.index, body=body)

        results = []
        for hit in esresults.get("hits", []).get("hits", []):
            uri = hit.get("_source", {}).get("id")
            if self.baseuri:
                uri = self.baseuri + uri
            score = hit.get("_score", 0)
            about = self._about(hit.get("_source", {}))

            sr = SearchResult(uri, about, score=score)
            results.append(sr)

        if not results:
            log.debug(f"No {self} results for {query}")
            if self.parts and (not ispart):
                for char in "([,:":
                    for qpart in query.split(char):
                        qpart = qpart.translate(str.maketrans("", "", ")]")).strip()
                        if qpart != query:
                            more = self.search_entities(
                                qpart,
                                context=context,
                                limit=limit,
                                add_about=add_about,
                                ispart=True,
                            )
                            if more:
                                return more
                            else:
                                log.debug(f"No {self} results for {qpart}")

        if (not results) and (not lenient) and (not ispart):
            log.debug(f"No {self} STRICT results for {query}")
            return self.search_entities(
                query, context=context, limit=limit, add_about=add_about, lenient=True,
            )

        return results

    def about(self, uri: str):
        id = str(uri).replace(self.baseuri, "")
        esresults = self.es.search(
            index=self.index, body={"query": {"match": {"id": id}}}
        )
        for h in esresults.get("hits", []).get("hits", []):
            return h.get("_source", {})

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
                    yield a, b
            except:
                pass

    @classmethod
    def yield_doc(
        cls,
        s,
        db,
        baseuris,
        prefLabel,
        altLabel=None,
        p_type=None,
        ent_surface_scores=None,
        unescape=False,
        normalize_scores=True,
        get_synonyms=None,
    ):
        ent_surface_scores = ent_surface_scores or {}
        plabel_score = {prefLabel: 1}
        if altLabel is not None:
            plabel_score[altLabel] = 0.5
        id = db.lookup_str(s)[1:-1]
        for baseuri in baseuris:
            id = id.replace(baseuri, "")

        surface_score = ent_surface_scores.get(id, {})
        if normalize_scores and surface_score:
            top = max(surface_score.values())
            surface_score = {sur: score / top for sur, score in surface_score.items()}

        for plabel, score in plabel_score.items():
            for l in db.o(s, plabel):
                label = db.lookup_str(l)
                if unescape:
                    label = label.encode().decode("unicode-escape")
                if label.endswith("@en"):
                    label = label[1:-4].lower()
                elif label.strip().endswith('"'):
                    label = label[1:-1].lower()
                else:
                    continue

                labels = get_synonyms(label) if get_synonyms else [label]
                for label in labels:
                    surface_score.setdefault(label, score)

        if surface_score:
            context = set()
            for _, o in db.po(s):
                for l in db.o(o, prefLabel):
                    label = db.lookup_str(l)
                    if unescape:
                        label = label.encode().decode("unicode-escape")
                    if label.endswith("@en"):
                        context.add(label[1:-4])
                    elif label.strip().endswith('"'):
                        context.add(label[1:-1])

            types = set()
            if p_type is not None:
                for t in db.o(s, p_type):
                    t = db.lookup_str(t)[1:-1]
                    for baseuri in baseuris:
                        t = t.replace(baseuri, "")
                    types.add(t)

            yield {
                "id": id,
                "type": list(types),
                "surface": [{"value": l, "score": c} for l, c in surface_score.items()],
                "context": list(context),
                "refs": db.count_o(s),
            }

    @classmethod
    def docs(
        cls,
        trident_path: Path,
        baseuris: typing.List[str] = (),
        uri_prefLabel: str = "http://www.w3.org/2004/02/skos/core#prefLabel",
        uri_altLabel: str = "http://www.w3.org/2004/02/skos/core#altLabel",
        uri_type: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        unescape: bool = False,
        ntask: int = None,
        tasktotal: int = None,
    ):
        import trident, tqdm, json

        db = trident.Db(str(trident_path))
        prefLabel = db.lookup_id(f"<{uri_prefLabel}>")
        altLabel = db.lookup_id(f"<{uri_altLabel}>")
        p_type = db.lookup_id(f"<{uri_type}>")
        assert (prefLabel is not None) or (altLabel is not None)

        for s, _ in tqdm.tqdm(db.os(prefLabel)):
            if ntotal and ((s % (tasktotal + 1)) != ntask):
                continue
            docs = cls.yield_doc(
                s, db, baseuris, prefLabel, altLabel, p_type, unescape=unescape
            )
            for doc in docs:
                print(json.dumps(doc))

    @classmethod
    def create(
        cls,
        trident_path: Path,
        index: str,
        surfaceFormsScores: Path = None,
        normalize_scores: bool = True,
        synonym_file: Path = None,
        recreate: bool = True,
        chunksize: int = 10 ** 4,
        limit: int = None,
        thread_count: int = 8,
        baseuris: typing.List[str] = (),
        uri_prefLabel: str = "http://www.w3.org/2004/02/skos/core#prefLabel",
        uri_altLabel: str = "http://www.w3.org/2004/02/skos/core#altLabel",
        uri_type: str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        unescape: bool = False,
        es_kwargs: typing.Dict = None,
    ):
        """Create Elasticsearch index from surfaceFormsScores file
        and optional trident db
        """
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
            import trident

            db = trident.Db(str(trident_path))
            prefLabel = db.lookup_id(f"<{uri_prefLabel}>")
            altLabel = db.lookup_id(f"<{uri_altLabel}>")
            p_type = db.lookup_id(f"<{uri_type}>")
            assert (prefLabel is not None) or (altLabel is not None)

            ent_surface_scores = {}
            if surfaceFormsScores:
                print("Loading surface forms from", surfaceFormsScores, file=sys.stderr)
                with tqdm.tqdm(total=limit) as bar:
                    with Path(surfaceFormsScores).open() as fo:
                        for i, line in enumerate(fo):
                            try:
                                ent, surface, score = line.split("\t")
                                score = float(score)
                                surface = surface.encode().decode("unicode-escape")
                                for val in get_synonyms(surface):
                                    ss = ent_surface_scores.setdefault(ent, {})
                                    ss[val] = max(ss.get(val, 0), score)
                            except Exception as e:
                                pass

                            if not i % chunksize:
                                bar.update(chunksize)

            for s in tqdm.tqdm(db.all_s()):
                if db.count_s(s):

                    docs = cls.yield_doc(
                        s,
                        db,
                        baseuris,
                        prefLabel,
                        altLabel,
                        p_type,
                        ent_surface_scores,
                        unescape,
                        normalize_scores=normalize_scores,
                        get_synonyms=get_synonyms,
                    )
                    for doc in docs:
                        yield {**action, **doc}

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


class ElasticDB(ElasticSearcher, GraphDB, NaryDB, WikiLookup):
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

    def __init__(
        self, cellType: CellType = SimpleCellType, cache=False, *args, **kwargs
    ):
        self.cellType = cellType
        self.cache = {} if cache else None
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
        if self.cache is not None:
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
                    if self.cache is not None:
                        self.cache.setdefault(triplepattern, set()).add(triple)

    def count(self, triplepattern):
        s, p, o = triplepattern
        body = self._triple_body(triplepattern)
        try:
            result = self.es.count(index=self.index, body=body)
        except Exception as e:
            result = {}
            log.error(e)
        return result.get("count", 0)

    def __len__(self):
        return 0

    def lookup_wikititle(self, title):
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
                                            for lm in self.cellType.literal_match(
                                                o, txt
                                            ):
                                                qm = QualifierMatchResult(
                                                    ci, (None, p, o), lm
                                                )
                                                qmatches.append(qm)

                            yield MatchResult((ci1, ci2), (e1, mainprop, e2), qmatches)

    @staticmethod
    def _wd_att(snak):
        att = {"prop": snak.get("property")}
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
        return att

    @classmethod
    def db_docs(cls, file: Path):
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
                statements = []
                for claims in indoc.get("claims", {}).values():
                    for claim in claims:
                        st = cls._wd_att(claim.get("mainsnak", {}))
                        qualifiers = []
                        for qs in claim.get("qualifiers", {}).values():
                            for q in qs:
                                qualifiers.append(cls._wd_att(q))
                        if qualifiers:
                            st["qualifiers"] = qualifiers
                        statements.append(st)
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
            ElasticSearcher.docs,
            ElasticDB.db_docs,
        ],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
