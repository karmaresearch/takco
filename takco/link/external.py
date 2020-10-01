"""
Wrappers for external linking APIs.
This module is executable. Run ``python -m takco.link.external -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
from decimal import Decimal
from datetime import datetime

from .base import Searcher, SearchResult, Database, WikiLookup

import logging as log
import typing


def get_json(url, params=None):
    import requests

    headers = {"Accept": "application/json"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.ok:
        return resp.json()


def get_redirects(url):
    import requests

    resp = requests.get(url, allow_redirects=True)
    return resp.url


from rdflib import Literal, URIRef
from rdflib.namespace import XSD, RDF
from rdflib.store import Store

XSD_date = XSD.date
RDF_type = RDF.type


class SparqlStore(Store):
    DEFAULT_PAGEQUERY = "?page schema:about ?s ."

    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",
        pagequery=None,
        configuration=None,
    ):
        self.endpoint = endpoint
        self.pagequery = self.DEFAULT_PAGEQUERY

    def _triple_query(self, query, triplepattern):
        s, p, o = triplepattern
        kv = {"s": s, "p": p, "o": o}
        bind = "".join(f"BIND ({ URIRef(v).n3() } as ?{k})" for k, v in kv.items() if v)
        q = query % bind
        results = get_json(self.endpoint, params={"format": "json", "query": q})
        return (results or {}).get("results", {}).get("bindings", [])

    def count(self, triplepattern) -> int:
        q = "select (count(*) as ?c) where { ?s ?p ?o . %s }"
        for binding in self._triple_query(q, triplepattern):
            return int(binding.get("c", {}).get("value", 0))

    def __len__(self):
        return self.count([None, None, None])

    def pages_about(
        self,
        triplepattern=(None, None, None),
        pageprefix="https://en.wikipedia.org/wiki/",
    ):
        s, p, o = triplepattern
        q = f"""
            select ?s ?page where {{
                { "?s ?p ?o ." if (p or o) else "" }
                {self.pagequery}
                FILTER(regex(str(?page), "{pageprefix}" ) )
                %s
            }}
        """
        s_pages = {}
        for binding in self._triple_query(q, triplepattern):
            s = binding.get("s", {}).get("value")
            page = binding.get("page", {}).get("value")
            if page:
                s_pages.setdefault(s, set()).add(page)
        return s_pages

    def _make_node(self, d):
        if d:
            if d.get("type") == "uri":
                return URIRef(d["value"])
            if d.get("type") == "literal":
                datatype = d.get("datatype")
                lang = d.get("xml:lang")
                return Literal(d["value"], lang=lang, datatype=datatype)

    def triples(self, triplepattern, context=None):
        q = "select ?s ?p ?o where { ?s ?p ?o . %s }"
        for binding in self._triple_query(q, triplepattern):
            yield tuple(self._make_node(binding[n]) for n in "spo"), None

    def _collect_triples(self, triplepattern):
        return [t for t in self._triples(triplepattern)]


class MediaWikiAPI(Searcher, WikiLookup):
    """A `MediaWiki API <https://www.mediawiki.org/wiki/API:Main_page>`_ endpoint.
    By default, uses the `Wikidata API <https://www.wikidata.org/w/api.php>`_.

    """

    def __init__(
        self,
        url: str = "https://www.wikidata.org/w/api.php",
        language: str = "en",
        get_json=get_json,
        ent_baseuri="http://www.wikidata.org/entity/",
        prop_baseuri="http://www.wikidata.org/prop/direct/",
        typeuri="http://www.wikidata.org/prop/direct/P31",
        log=log,
    ):
        self.url = url
        self.language = language
        self.get_json = get_json
        self.ent_baseuri = ent_baseuri
        self.prop_baseuri = prop_baseuri
        self.typeuri = typeuri
        self.log = log

    def _mainsnaks(self, claims):
        for cs in claims.values():
            for c in cs:
                if "mainsnak" in c:
                    yield c["mainsnak"]

    def _query(self, **params):
        params["format"] = "json"
        self.log.debug(f"Requesting {self.__class__.__name__} with {params}")
        return self.get_json(self.url, params=params)

    def lookup_wikititle(
        self, title: str, site: str = "enwiki", normalize: bool = True,
    ):
        """Gets the URI of a Wikibase entity based on wikipedia title.

        Args:
            title: The title of the corresponding page


        >>> MediaWikiAPI().lookup_wikititle('amsterdam')
        'http://www.wikidata.org/entity/Q727'

        """
        results = self._query(
            action="wbgetentities",
            sites=site,
            titles=title,
            normalize=int(normalize),
            props="",
            languages=self.language,
        )
        if results:
            for i in results.get("entities", []):
                if str(i) == "-1":
                    continue
                return self.ent_baseuri + i

    def snak2rdf(self, snak):
        """Convert Wikidata snak to RDF

        See also:
            - https://www.mediawiki.org/wiki/Wikibase/DataModel/JSON
            - https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format

        """
        if snak.get("snaktype") == "value":

            if snak.get("datatype") == "wikibase-item":
                return URIRef(self.ent_baseuri + snak["datavalue"]["value"]["id"])

            elif snak.get("datatype") == "monolingualtext":
                text = str(snak["datavalue"]["value"]["text"])
                lang = str(snak["datavalue"]["value"]["language"])
                return Literal(text, lang=lang)

            elif snak.get("datatype") == "quantity":
                # Ignore weird units
                amount = Decimal(snak["datavalue"]["value"]["amount"])
                unit = snak["datavalue"]["value"]["unit"]
                try:
                    unit = URIRef(unit)
                except:
                    unit = None
                return Literal(amount, datatype=unit)

            elif snak.get("datatype") == "time":
                # Ignore weird dates
                time = str(snak["datavalue"]["value"]["time"])
                return Literal(time, datatype=XSD_date)

            elif snak.get("datatype") == "globe-coordinate":
                # Ignore weird globes
                lat = str(snak["datavalue"]["value"]["latitude"])
                lon = str(snak["datavalue"]["value"]["longitude"])
                wkt = URIRef("http://www.opengis.net/ont/geosparql#wktLiteral")
                return Literal(f"POINT({lon} {lat})", datatype=wkt)

            else:
                return Literal(str(snak["datavalue"]["value"]))

    def get_claims(self, *ids: str, mainsnak: bool = True):
        """Gets the claims for multiple Wikibase entities.

        Args:
            ids: The IDs of the entities to get the data from
        """
        if ids:
            results = self._query(
                action="wbgetentities", ids="|".join(ids), props="claims",
            )
            if results:
                ent_claims = {}
                for k, v in results.get("entities", {}).items():
                    claims = v.get("claims")
                    if mainsnak:
                        claims = list(self._mainsnaks(claims))
                    p_vclaims = {}
                    for claim in claims:
                        p = self.prop_baseuri + claim.pop("property")
                        claim = self.snak2rdf(claim)
                        if claim:
                            p_vclaims.setdefault(p, []).append(claim)
                    ent_claims[k] = p_vclaims
                return ent_claims

    def search_entities(
        self,
        search: str,
        context=(),
        limit: int = 1,
        add_about: bool = False,
        mainsnak: bool = True,
    ) -> typing.List[SearchResult]:
        """Searches for entities using labels and aliases.

        .. role:: pre

        See also:

            - |wbsearchentities|_

        .. |wbsearchentities| replace:: ``wbsearchentities``
        .. _wbsearchentities: https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities

        """
        if not search:
            return {}
        results = self._query(
            action="wbsearchentities",
            search=search,
            limit=limit,
            language=self.language,
        )
        if results:
            results = results.get("search", [])

            if add_about:
                ids = set(s["id"] for s in results)
                ent_claims = self.get_claims(*ids, mainsnak=mainsnak)
                for s in results:
                    s.update(ent_claims.get(s["id"], {}))
                    s[RDF_type] = s.get(self.typeuri, [])

            return [SearchResult(r.pop("concepturi"), r) for r in results]
        else:
            log.debug(f"No {self.__class__.__name__} results for {search}")


class DBpediaLookup(Searcher, WikiLookup):
    """ A `DBpedia Lookup <https://wiki.dbpedia.org/lookup>`_ instance."""

    def __init__(
        self,
        url: str = "http://lookup.dbpedia.org/api/search.asmx/KeywordSearch",
        language: str = "en",
        get_json=get_json,
        log=log,
    ):
        self.url = url
        self.language = language
        self.get_json = get_json
        self.log = log

    def _query(self, **params):
        self.log.debug(f"Requesting {self.__class__.__name__} with {params}")
        return self.get_json(self.url, params=params)

    def lookup_wikititle(self, title: str) -> str:
        """Gets the URI for a DBpedia entity based on wikipedia title."""
        title = title.replace(" ", "_")
        redir = get_redirects(f"http://dbpedia.org/page/{title}")
        return str(redir).replace("/page/", "/resource/")

    def search_entities(
        self, search: str, context=(), limit: int = 1, **kwargs,
    ) -> typing.List[SearchResult]:
        """Searches for entities using the Keyword Search API.
        The Keyword Search API can be used to find related DBpedia resources for a
        given string. The string may consist of a single or multiple words.

        """
        if not search:
            return {}
        results = self._query(
            QueryString=search,
            MaxHits=limit or 1000,
            #   language=self.language,
        )
        if results:
            sr = []
            for r in results.get("results", []):
                if "classes" in r:
                    r[RDF_type] = r.pop("classes")
                score = 1 - (1 / (1 + r.get("refCount", 0)))
                sr.append(SearchResult(r.pop("uri"), r, score=score))
            return sr


if __name__ == "__main__":
    import defopt, typing, json, enum

    class Searchers(enum.Enum):
        mw = MediaWikiAPI
        db = DBpediaLookup

    def test(
        kind: Searchers, query: str, limit: int = 1, add_about: bool = False,
    ):
        """Search for entities

        Args:
            kind: Searcher (mw=MediaWikiAPI, db=DBpediaLookup)
            query: Query string
            limit: Limit results
        """
        result = kind.value().search_entities(
            query, context=(), limit=limit, add_about=add_about
        )
        return result

    funcs = [test]

    r = defopt.run(funcs, strict_kwonly=False, show_types=True)
    print(json.dumps(r))
