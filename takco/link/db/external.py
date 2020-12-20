"""
Wrappers for external linking APIs.
This module is executable. Run ``python -m takco.link.external -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
from decimal import Decimal

from ..base import Searcher, SearchResult, Database, Lookup

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

try:
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
except Exception as e:
    log.warn(e)


class MediaWikiAPI(Searcher, Lookup):
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

    def lookup_title(
        self, title: str, site: str = "enwiki", normalize: bool = True,
    ):
        """Gets the URI of a Wikibase entity based on wikipedia title.

        Args:
            title: The title of the corresponding page


        >>> MediaWikiAPI().lookup_title('amsterdam')
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
                    p_vclaims: typing.Dict = {}
                    for claim in claims:
                        p = self.prop_baseuri + claim.pop("property")
                        claim = self.snak2rdf(claim)
                        if claim:
                            p_vclaims.setdefault(p, []).append(claim)
                    ent_claims[k] = p_vclaims
                return ent_claims

    def search_entities(
        self,
        query_contexts,
        limit: int = 1,
        add_about: bool = False,
        mainsnak: bool = True,
    ):
        """Searches for entities using labels and aliases.

        .. role:: pre

        See also:

            - |wbsearchentities|_

        .. |wbsearchentities| replace:: ``wbsearchentities``
        .. _wbsearchentities: https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities

        """
        resultsets: typing.List[typing.List[SearchResult]] = []
        for search, _ in query_contexts:
            if not search:
                resultsets.append( [] )
                continue
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

                rs = [SearchResult(r.pop("concepturi"), r) for r in results]
                resultsets.append( rs )
            else:
                log.debug(f"No {self.__class__.__name__} results for {search}")
                resultsets.append( [] )
        return resultsets


class DBpediaLookup(Searcher, Lookup):
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
        params = {**params, "format": "json"}
        self.log.debug(f"Requesting {self.__class__.__name__} with {params}")
        return self.get_json(self.url, params=params)

    def lookup_title(self, title: str) -> str:
        """Gets the URI for a DBpedia entity based on wikipedia title."""
        title = title.replace(" ", "_")
        redir = get_redirects(f"http://dbpedia.org/page/{title}")
        return str(redir).replace("/page/", "/resource/")

    def search_entities(
        self, query_contexts, limit: int = 1, **kwargs,
    ):
        """Searches for entities using the Keyword Search API.
        The Keyword Search API can be used to find related DBpedia resources for a
        given string. The string may consist of a single or multiple words.

        """
        resultsets: typing.List[typing.List[SearchResult]] = []
        for search, _ in query_contexts:
            if not search:
                resultsets.append( [] )
                continue
            results = self._query(
                QueryString=search,
                MaxHits=limit or 1000,
                #   language=self.language,
            )
            if results:
                sr = []
                for r in results.get("docs", []):
                    for uri in r.get("resource", []):
                        if "type" in r:
                            r[RDF_type] = r.pop("type")
                            refcount = int((r.get("refCount") or ["0"])[0])
                        score = 1 - (1 / (1 + refcount))
                        sr.append(SearchResult(uri, r, score=score))
                resultsets.append( sr )
            else:
                resultsets.append( [] )
        return resultsets


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
            [(query, ())], context=(), limit=limit, add_about=add_about
        )
        return result

    funcs = [test]

    r = defopt.run(funcs, strict_kwonly=False, show_types=True)
    print(json.dumps(r))
