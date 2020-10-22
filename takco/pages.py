from .util import Config, get_executor_kwargs
import typing
import logging as log
from pathlib import Path


class WikiPages:
    """
    Download Wikipedia articles

    Downloads HTML files for entities that have a certain predicate/object in a DB.

    Args:
        dbconfig: DB configuration (default: Wikidata)
        pred: Predicate URI
        obj: Object URI
        urlprefix: Wikipedia URL prefix
        localprefix: Wikipedia URL prefix for locally hosted pages
        encoding: Force encoding (use "guess" for guessing)
    """

    def __init__(
        self,
        dbconfig: Config = {"class": "GraphDB", "store": {"class": "SparqlStore"}},
        pred: str = None,
        obj: str = None,
        urlprefix: str = "https://en.wikipedia.org/wiki/",
        localurlprefix: str = None,
        sample: int = None,
        justurls: bool = False,
        encoding: str = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):
        self.justurls = justurls
        self.ex = get_executor_kwargs(executor, assets)

        from . import link

        with Config(dbconfig, assets).init_class(**link.__dict__) as db:

            ent_abouturl = []
            for e, ps in db.pages_about([None, pred, obj]).items():
                for pageurl in ps:
                    if localurlprefix:
                        pageurl = pageurl.replace(urlprefix, localurlprefix)
                    ent_abouturl.append((e, pageurl))
            self.ent_abouturl = ent_abouturl[:sample]

    @staticmethod
    def download(
        ent_abouturl: typing.Collection[typing.Tuple[str, str]], encoding=None
    ):
        """Download html pages from urls

        Args:
            ent_abouturl: A ``[(uri,url)]`` list of entitu URIs and page URLs
            encoding: Page encoding (use "guess" to use requests' apparent encoding)
        """
        import requests

        for e, url in ent_abouturl:
            result = requests.get(url)
            if encoding:
                if encoding == "guess":
                    result.encoding = result.apparent_encoding
                else:
                    result.encoding = encoding
            if result.status_code == 200:
                yield {
                    "url": url,
                    "about": e,
                    "html": result.text,
                }

    def get(self):
        ent_abouturl = self.ent_abouturl
        if self.justurls:
            return ({"entity": e, "page": url} for e, url in ent_abouturl)
        executor, exkw = self.ex
        log.info(f"Downloading {len(ent_abouturl)} pages with executor {executor}")
        return executor(ent_abouturl, **exkw).pipe(self.download, encoding=encoding)


class WarcPages:
    """Load HTML pages from WARC files

    Args:
        globstrings: Glob strings for WARC gz files
        datadir: Data directory

    """

    def __init__(
        self,
        globstrings: typing.List[str] = (),
        datadir: Path = None,
        executor: Config = None,
        assets: typing.List[Config] = (),
    ):

        self.ex = get_executor_kwargs(executor, assets)

        fnames = [fname for g in globstrings for fname in Path(".").glob(g)]
        assert len(fnames), f"No glob results for {globstrings}"
        self.fnames = fnames

    @staticmethod
    def parse_warc(fnames):
        """Yield html pages from WARC files"""
        from warcio.archiveiterator import ArchiveIterator

        for fname in fnames:
            with open(fname, "rb") as stream:
                for record in ArchiveIterator(stream):
                    if record.rec_type == "response":
                        url = record.rec_headers.get_header("WARC-Target-URI")
                        e = None
                        if "?about=" in url:
                            url, e = url.rsplit("?about=", 1)

                        text = record.content_stream().read().decode()
                        yield {
                            "url": url,
                            "about": e,
                            "html": text,
                        }

    def get(self):
        fnames = self.fnames
        executor, exkw = self.ex
        log.info(
            f"Extracting pages from {len(fnames)} warc files using executor {executor}"
        )
        return executor(fnames, **exkw).pipe(self.parse_warc)
