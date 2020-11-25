from .util import HashBag, TqdmHashBag, robust_json_loads_lines
import typing
import logging as log
from pathlib import Path
import glob
from abc import ABC, abstractmethod

from . import link


class PageSource(ABC):
    @abstractmethod
    def get(self, executor: HashBag = HashBag([])):
        pass


class WikiPages(PageSource):
    """
    Download Wikipedia articles

    Downloads HTML files for entities that have a certain predicate/object in a DB.

    Args:
        db: DB configuration (default: Wikidata)
        pred: Predicate URI
        obj: Object URI
        urlprefix: Wikipedia URL prefix
        localprefix: Wikipedia URL prefix for locally hosted pages
        encoding: Force encoding (use "guess" for guessing)
    """

    def __init__(
        self,
        db: link.Database,
        pred: str = None,
        obj: str = None,
        urlprefix: str = "https://en.wikipedia.org/wiki/",
        localurlprefix: str = None,
        sample: int = None,
        justurls: bool = False,
        encoding: str = None,
    ):
        self.justurls = justurls
        self.encoding = encoding

        with db:

            ent_abouturl = []
            for e, ps in db.pages_about([None, pred, obj]).items():
                for pageurl in ps:
                    if localurlprefix:
                        pageurl = pageurl.replace(urlprefix, localurlprefix)
                    ent_abouturl.append((e, pageurl))
            self.ent_abouturl = ent_abouturl[:sample]

    @staticmethod
    def download(
        ent_abouturl: typing.Collection[typing.Tuple[str, str]], encoding: str
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

    def get(self, executor: HashBag = HashBag([])):
        ent_abouturl = self.ent_abouturl
        if self.justurls:
            return ({"entity": e, "page": url} for e, url in ent_abouturl)
        log.info(f"Downloading {len(ent_abouturl)} pages with executor {executor}")
        return executor.new(ent_abouturl).pipe(self.download, encoding=self.encoding)


class WarcPages(PageSource):
    """Load HTML pages from WARC files

    Args:
        globstrings: Glob strings for WARC gz files
        datadir: Data directory

    """

    def __init__(
        self, globstrings: typing.Union[str, typing.List[str]], datadir: Path = None,
    ):
        if not isinstance(globstrings, list):
            globstrings = [globstrings]

        fnames = [fname for g in globstrings for fname in glob.glob(g)]
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

    def get(self, executor: HashBag = HashBag([])):
        fnames = self.fnames
        log.info(
            f"Extracting pages from {len(fnames)} warc files using executor {executor}"
        )
        return executor.new(fnames).pipe(self.parse_warc)


class LinePages(PageSource):
    def __init__(
        self,
        globstrings: typing.Union[str, typing.List[str]],
        datadir: Path = None,
        lookup: link.Lookup = None,
        title_regex: str = None,
    ):
        if not isinstance(globstrings, list):
            globstrings = [globstrings]

        fnames = [fname for g in globstrings for fname in glob.glob(g)]
        assert len(fnames), f"No glob results for {globstrings}"
        self.fnames = fnames

        self.lookup = lookup

        import re

        self.title_regex = re.compile(title_regex) if title_regex else None

    @staticmethod
    def parse_line(fnames, lookup, title_regex):
        import json

        if lookup is not None:
            lookup.__enter__()

        for fname in fnames:
            for line in open(fname):
                try:
                    url, html = line.rstrip().split(None, 1)

                    title = url
                    if title_regex:
                        title = title_regex.match(url).group(1)

                    about = title
                    if lookup is not None:
                        about = lookup.lookup_title(title)

                    yield {
                        "url": url,
                        "about": about,
                        "html": json.loads(html),
                    }

                except Exception as e:
                    log.error(e)

                if (lookup is not None) and hasattr(lookup, "flush"):
                    lookup.flush()

        if lookup is not None:
            lookup.__exit__(None, None, None)

    def get(
        self, executor: HashBag = HashBag([]),
    ):
        fnames = self.fnames
        log.info(
            f"Extracting pages from {len(fnames)} line files using executor {executor}"
        )
        return executor.new(fnames).pipe(self.parse_line, self.lookup, self.title_regex)
