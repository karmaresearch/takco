import typing
import logging as log
from pathlib import Path
import glob
from abc import ABC, abstractmethod

from .. import link


class Page(typing.NamedTuple):
    url: str  #:
    html: str  #:
    about: typing.Optional[str]  #:


class PageSource(ABC):

    sources: typing.Collection[typing.Any]

    @abstractmethod
    def load(self, sources: typing.Collection[typing.Any]) -> typing.Iterable[Page]:
        pass


class WikiPages(PageSource):
    """
    Download Wikipedia articles

    Downloads HTML files for entities that have a certain predicate/object in a DB.

    Args:
        ent_abouturl: A ``[(uri,url)]`` collection of entity URIs and page URLs
        encoding: Page encoding (use "guess" to use requests' apparent encoding)
    """

    def __init__(
        self,
        ent_abouturl: typing.Collection[typing.Tuple[str, str]],
        encoding: str = None,
    ):
        self.ent_abouturl = ent_abouturl
        self.encoding = encoding

    @property
    def sources(self):
        return self.ent_abouturl

    def load(self, sources=None):
        import requests

        if sources is None:
            sources = self.sources

        for e, url in sources:
            result = requests.get(url)
            if self.encoding:
                if self.encoding == "guess":
                    result.encoding = result.apparent_encoding
                else:
                    result.encoding = self.encoding
            if result.status_code == 200:
                yield Page(
                    url=url, about=e, html=result.text,
                )


class WarcPages(PageSource):
    """Load HTML pages from WARC files

    Args:
        globstrings: Glob strings for WARC gz files
    """

    def __init__(
        self, globstrings: typing.Union[str, typing.List[str]],
    ):
        if not isinstance(globstrings, list):
            globstrings = [globstrings]

        fnames = [fname for g in globstrings for fname in glob.glob(g)]
        assert len(fnames), f"No glob results for {globstrings}"
        self.fnames = fnames

    @property
    def sources(self):
        return self.fnames

    def load(self, sources=None):
        from warcio.archiveiterator import ArchiveIterable

        if sources is None:
            sources = self.sources

        for fname in sources:
            with open(fname, "rb") as stream:
                for record in ArchiveIterable(stream):
                    if record.rec_type == "response":
                        url = record.rec_headers.get_header("WARC-Target-URI")
                        e = None
                        if "?about=" in url:
                            url, e = url.rsplit("?about=", 1)

                        text = record.content_stream().read().decode()
                        yield Page(
                            url=url, about=e, html=text,
                        )


class LinePages(PageSource):
    def __init__(
        self,
        globstrings: typing.Union[str, typing.List[str]],
        datadir: Path = None,
        lookup: link.Lookup = None,
        title_regex: typing.Optional[str] = None,
    ):
        if not isinstance(globstrings, list):
            globstrings = [globstrings]

        fnames = [fname for g in globstrings for fname in glob.glob(g)]
        assert len(fnames), f"No glob results for {globstrings}"
        self.fnames = fnames

        self.lookup = lookup

        import re

        self.title_regex = re.compile(title_regex) if title_regex else None

    @property
    def sources(self):
        return self.fnames

    def load(self, sources=None):
        import json

        if sources is None:
            sources = self.sources

        if self.lookup is not None:
            self.lookup.__enter__()

        for fname in sources:
            for line in open(fname):
                try:
                    url, html = line.rstrip().split(None, 1)

                    title = url
                    if self.title_regex is not None:
                        m = self.title_regex.match(url)
                        if m:
                            title = m.group(1)

                    about = title
                    if self.lookup is not None:
                        about = self.lookup.lookup_title(title)

                    yield Page(
                        url=url, about=about, html=json.loads(html),
                    )

                except Exception as e:
                    log.error(e)

                if self.lookup is not None:
                    self.lookup.flush()

        if self.lookup is not None:
            self.lookup.__exit__(None, None, None)
