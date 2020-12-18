import typing
import logging as log
from pathlib import Path
from dataclasses import dataclass

import re
import urllib.parse as ul

from ..base import Lookup

try:
    import dawg
except:
    log.error("Cannot import DAWG")


@dataclass
class DawgLookup(Lookup):
    """Dawg Lookup

    Args:
        path: Path to intdawg file
        prefix: URI prefix
        extract: Regex for extracting title

    """

    path: Path
    prefix: str
    extract: typing.Optional[typing.Pattern] = None
    lookup: typing.Optional[dawg.IntDAWG] = None

    def __post_init__(self):
        if isinstance(self.extract, str):
            self.extract = re.compile(self.extract)

    def __enter__(self):
        self.lookup = dawg.IntDAWG()
        self.lookup.load(self.path)
        return self

    def __exit__(self, *args):
        try:
            delattr(self, 'lookup')
        except AttributeError:
            pass


    def lookup_title(self, title: str) -> typing.Optional[str]:
        assert self.lookup
        if self.extract:
            m = re.match(self.extract, title)
            if m and m.groups():
                title = m.groups()[0]
        title = ul.unquote_plus(title)

        if title in self.lookup:
            return self.prefix + str(self.lookup[title])

        return None
