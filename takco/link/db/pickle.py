"""
This module is executable. Run ``python -m takco.link.sqlite -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

import typing
from pathlib import Path
import logging as log
import pickle

from ..base import Lookup


class PickleLookup(Lookup):
    def __init__(self, file: Path, baseuri="", fallback: Lookup = None):
        self.file = file
        self.baseuri = baseuri
        self.index = pickle.load(open(self.file, "rb"))
        self.fallback = fallback

    def lookup_title(self, title: str) -> str:
        """Gets the URI for a DBpedia entity based on wikipedia title."""
        title = title.replace(" ", "_").lower()
        try:
            if title in self.index:
                uri = self.index[title]
                if uri:
                    return self.baseuri + str(uri)
            else:
                if self.fallback:
                    uri = self.fallback.lookup_title(title)
                    log.debug(f"Fallback Wikititle for {title}: {uri}")
                    if str(uri) == "-1":
                        uri = None
                    self.index[title] = (
                        str(uri).replace(self.baseuri, "") if uri else None
                    )
                    return uri
        except Exception as e:
            log.error(e)

    def write(self, file: Path):
        with open(self.file, "wb") as fw:
            pickle.dump(self.index, fw)

    @classmethod
    def create():
        pass


if __name__ == "__main__":
    import defopt, json, os

    log.getLogger().setLevel(getattr(log, os.environ.get("LOGLEVEL", "WARN")))

    r = defopt.run(
        [PickleLookup.create],
        strict_kwonly=False,
        show_types=True,
        parsers={typing.Dict: json.loads},
    )
    print(r)
