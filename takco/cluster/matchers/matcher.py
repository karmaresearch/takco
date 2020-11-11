from pathlib import Path
import logging as log
import sqlite3
import re
import pickle
from typing import (
    Optional,
    Dict,
    Set,
    Collection,
    Iterable,
    Tuple,
)
from abc import ABC, abstractmethod

import toml


def default_tokenize(text):
    if text.startswith("_"):
        return [text]
    return re.split(r"\W+", text.lower())  # pylint: disable=no-member


ScoredMatch = Tuple[Tuple[int, int, int, int], float]
Table = dict


class Matcher(ABC):
    name: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def __hash__(self):
        name = getattr(self, "name") if hasattr(self, "name") else ""
        return hash((self.__class__, name))

    def __sizeof__(self):
        return len(pickle.dumps(self))

    @abstractmethod
    def add(self, table: Table):
        """Add table to matcher index structures

        Args:
            table: Table to add
        """
        pass

    @abstractmethod
    def merge(self, matcher) -> "Matcher":
        """Merge this matcher with another

        Args:
            matcher (Matcher): Other matcher

        Returns:
            Merged matcher
        """
        return self

    @abstractmethod
    def index(self):
        """Create efficient index structures, possibly serialize to disk"""
        pass

    def close(self):
        pass

    def prepare_block(self, tableid_colids: Dict[int, Set[int]]):
        """Prepare matcher for blocking

        Args:
            tableid_colids: Mapping of global table IDs to column IDs
        """
        pass

    def block(self, tableid: int, colids: Collection[int]) -> Iterable[int]:
        """Find probably similar tables quickly

        Args:
            tableid: Table ID
            colids: Column IDs

        Returns:
            Global table IDs of similar tables
        """
        return ()

    @abstractmethod
    def match(
        self,
        tableid_colids_pairs: Iterable[
            Tuple[Tuple[int, Set[int]], Tuple[int, Set[int]]]
        ],
    ) -> Iterable[ScoredMatch]:
        """Calculate column similarities

        Args:
            tableid_colids_pairs: Pairs of table IDs and their column IDs

        Returns:
            Scored matches
        """
        pass
