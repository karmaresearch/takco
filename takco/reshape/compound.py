from typing import List, Dict, Tuple, NamedTuple, Any, Iterator
import re
import datetime
from collections import Counter

import logging as log

OffsetLinks = Dict[Tuple[int, int], str]
Cell = Dict


class CompoundSplit(NamedTuple):
    prefix: str  #: Compound part prefix
    dtype: str  #: Data type
    newcol: List[Cell]  #: New column


def get_cell_offsetlinks(column: List[Cell]) -> List[Tuple[str, OffsetLinks]]:
    """Utility function for getting cell link offsets"""
    cell_links = []
    for ri, cell in enumerate(column):
        text = cell.get("text", "")
        links = {}
        if text:
            for link in cell.get("surfaceLinks", []):
                if link.get("linkType", None) == "INTERNAL":
                    href = link.get("target", {}).get("href")
                    href = href or str(link.get("target", {}).get("id", ""))
                    start = link.get("offset", -1)
                    end = link.get("endOffset", -1)
                    if href and start >= 0 and end >= 0:
                        if start == 0 and end == 1:
                            end = len(text)  # HACK
                        links[(start, end)] = href
        cell_links.append((text, links))
    return cell_links


class CompoundSplitter:
    pass


class SpacyCompoundSplitter(CompoundSplitter):
    CARDINAL = re.compile(r"[\d,.]+")
    YEAR = re.compile(r"\d{4}")
    URL_REGEX = r"(?:(?:http://)|(?:www.))[^ ]+"
    URL = re.compile(URL_REGEX)
    EPOCH = datetime.datetime.utcfromtimestamp(0)
    EMPTY = ["", "-"]
    DASHES = "-–—"

    def __init__(self, model="en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model)

        suffixes = self.nlp.Defaults.infixes + tuple(self.DASHES)
        suffix_regex = spacy.util.compile_infix_regex(suffixes)
        self.nlp.tokenizer.infix_finditer = suffix_regex.finditer

        ruler = spacy.pipeline.EntityRuler(self.nlp, overwrite_ents=True)
        months = "january|february|march|april|may|june|july|august|september|october|november|december"
        patterns = [
            {
                "label": "SEASON",
                "pattern": [
                    {"TEXT": {"REGEX": r"^\d{4}$"}},
                    {"TEXT": {"REGEX": r"^[" + self.DASHES + "]$"}},
                    {"TEXT": {"REGEX": r"^\d{2}$"}},
                ],
            },
            {
                "label": "DATE",
                "pattern": [
                    {"LIKE_NUM": True},
                    {"LEMMA": {"REGEX": r"(?i)^(" + months + ")$"}},
                    {"LIKE_NUM": True},
                ],
            },
            {
                "label": "DATE",
                "pattern": [
                    {"LEMMA": {"REGEX": r"(?i)^(" + months + r")\d{1,2},\d{4}$"}},
                ],
            },
            {
                "label": "DATE",
                "pattern": [
                    {"TEXT": {"REGEX": r"^\d{4}$"}},
                    {"TEXT": {"REGEX": r"^[" + self.DASHES + r"]$"}},
                    {"TEXT": {"REGEX": r"^\d{2}$"}},
                    {"TEXT": {"REGEX": r"^[" + self.DASHES + r"]$"}},
                    {"TEXT": {"REGEX": r"^\d{2}$"}},
                ],
            },
            {"label": "URL", "pattern": [{"LEMMA": {"REGEX": self.URL_REGEX}},],},
        ]
        ruler.add_patterns(patterns)
        self.nlp.add_pipe(ruler)

    @staticmethod
    def make_linked_doc(doc, links):
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start : ent.end])

        if links:
            for (start, end), href in links.items():
                with doc.retokenize() as retokenizer:
                    s = doc.char_span(start, end)
                    if s and len(s):
                        retokenizer.merge(s)
            ents = []
            for (start, end), href in links.items():
                s = doc.char_span(start, end, label="ENTITY", kb_id=href)
                if s:
                    ents.append(s)

            for e in doc.ents:
                between = lambda x, e: e.start <= x <= e.end
                overlap = lambda a, b: between(a.start, b) or between(a.end, b)
                # Only add nlp entity if it doesn't clash with linked entity
                if not any(overlap(e, e_) or overlap(e_, e) for e_ in ents):
                    ents.append(e)
            doc.ents = ents
        return doc

    @staticmethod
    def get_typepattern(doc):
        parts, enttypes, entlinks = [], [], []
        part, enttype, entpart, enthref = "", None, "", None
        for t in doc:
            if t.ent_type_:
                if not enttype:
                    parts.append(part)
                part = ""
                if enttype and enttype != str(t.ent_type_):
                    enttypes.append(enttype)
                enttype = str(t.ent_type_)
                entpart += t.text_with_ws
                enthref = t.ent_kb_id_
            else:
                if enttype:
                    enttypes.append(enttype)
                    entlinks.append((entpart, enthref))
                enttype, entpart, enthref = None, "", None
                part += t.text_with_ws
        if enttype:
            enttypes.append(enttype)
            entlinks.append((entpart, enthref))
        parts.append(part)

        if any(enttypes):
            # Remove quotes
            if (parts[0].strip() == '"') and (parts[-1].strip() == '"'):
                parts = ["", ""]
        else:
            # It's a sentence if it has >1 words, the root is a verb, and it's not quoted
            has_root_verb = any(s.root.pos_ == "VERB" for s in doc.sents)
            is_quoted = (parts[0].strip() == '"') and (parts[-1].strip() == '"')
            if (len(doc) > 1) and has_root_verb and (not is_quoted):
                sent = [p + e[1] for p, e in zip(parts, enttypes + [("", "", "")])]
                parts, enttypes, entlinks = (
                    ["", ""],
                    ["SENTENCE"],
                    [("".join(sent), None)],
                )

        return tuple(parts), tuple(enttypes), tuple(entlinks)

    def candidate_splits(self, column: List[Cell]):
        cellset, offsetlinks = zip(*dict(get_cell_offsetlinks(column)).items())
        if len(cellset) > 1:
            docs = self.nlp.pipe(cellset, batch_size=50)
            patterns = [
                self.get_typepattern(self.make_linked_doc(doc, links))
                for links, doc in zip(offsetlinks, docs)
            ]
            return cellset, patterns
        return cellset, ()

    def find_splits(self, column: List[Cell]) -> Iterator[CompoundSplit]:
        cellset, patterns = self.candidate_splits(column)
        pattern_freq = Counter((part, types) for part, types, links in patterns)
        if pattern_freq:
            # log.debug(f"Found patterns ({len(cellset)}) {pattern_freq.most_common(3)}")

            for (colparts, coltypes), freq in pattern_freq.most_common(1):
                # Check if the most frequent pattern occurs in over half of cells
                if len(coltypes) > 1 and freq > len(cellset) / 2:

                    # Turn the pattern into a regular expression
                    reparts = [re.escape(p.strip()) for p in colparts]
                    for equiv in [self.DASHES]:
                        reparts = [
                            "[%s]" % equiv if p and p in equiv else p for p in reparts
                        ]
                    pattern_regex = re.compile("(.+?)".join(reparts))

                    cell_pattern = dict(zip(cellset, patterns))
                    newcols, restcol = [], []
                    for cellobj in column:
                        nomatch = ""
                        cell = cellobj.get("text")
                        if cell in cell_pattern:
                            parts, types, links = cell_pattern[cell]
                            if (colparts, coltypes) == (parts, types):
                                newcols.append(
                                    [
                                        {
                                            "text": cell.strip(),
                                            "surfaceLinks": [
                                                {
                                                    "linkType": "INTERNAL",
                                                    "target": {"href": href},
                                                    "offset": 0,
                                                    "endOffset": len(cell),
                                                }
                                            ],
                                        }
                                        for cell, href in links
                                    ]
                                )
                                restcol.append({})
                            else:
                                # Try to match the pattern regex
                                m = pattern_regex.match(cell)
                                if m:
                                    newcols.append(
                                        [{"text": g.strip()} for g in m.groups()]
                                    )
                                    restcol.append({})
                                else:
                                    newcols.append([{} for _ in coltypes])
                                    restcol.append({"text": cell})

                    newcols = list(zip(*newcols))
                    if any(restcol):
                        newcols.append(restcol)
                        coltypes = coltypes + (None,)
                    else:
                        colparts = list(colparts)[:-1]

                    for part, dtype, newcol in zip(colparts, coltypes, newcols):
                        yield CompoundSplit(part, dtype, newcol)
