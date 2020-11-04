"""
This module is executable. Run ``python -m takco.evaluate.dataset.t2d -h`` for help.
"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import logging as log
import csv
import json
import html
import urllib

from .dataset import Dataset


class T2D(Dataset):
    def get_name(self, fpath):
        return Path(fpath).name.split(".")[0]

    def __init__(self, root: Path, version=2, fixuri={}, **kwargs):
        root = Path(root)
        fixuri = dict(fixuri)

        def fix_uri(uri):

            uri = html.unescape(urllib.parse.unquote(uri))
            if version == 2:
                uri = urllib.parse.unquote(uri)

            uri = uri.replace("dbpedia_org", "dbpedia.org")
            uri = uri.replace("/page/", "/resource/")

            uri = fixuri.get(uri, uri)

            if uri.startswith("http://dbpedia.org/resource/"):
                uripart = uri.replace("http://dbpedia.org/resource/", "")
                uripart = urllib.parse.quote_plus(uripart, safe="'()&,!:")
                uri = "http://dbpedia.org/resource/" + uripart

            return uri

        vname = {
            "table": ["tables_instance", "tables"],
            "entity": ["entities_instance", "instance"],
            "class": ["classes_instance.csv", "classes_GS.csv"],
            "property": ["attributes_instance", "property"],
        }
        vpath = {k: root / Path(v[version - 1]) for k, v in vname.items()}

        # Rows
        table_rows = {}
        for fname in vpath["table"].glob("*"):
            tablefile = open(fname, "rb").read().decode("utf-8", errors="ignore")
            if version == 1:
                table_rows[self.get_name(fname)] = list(
                    csv.reader(tablefile.splitlines())
                )
            if version == 2:
                table_rows[self.get_name(fname)] = list(
                    zip(*json.loads(tablefile).get("relation", []))
                )
        log.info(f"Read {len(table_rows)} tables from {vpath['table']}")
        assert any(rows for rows in table_rows.values())

        # Properties (TODO: col-col props)
        table_properties = {}
        table_keycol = {}
        for fname in vpath["property"].glob("*"):
            name = self.get_name(fname)

            rows = list(csv.reader(open(fname)))
            for row in rows:
                if len(row) == 4:
                    uri, header, iskey, colnum = row
                else:
                    uri, header, colnum = row
                if iskey.lower() == "true":
                    table_keycol[name] = int(colnum)

            tocol_props = {}
            for row in rows:
                if len(row) == 4:
                    uri, header, iskey, colnum = row
                else:
                    uri, header, colnum = row

                tocol_props[str(colnum)] = {fix_uri(uri): 1.0}
            keycol = table_keycol.get(name, -1)
            table_properties[name] = {str(keycol): tocol_props} if tocol_props else {}

        # Classes
        table_class = {}
        if vpath["class"].exists():
            for row in csv.reader(open(vpath["class"])):
                if len(row) == 3:
                    fname, label, uri = row
                else:
                    fname, label, uri, keys = row
                name = self.get_name(fname)
                keycol = table_keycol.get(name, -1)
                table_class[name] = {str(keycol): {fix_uri(uri): 1.0}}

        # Entities
        table_entities = {}
        numheaderrows = {}
        for fname in vpath["entity"].glob("*"):
            name = self.get_name(fname)
            keycol = table_keycol.get(name, -1)

            row_uris = {}
            for row in csv.reader(open(fname)):
                if row:
                    uri, celltext, rownum = row
                    rownum = str(int(rownum) - 1)
                    row_uris[rownum] = {fix_uri(uri): 1.0}

            numheaderrows[name] = 1
            if any(int(ri) < 0 for ri in row_uris):
                numheaderrows[name] = 0
                row_uris = {str(int(ri) + 1): uris for ri, uris in row_uris.items()}

            table_entities[name] = {str(keycol): row_uris} if row_uris else {}
        log.info(f"Read {len(table_rows)} entity tables from {vpath['entity']}")

        assert bool(set(table_entities) & set(table_rows))

        table_info = [table_rows, table_entities, table_class, table_properties]
        names = set.union(*map(set, table_info))
        self.tables = []
        for name in sorted(names):
            self.tables.append(
                {
                    "name": name,
                    "headers": table_rows.get(name, [])[: numheaderrows.get(name)],
                    "rows": table_rows.get(name, [])[numheaderrows.get(name) :],
                    "entities": table_entities.get(name, {}),
                    "classes": table_class.get(name, {}),
                    "properties": table_properties.get(name, {}),
                    "keycol": table_keycol.get(name),
                }
            )

    def get_unannotated_tables(self):
        for table in self.tables:
            rows = [[{"text": c} for c in row] for row in table.get("rows")]
            headers = [[{"text": c} for c in row] for row in table.get("headers")]
            yield {
                "_id": table.get("name", ""),
                "tableData": rows,
                "tableHeaders": headers,
                "keycol": table["keycol"],
            }

    def get_annotated_tables(self):
        return {table["name"]: table for table in self.tables}


TYPE_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def yield_triples_from_tablefile(fname, typename, add_labels=True):
    import pandas as pd

    s_type = "http://dbpedia.org/ontology/" + typename
    t = pd.read_csv(fname, index_col=0, header=[0, 1, 2, 3], dtype="str")
    for s, row in t.iterrows():
        yield ("<%s>" % s, "<%s>" % TYPE_URI, "<%s>" % s_type)

        for (pred_label, pred, typ_label, typ), vals in row.iteritems():
            if (not pd.isna(vals)) and (not pred_label.endswith("_label")) and vals:
                if vals[0] == "{" and vals[-1] == "}":
                    vals = vals[1:-1].split("|")
                else:
                    vals = [vals]
                for val in vals:
                    if typ_label in ["XMLSchema#string", "rdf-schema#Literal"]:
                        o = '"%s"' % val.replace('"', '\\"')
                    elif "XMLSchema" in typ_label:
                        o = '"%s"^^<%s>' % (val, typ)
                    elif val.startswith("http"):
                        o = "<%s>" % val
                    else:
                        o = '"%s"' % val.replace('"', '\\"')

                    yield ("<%s>" % s, "<%s>" % pred, o)


LABEL_URI = "http://www.w3.org/2000/01/rdf-schema#label"


def get_file_lines(fname):
    t = None
    if Path(fname).is_file():
        try:
            from subprocess import run

            wc = run(["wc", "-l", fname], capture_output=True)
            t = int(wc.stdout.split()[0])
        except:
            pass
    return t


def surfaceforms_from_prefixuri(uri, prefix):
    if uri.startswith(prefix) and not "__" in uri:
        uripart = uri.replace(prefix, "")
        uripart = urllib.parse.unquote(urllib.parse.unquote(uripart))
        l = uripart.replace("_", " ").strip().lower()
        yield uripart, l, 1

        if l[-1] == ")" and "(" in l:
            part1, x = l.rsplit("(", 1)
            yield uripart, part1.strip().lower(), 1

        if ", " in l:
            part1, x = l.rsplit(", ", 1)
            if not (("and " in x) or ("& " in x)):
                yield uripart, part1.strip().lower(), 1


def all_surface(
    uri_pagetitle_file: Path, surfaceformscore_file: Path, redirect_file: Path = None,
):
    """Collect all surfaceforms, print as (URI, surfaceform-json) TSV

    Args:
        uri_pagetitle_file: TSV of (uri, page title) pairs
        surfaceformscore_file: surfaceFormsScore file
        redirect_file: Redirect file
    """
    import sys, tqdm, json

    assert uri_pagetitle_file.exists()
    assert surfaceformscore_file.exists()

    def load_synonyms(lines: Path):
        import urllib.parse as ul

        for line in lines:
            line = ul.unquote_plus(line.strip())
            try:
                a, b = line.split("\t", 1)
                a, b = a.strip(), b.strip()
                if a and b and (a != b):
                    yield b, a
                    yield a, b
            except:
                pass

    syn, check_syn = {}, ()
    if redirect_file:
        print("Loading synonyms from", redirect_file, file=sys.stderr)
        t = get_file_lines(redirect_file)
        syn = dict(load_synonyms(tqdm.tqdm(redirect_file.open(), total=t)))

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

    ent_surface_scores = {}
    if surfaceformscore_file:
        import urllib.parse as ul

        t = get_file_lines(surfaceformscore_file)
        print(f"Loading surface forms from {surfaceformscore_file}", file=sys.stderr)
        with Path(surfaceformscore_file).open() as fo:
            for i, line in enumerate(tqdm.tqdm(fo, total=t)):
                try:
                    line = ul.unquote_plus(line.strip())
                    ent, surface, score = line.split("\t")
                    score = float(score)
                    if "\\u" in surface:
                        surface = surface.encode("utf8").decode("unicode-escape")
                    for val in get_synonyms(surface):
                        ss = ent_surface_scores.setdefault(ent, {})
                        ss[val] = max(ss.get(val, 0), score)
                except Exception as e:
                    pass

    t = get_file_lines(uri_pagetitle_file)
    for line in tqdm.tqdm(open(uri_pagetitle_file), total=t):
        try:
            uri, pagetitle = line.strip().split(None, 1)
            import urllib.parse as ul

            pagetitle = ul.unquote_plus(pagetitle)
            surface_score = ent_surface_scores.get(pagetitle)
            if surface_score:
                top = max(surface_score.values())
                surface_score = {
                    sur: round((score / top) if score != 1.0 else 1.0, 5)
                    for sur, score in surface_score.items()
                }
                print(uri, json.dumps(surface_score), sep="\t")
        except:
            pass


if __name__ == "__main__":
    import defopt, json

    log.getLogger().setLevel(log.DEBUG)

    def tables(path: Path, version: int = 2):
        print(json.dumps(T2D(path, version=version).tables))

    def dbpedia_subset(fname: Path):
        """Convert Dbpedia subset tables to triples.

        Download the subset file from:
        ``http://data.dws.informatik.uni-mannheim.de/webtables/dbpedia_subset.tar.gz``


        """
        import tarfile

        tar = tarfile.open(fname, "r:gz")
        for tarinfo in tar:
            if tarinfo.name.endswith(".csv"):
                typename = tarinfo.name.replace(".csv", "")
                fname = tar.extractfile(tarinfo)
                for i, (s, p, o) in enumerate(
                    yield_triples_from_tablefile(fname, typename)
                ):
                    print(f"{s} {p} {o} .")

    def extra_surface(urifile: Path, prefix: str = "http://dbpedia.org/resource/"):
        """Add surface forms from prefixed URIs (default: DBpedia) """
        import tqdm

        for line in tqdm.tqdm(open(urifile), total=get_file_lines(urifile)):
            try:
                s, *_ = line.split()
            except Exception as e:
                log.error(e)
                s = ""
            uri = s.strip()
            if uri[0] == "<" and uri[-1] == ">":
                uri = uri[1:-1]
            for name, surface, score in surfaceforms_from_prefixuri(uri, prefix):
                print(name, surface, score, sep="\t")

    defopt.run(
        [tables, dbpedia_subset, extra_surface, all_surface], strict_kwonly=False
    )
