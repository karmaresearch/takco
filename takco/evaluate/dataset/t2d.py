from pathlib import Path
import logging as log
import csv
import json
import html
import urllib
import typing

from .dataset import Dataset


class T2D(Dataset):
    fixuri: typing.Dict[str, str]

    def get_name(self, fpath):
        return Path(fpath).name.split(".")[0]

    def __init__(
        self,
        datadir: Path = None,
        resourcedir: Path = None,
        path: Path = None,
        version=None,
        fixuri: typing.Union[Path, typing.Dict[str,str]] = {},
        **kwargs,
    ):
        kwargs = self.params(
            path=path, datadir=datadir, resourcedir=resourcedir, fixuri=fixuri, **kwargs
        )
        root = Path(kwargs.get("path", "."))
        if isinstance(fixuri, str):
            self.fixuri = dict(list(csv.reader(Path(fixuri).open())))
        elif isinstance(fixuri, dict):
            self.fixuri = dict(fixuri)
        
        self.version = version
        
        keycolfile = None
        entitydir: typing.Optional[Path]
        if version == 1:
            tabledir = root.joinpath("tables_instance")
            entitydir = root.joinpath("entities_instance")
            classfile = root.joinpath("classes_instance.csv")
            propdir  = root.joinpath("attributes_instance")
        elif version == 2:
            tabledir = root.joinpath("tables")
            entitydir = root.joinpath("instance")
            classfile = root.joinpath("classes_GS.csv")
            propdir = root.joinpath("property")
        else:
            tabledir = Path(kwargs["tabledir"])
            entitydir = Path(kwargs['entitydir']) if 'entitydir' in kwargs else None
            classfile = Path(kwargs["classfile"])
            propdir = Path(kwargs["propdir"])
            keycolfile = Path(kwargs['keycolfile']) if 'keycolfile' in kwargs else None

        table_rows = self.get_table_rows(tabledir)
        if keycolfile:
            table_keycol = self.get_table_keycol_from_file(keycolfile)
        else:
            table_keycol = self.get_table_keycol_from_props(propdir)
        table_properties = self.get_table_properties(propdir, table_keycol)
        table_class = self.get_table_class(classfile, table_keycol)
        if entitydir:
            table_numheaderrows = self.get_table_numheaderrows(entitydir)
            table_entities = self.get_table_entities(entitydir, table_keycol, table_numheaderrows)
        else:
            table_numheaderrows = {}
            table_entities = {}


        self.tables = []
        for name in sorted(table_rows):
            # TODO: make this suck less
            self.tables.append(
                {
                    "name": name,
                    "headers": table_rows.get(name, [])[: table_numheaderrows.get(name, 1)],
                    "rows": table_rows.get(name, [])[table_numheaderrows.get(name, 1) :],
                    "entities": table_entities.get(name, {}),
                    "classes": table_class.get(name, {}),
                    "properties": table_properties.get(name, {}),
                    "keycol": table_keycol.get(name),
                }
            )
    
    def fix_uri(self, uri):
        uri = html.unescape(urllib.parse.unquote(uri))
        if self.version == 2:
            uri = urllib.parse.unquote(uri)

        uri = uri.replace("dbpedia_org", "dbpedia.org")
        uri = uri.replace("/page/", "/resource/")

        uri = self.fixuri.get(uri, uri)

        if uri.startswith("http://dbpedia.org/resource/"):
            uripart = uri.replace("http://dbpedia.org/resource/", "")
            uripart = urllib.parse.quote_plus(uripart, safe="'()&,!:")
            uri = "http://dbpedia.org/resource/" + uripart

        return uri

    def get_table_rows(self, tabledir):
        table_rows = {}
        for fname in tabledir.glob("*"):
            tablefile = open(fname, "rb").read().decode("utf-8", errors="ignore")
            if self.version == 2:
                table_rows[self.get_name(fname)] = list(
                    zip(*json.loads(tablefile).get("relation", []))
                )
            else:
                table_rows[self.get_name(fname)] = list(
                    csv.reader(tablefile.splitlines())
                )
                
        log.info(f"Read {len(table_rows)} tables from {tabledir}")
        assert any(rows for rows in table_rows.values())
        return table_rows

    def get_table_keycol_from_props(self, propdir):
        table_keycol = {}
        for fname in propdir.glob("*"):
            name = self.get_name(fname)
            for row in csv.reader(open(fname)):
                if len(row) == 4:
                    uri, header, iskey, colnum = row
                else:
                    uri, header, colnum = row
                    iskey = ''
                assert header is not None and uri is not None
                if iskey.lower() == "true":
                    table_keycol[name] = int(colnum)
        log.info(f"Read {len(table_keycol)} key cols from {propdir}")
        return table_keycol

    def get_table_keycol_from_file(self, keycolfile):
        table_keycol = {}
        if keycolfile.exists():
            for table_fname, keycol in csv.reader(open(keycolfile)):
                name = self.get_name(table_fname)
                table_keycol[name] = int(keycol)
        log.info(f"Read {len(table_keycol)} key cols from {keycolfile}")
        return table_keycol


    def get_table_properties(self, propdir, table_keycol):
        table_properties = {}
        for fname in propdir.glob("*"):
            name = self.get_name(fname)
            keycol = table_keycol.get(name)
            tocol_props = {}
            for row in csv.reader(open(fname)):
                if len(row) == 4:
                    uri, header, iskey, colnum = row
                else:
                    uri, header, colnum = row
                    iskey = ''
                assert header is not None and iskey is not None
                if str(keycol) != str(colnum):
                    tocol_props[str(colnum)] = {self.fix_uri(uri): 1.0}

            table_properties[name] = {str(keycol): tocol_props} if tocol_props else {}
        log.info(f"Read {len(table_properties)} prop files from {propdir}")
        return table_properties

    def get_table_class(self, classfile, table_keycol):
        table_class = {}
        if classfile.exists():
            for row in csv.reader(open(classfile)):
                if len(row) == 3:
                    table_fname, label, uri = row
                else:
                    table_fname, label, uri, keys = row
                    assert keys is not None
                assert label is not None
                name = self.get_name(table_fname)
                keycol = table_keycol.get(name, -1)
                table_class[name] = {str(keycol): {self.fix_uri(uri): 1.0}}
        log.info(f"Read {len(table_class)} class tables from {classfile}")
        return table_class

    def get_table_numheaderrows(self, entitydir):
        numheaderrows = {}
        for fname in entitydir.glob("*"):
            name = self.get_name(fname)
            rownums = [r[-1] for r in csv.reader(open(fname)) if r]
            numheaderrows[name] = 1
            if any(int(ri) < 0 for ri in rownums):
                numheaderrows[name] = 0
        log.info(f"Read {len(numheaderrows)} table headerrows from {entitydir}")
        return numheaderrows

    def get_table_entities(self, entitydir, table_keycol, numheaderrows):
        table_entities = {}
        for fname in entitydir.glob("*"):
            name = self.get_name(fname)
            keycol = table_keycol.get(name, -1)

            row_uris = {}
            for row in csv.reader(open(fname)):
                if row and len(row) == 3:
                    uri, celltext, rownum = row
                    assert celltext is not None
                    rownum = str(int(rownum) - 1)
                    row_uris[rownum] = {self.fix_uri(uri): 1.0}
                else:
                    log.error(f"Bad row: {row}")

            if numheaderrows.get(name, 1) == 0:
                row_uris = {str(int(ri) + 1): uris for ri, uris in row_uris.items()}

            table_entities[name] = {str(keycol): row_uris} if row_uris else {}
        log.info(f"Read {len(table_entities)} entity tables from {entitydir}")
        return table_entities