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
                uripart = urllib.parse.quote_plus(uripart, safe="'()&,")
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
            table_properties[name] = {str(keycol): tocol_props}

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
            for uri, celltext, rownum in csv.reader(open(fname)):
                rownum = str(int(rownum) - 1)
                row_uris[rownum] = {fix_uri(uri): 1.0}

            numheaderrows[name] = 1
            if any(int(ri) < 0 for ri in row_uris):
                numheaderrows[name] = 0
                row_uris = {str(int(ri) + 1): uris for ri, uris in row_uris.items()}

            table_entities[name] = {str(keycol): row_uris}

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
                    "classes": table_class.get(name),
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

TYPE_URI = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

def yield_triples_from_tablefile(fname, typename, add_labels=True):
    import pandas as pd
    
    s_type = 'http://dbpedia.org/ontology/' + typename
    t = pd.read_csv(fname, index_col=0, header=[0,1,2,3], dtype='str')
    for s,row in t.iterrows():
        yield ('<%s>' % s, '<%s>' % TYPE_URI, '<%s>' % s_type)
        
        for (pred_label, pred, typ_label, typ), vals in row.iteritems():
            if (not pd.isna(vals)) and (not pred_label.endswith('_label')) and vals:
                if vals[0] == '{' and vals[-1] == '}':
                    vals = vals[1:-1].split('|')
                else:
                    vals = [vals]
                for val in vals:
                    if typ_label in ['XMLSchema#string', 'rdf-schema#Literal']:
                        o = '"%s"' % val.replace('"','\\"')
                    elif 'XMLSchema' in typ_label:
                        o = '"%s"^^<%s>' % (val, typ)
                    elif val.startswith('http'):
                        o = '<%s>' % val
                    else:
                        o = '"%s"' % val.replace('"','\\"')

                    yield ('<%s>' % s, '<%s>' % pred, o)

LABEL_URI = 'http://www.w3.org/2000/01/rdf-schema#label'
def labels_from_dbpediauri(db):
    plabel = db.lookup_id(f"<{LABEL_URI}>")
    
    for i in range(db.n_terms()):
        uri = db.lookup_str(i)[1:-1]
        if uri.startswith('http://dbpedia.org/resource/') and not '__' in uri:
            uripart = uri.replace('http://dbpedia.org/resource/', '').replace('_', ' ')
            uripart = urllib.parse.unquote(urllib.parse.unquote(uripart))
            l = '"%s"' % uripart.strip().replace('"','\\"')
            li = db.lookup_id(l)
            if not li or not db.exists(i, plabel, li):
                yield ('<%s>'%uri, '<%s>'%LABEL_URI,  l)

            if uripart[-1] == ')' and '(' in uripart:
                part1, x = uripart.rsplit('(', 1)
                l = '"%s"' % part1.strip().replace('"','\\"')
                li = db.lookup_id(l)
                if not li or not db.exists(i, plabel, li):
                    yield ('<%s>'%uri, '<%s>'%LABEL_URI, l)

            if ', ' in uripart:
                part1, x = uripart.rsplit(', ', 1)
                l = '"%s"' % part1.strip().replace('"','\\"')
                li = db.lookup_id(l)
                if not li or not db.exists(i, plabel, li):
                    yield ('<%s>'%uri, '<%s>'%LABEL_URI, l)
                    
if __name__ == "__main__":
    import defopt, json

    log.getLogger().setLevel(log.DEBUG)

    def tables(path: Path, version: int = 2):
        print(json.dumps(T2D(path, version=version).tables))
        
    def dbpedia_subset(fname: Path):
        """Process Dbpedia subset
        
        Download the subset file from:
        ``http://data.dws.informatik.uni-mannheim.de/webtables/dbpedia_subset.tar.gz``

        
        """
        import tarfile
        tar = tarfile.open(fname, "r:gz")
        for tarinfo in tar:
            if tarinfo.name.endswith('.csv'):
                typename = tarinfo.name.replace('.csv', '')
                fname = tar.extractfile(tarinfo)
                for i,(s,p,o) in enumerate(yield_triples_from_tablefile(fname, typename)):
                    print(f"{s} {p} {o} .")
    
    def extra_labels(tridentdbdir: Path):
        """Add labels from Dbpedia URIs """
        import trident
        import json
        
        db = trident.Db(str(tridentdbdir))
        for s,p,o in labels_from_dbpediauri(db):
            print(f"{s} {p} {o} .")
        

    defopt.run([tables, dbpedia_subset, extra_labels], strict_kwonly=False)
