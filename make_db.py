"""
Put facts & labels into a sqlite database
"""

import os, sys, trident, sqlite3

try: 
    _, KBPATH, DBPATH  = sys.argv
    
    g = trident.Db(KBPATH)
    label_p = g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>')
    
    print(f'The KB has {g.count_p(label_p)} labels and {g.n_triples()} triples.')
    
    db = sqlite3.connect(DBPATH)
    db.executescript("""
        CREATE VIRTUAL TABLE Labels USING FTS5(lbl);
        
        CREATE TABLE Triples(s INTEGER, p INTEGER, o INTEGER);
        CREATE INDEX Triples_s ON Triples(s);
        CREATE INDEX Triples_p ON Triples(p);
        CREATE INDEX Triples_o ON Triples(o);
    """)
    db.commit()
    
except Exception as e:
    print(__doc__)
    raise e

def rdf_datatype(node):
    '''From RDF datatype string, get (root, type) pair'''
    if not node:
        return '', ''
    if node[0] == '<':
        if node[-1] == '>':
            return node[1:-1], 'uri' # uri
    elif node[0] == '"':
        if node[-1] == '"':
            return node[1:-1], 'str' # string
        elif node[-1] == '>' and '^^' in node:
            lit, dtype = node.rsplit('^^', 1)
            return lit[1:-1], dtype[1:-1] # typed literal
        elif '@' in node:
            lit, lang = node.rsplit('@', 1)
            return lit[1:-1], '@%s'%lang # language string
        else:
            return node[1], 'str' # string
    return '', ''


    
def labels():
    for s,p,o in g.all():
        if (s is not None) and (o is not None):
            node = g.lookup_str(o)
            if node:
                val, typ = rdf_datatype(node)
                if node and val:
                    yield (o, val)

print(f'Indexing labels...')
db.executemany("INSERT OR REPLACE INTO Labels(rowid,lbl) VALUES(?, ?)", labels())

print(f'Indexing triples...')
db.executemany("INSERT OR IGNORE INTO Triples(s,p,o) VALUES(?,?,?)", g.all())

db.commit()
