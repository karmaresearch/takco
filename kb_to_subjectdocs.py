"""
Creates a newline-delimited json file for Elasticsearch of labels and the entities they refer to
Usage: python kb_to_docs.py [DB_PATH]
"""

import sys
try:
    _, DB_PATH = sys.argv
except Exception as e:
    print(__doc__)
    raise e
    
import trident
# trident.setLoggingLevel(0)
g = trident.Db(DB_PATH)
type_id = g.lookup_id('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')
label_ids = [
    g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>'),
    g.lookup_id('<http://dbpedia.org/ontology/wikiPageWikiLinkText>')
]
label_ids = [i for i in label_ids if i != None]
redir_ids = [
    g.lookup_id('<http://dbpedia.org/ontology/wikiPageRedirects>'),
    g.lookup_id('<http://dbpedia.org/ontology/wikiPageDisambiguates>')
]
redir_ids = [i for i in redir_ids if i != None]

print('Trident KB has %d terms, %d triples' % (g.n_terms(), g.n_triples()), file=sys.stderr)
for label_id in label_ids:
    print('Using label predicate %s' % g.lookup_str(label_id), file=sys.stderr)
for redir_id in redir_ids:
    print('Using redirect predicate %s' % g.lookup_str(redir_id), file=sys.stderr)

def get_labels(s):
    for label_id in label_ids:
        for label in g.o(s, label_id):
            yield label

def get_redirs(s):
    for redir_id in redir_ids:
        for redir in g.o(s, redir_id):
            yield redir

import util

label_entities = {}
n = len(g.all_s())
for i,s in enumerate(g.all_s()):
    if not i % 1000: print(f'Labeling entity {i:10d}/{n}', end='\r', file=sys.stderr)
    for s_resolved in (set(get_redirs(s)) or [s]):
        
        for label_node in get_labels(s):
            label_node_str = g.lookup_str(label_node)
            if not label_node_str:
                continue
            root, dtype = util.rdf_datatype(label_node_str)
            if dtype == 'str' or dtype.startswith('@'):
                for label in util.label_variants(root):
                    label = label.encode().decode('unicode-escape').lower()
                    if label.isnumeric():
                        continue
                    label_entities.setdefault(label, set()).add( s_resolved )



def stream():
    for label,ids in sorted(label_entities.items(), key=lambda x: -len(x[1])): 
        doc = {
            'label': label,
            'p': g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>'),
            'ids': list(ids),
        }
        yield doc
        
        # Find the subjects that these entities are the objects of
        p_subjects = {}
        for o in ids:
            for p,s in g.ps(o):
                p_subjects.setdefault(p, set()).add(s)
        for p, subjects in p_subjects.items():
            doc = {
                'label': label,
                'p': p,
                'ids': list(subjects),
            }
            yield doc

import json
for i,doc in enumerate(stream()):
    print(json.dumps(doc, ensure_ascii=False))
    if 0 == i % 1000:
        print('indexed %d    ' % i, end='\r', file=sys.stderr)
        sys.stderr.flush()
