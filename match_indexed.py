"""
Matches cells of entity-attribute table rows to links of row-entity candidates in the KB
This uses an Elasticsearch index but it's sadly very slow.

Usage: python match_indexed.py [table fnames file] [candidate fnames file] [Trident db path] [outdir]
"""

import trident, util
import pandas as pd
import os, glob, sys, urllib

from elasticsearch import Elasticsearch
es = Elasticsearch()

def att_query(att, subject_ids):
    return {
        "_source": ["label", "p"],
        "query": {
            "bool": {
                "filter": [
                    {"terms":{"ids": list(subject_ids) }}
                ],
                "should":[
                    {"match":{"label": {"query":att }}},
                    {"match_phrase":{"label": {"query":att, "slop":1}}},
                ]
            }
        }
    }

def search_attributes(index, att, subject_ids, size=100):
    q = att_query(att, subject_ids)
    q['size'] = size
    hits = es.search(index=index, body=q).get('hits', {}).get('hits', [])
    subject_ids = set(subject_ids)
    for hit in hits:
        label = hit.get('_source', {}).get('label','')
        p = hit.get('_source', {}).get('p', -1)
        matched_ids = subject_ids & set(hit.get('_source', {}).get('ids', []))
        tokenjacc = util.tokenjaccard(label, att)
        if tokenjacc:
            yield tokenjacc, p, label, matched_ids

# Loop over candidates
def match(index, tablename, webtable, candidates, inlinks=True):
    tablerow_cells = {}
    for r, cells in enumerate(webtable.rows_split):
        rowname = f'{tablename}~Row{r:d}'
        tablerow_cells[rowname] = cells

    matches = []
    link_rows = {}
    for rowname, group in candidates.groupby('row'):
        ids = set(group['id'])
        print(f'{rowname:>50s}', end='\r', file=sys.stderr)
        cells = tablerow_cells[rowname]
        for colnr, cellvals in enumerate(cells):
            print(colnr, cellvals)
            for cellval in cellvals:
                for tokenjacc, p, label, matched_ids in search_attributes(index, cellval, ids):
#                     print(f'{tokenjacc:5.2f} {len( matched_ids):2d} {label}')
                    pass
                
#         s, s_uri, tablename, rownr = row.id, row.entity, row.table, row.rownr
#         if s == -1: continue
#         print(f'{row.row:>50s} : {s_uri} ', end='\r', file=sys.stderr)

if __name__ == '__main__':
    import sys, os.path, json
    try:
        _, table_fnames, candidate_fnames, DB_PATH, INDEX, outdir = sys.argv
        table_fnames = open(table_fnames).read().split()
        table_names = set([os.path.basename(fname) for fname in table_fnames])
        candidate_fnames = open(candidate_fnames).read().split()
        index = INDEX
        
        def make_dir(name):
            path = os.path.join(outdir, name)
            os.makedirs(path, exist_ok=True)
            return path

        os.makedirs(outdir, exist_ok=True)

        linkdir = make_dir('links')
        entitydir = make_dir('entities')
        matchdir = make_dir('matches')

    except Exception as e:
        print(__doc__)
        raise e
    
    # trident.setLoggingLevel(0)
    g = trident.Db(DB_PATH)
    type_id = g.lookup_id('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')
    label_id = g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>')
    assert label_id != None
    print('Trident KB has %d terms, %d triples' % (g.n_terms(), g.n_triples()))
    def try_lookup(uri):
        uri = util.quote(util.unquote(util.unquote(uri)))
        return g.lookup_id('<%s>' % uri) or -1
    
    ### LOAD TABLES ###
    webtables = {}
    for fname in table_fnames:
        name = os.path.basename(fname)
        print(f'Reading table {name:>80s}', end='\r', file=sys.stderr)
        webtables[name] = util.WebTable.from_csv(fname)
    print(file=sys.stderr)

    ### LOAD CANDIDATES ###
    candidates = pd.concat([pd.read_csv(fname) for fname in candidate_fnames], ignore_index=True)
    candidates['table'] = candidates['row'].map(lambda row: row.split('~')[0])
    candidates['rownr'] = candidates['row'].map(lambda r: int(r.split('~Row')[1]))
    candidates = candidates.sort_values(['table', 'rownr'])
    # Filter based on table names
    candidates = candidates.loc[candidates.table.map(lambda x: x in table_names)]
    print('candidates:', len(candidates), '; cols:', list(candidates.columns))
    print('%d unique rows' % len(set(candidates['row'])), '; %d tables' % len(set(candidates['table'])))
    print('%d unique entities' % len(set(candidates['entity'])))
    sys.stdout.flush()
    
    ## CHECK ENTITY CANDIDATES ##
    candidates['entity'] = candidates.entity.map(lambda uri: uri.replace('/page/', '/resource/'))
    candidates['name'] = candidates['entity'].map(lambda s: s.replace('http://dbpedia.org/resource/', ''))
    candidates['id'] = candidates.entity.map(try_lookup)
    uris_notfound = set([uri for uri,i in set(zip(candidates['entity'],candidates['id'])) if i == -1])
    print('%d candidate entities not in KB' % len(uris_notfound))
    if uris_notfound:
        print('Missing candidate entities:')
        for uri in uris_notfound:
            print(' ', uri)
    sys.stdout.flush()
    
    ### MATCH CELLS TO LINKS ###
    gr = candidates.groupby('table')
    for tablename, pertable_idx in gr.indices.items():
        webtable = webtables.get(tablename, None)
        if not webtable:
            print('Missing table: %s' % row['table'])
            continue
        matchtable = match(index, tablename, webtable, gr.get_group(tablename).copy())

