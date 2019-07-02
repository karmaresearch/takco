"""
Finds row-entity candidates for entity-attribute tables
Usage: python candidates.py [table fnames file] [Elasticsearch index name] [URI_PREFIX] [outdir]
"""

import sys, os.path, json
try:
    _, table_fnames, INDEX, URI_PREFIX, outdir = sys.argv
    os.makedirs(outdir, exist_ok=True)
    candidatedir = os.path.join(outdir, 'candidates')
    os.makedirs(candidatedir, exist_ok=True)
    
    N_WORKERS = 8
    
    table_fnames = open(table_fnames).read().split()
except Exception as e:
    print(__doc__)
    raise e

import util    

tablerow_content = {}
table_row_keys = {}
for fname in table_fnames:
    name = os.path.basename(fname)
    print(f'Reading table {name:>80s}', end='\r', file=sys.stderr)
    webtable = util.WebTable.from_csv(fname)
    if webtable.keycol == None:
        print(f'Table {name} has no key column!')
        webtable.print_colstats()
        
    for r, cell_val in webtable.enumerate_keys():
        table_row_keys.setdefault(name, {}).setdefault(r, set()).add(cell_val)
        rowname = f'{name}~Row{r:d}'
        tablerow_content[rowname] = '|'.join(webtable.rows[r])
print(file=sys.stderr)

print(f'Loaded {len(table_row_keys)} tables, {len(tablerow_content)} rows', file=sys.stderr)
assert min([len(r) for t,r in table_row_keys.items()]) > 1

from elasticsearch import Elasticsearch
es = Elasticsearch()


def query(key):
    return {
        "query": {
            "bool": {
                "should":[
                    {"match":{"label": {"query":key }}},
                    {"match_phrase":{"label": {"query":key, "slop":1}}},
#                     {"match":{"label": {"query":key, "analyzer": "myanalyzer" }}},
#                     {"match_phrase":{"label": {"query":key, "slop":1, "analyzer": "myanalyzer"}}},                    
                ]
            }
        }
    }

search_cache = {}
def search(key, cutoff=0.8, size=3):
    if not key:
        return []
    if key in search_cache:
        return search_cache[key]
    
    body = query(key)
    body['size'] = size
    
    try:
        hits = es.search(index=INDEX, body=body).get('hits', {}).get('hits', [])
    except Exception as e:
        print(f'INDEX ERROR on {key}')
        print(e, file=sys.stderr)
        return []
    
    for hit in hits:
        label = hit.get('_source', {}).get('label', '')
        hit['_score'] = util.tokenjaccard(label, key)
    
    hits = sorted(hits, key=lambda h: -h.get('_score', 0))
    
    # duplicate T2K cutoff method
    max_score = max((h.get('_score', 0) for h in hits), default=1)
    if len(hits) > 1 and hits[1]['_score'] < cutoff*max_score:
        hits = [hits[0]]
    else:
        hits = hits[:3]
    
    search_cache[key] = hits
    return hits



import os, csv
def search_candidates(fname, cutoff=0.8, size=50):
    row_hasresults = {}
    name = os.path.basename(fname)
    
    with open(os.path.join(candidatedir, name), 'w') as fw:
        cw = csv.writer(fw)
        cw.writerow(["row","rowcontent","entity","score","label"])
#         if name not in table_row_keys:
#             print(f'Table MISSING {name}')
        for r, keys in table_row_keys.get(name, {}).items():
            row_hasresults.setdefault(r, False)
            row = '%s~Row%d' % (name,r)
            
            for key in keys:
                
                hits = search(key, cutoff=cutoff, size=size) # Search for key

                scored_uriparts = {}
                
                for hit in hits:    
                    label = hit.get('_source', {}).get('label', '')
                    score = hit.get('_score', 0)
                    
                    for uripart in hit.get('_source', {}).get('ids', []):
                        if uripart:
                            oldscore, _ = scored_uriparts.get(uripart, (0, None))
                            if score > oldscore:
                                scored_uriparts[uripart] = (score, label)
                
                for uripart, (score, label) in scored_uriparts.items():
                    uri = URI_PREFIX + str(uripart)
                    uri = util.quote(util.unquote(util.unquote(uri))) # idk
                    rowcontent = tablerow_content.get(row, '')
                    cw.writerow([row, rowcontent, uri, '%.4f'%score, label])
                    row_hasresults[r] = True
    return name, row_hasresults
    

import concurrent.futures
def search_all_candidates(fnames):
    l = len(fnames)
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        tasks = { executor.submit(search_candidates, fname) 
                  for fname in fnames }
        for i, res_task in enumerate(concurrent.futures.as_completed(tasks)):
#             continue
            name, row_hasresults = res_task.result()
            got = sum(row_hasresults.values())
            tot = len(row_hasresults)
            print(f'{i:4d} / {l:4d} searched; results for {got:4d}/{tot:4d} {name:>80s}', file=sys.stderr)


            
search_all_candidates(table_fnames)
print(file=sys.stderr)

