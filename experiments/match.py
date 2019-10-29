"""
Matches cells of entity-attribute table rows to links of row-entity candidates in the KB
Usage: python match.py [table fnames file] [candidate fnames file] [Trident db path] [inlinks 0/1] [outdir]
"""

import trident, util
import pandas as pd
import os, glob, sys, urllib

p_redir = {}
s_linkscache = {}
def get_s_links(s, inlinks=True):
    if s == -1: return
    # Yield all in- and out-links of s (including null-valued)
    if s in s_linkscache:
        yield from s_linkscache[s]
        return
    for d, pvs in [(+1, g.po(s))] + ([(-1, g.ps(s))] if inlinks else []):
        for p,v in pvs:
            p = p_redir.get(p,p)
            for v in [v,-1]:
                link = d,p,v
                s_linkscache.setdefault(s, set()).add( link )
                yield link

link_count = {}
def count_link(d,p,v):
    # Count number of entities with this (possibly null-valued) in- or out-link
    if (d,p,v) in link_count:
        return link_count[(d,p,v)]
    else:
        count = g.count_p(p) if v == -1 else (g.n_s(p,v) if d == 1 else g.n_o(v,p))
        link_count[(d,p,v)] = count
        return count

s_strscache = {}
def get_s_strs(v):
    # Yield all strings of v (value if v is literal, label if v is entity)
    if v == -1: return
    if v in s_strscache:
        yield from s_strscache[v]
        return
    v_rdf = g.lookup_str(v) or ''
    node, typ = util.rdf_datatype(v_rdf)
    if typ == 'str' or typ.startswith('@'):
        try:
            node = node.encode().decode('unicode-escape')
        except Exception as e:
#             print('Problem with [%s]' % node)
            pass
        for variant in util.label_variants(node):
            s_strscache.setdefault(v, set()).add(variant)
            yield variant
    elif typ == 'uri':
        for s_label in g.o(v, label_id):
            labels = set(get_s_strs(s_label))
            cache = s_strscache.setdefault(v, set())
            cache |= labels
            yield from labels

# # inverse index?
# s_index = {}
# def fill_s_index(s, inlinks=True):
#     strlink_link, token_strlinks = {}, {}
#     strlinks = ((d,p,v,l) for d,p,v in get_s_links(s, inlinks=inlinks) for l in get_s_strs(v))
#     for i,(d,p,v,lbl) in enumerate(strlinks):
#         toks = util.tokenize(lbl)
#         strlink_link[i] = (d,p,v,lbl, len(toks))
#         for tok in toks:
#             token_strlinks.setdefault(tok, set()).add(i)
#     s_index[s] = (strlink_link, token_strlinks)

# from collections import Counter
# def lookup_s_index(s, cellval, inlinks=True):
#     if s not in s_index:
#         fill_s_index(s, inlinks=inlinks)
#     strlink_link, token_strlinks = s_index[s]
#     toks = util.tokenize(cellval)
#     toklen = len(toks)
#     link_tokcount = Counter()
#     for tok in toks:
#         link_tokcount.update(token_strlinks.get(tok, []))
#     for i,c in link_tokcount.most_common():
#         d,p,v, lbl,l = strlink_link.get(i, (0,-1,-1,'',0))
#         tokenjacc = c / (toklen + l - c)
#         yield tokenjacc, (d,p,v, lbl)
            
# Loop over candidates
def match(tablename, webtable, candidates, inlinks=True, p_redir={}):
    tablerow_cells = {}
    for r, cells in enumerate(webtable.rows_split):
        rowname = f'{tablename}~Row{r:d}'
        tablerow_cells[rowname] = cells

    matches = []
    link_rows = {}
    for row in candidates.itertuples():
        s, s_uri, tablename, rownr = row.id, row.entity, row.table, row.rownr
        if s == -1: continue
        print(f'{row.row:>50s} : {s_uri} ', end='\r', file=sys.stderr)

        rowcells = tablerow_cells.get(row.row, [])
        if not cells:
            print('Missing row: %s' % row.row)
            continue

        # Get link [bottleneck: link loops]
        for d,p,v in get_s_links(s, inlinks=inlinks):
            link_rows.setdefault((d,p,v), set()).add(rownr)
            for colnr,cell in enumerate(rowcells):
                if webtable.col_isnumeric.get(colnr, False):
                    continue # skip numeric cols
                
                besttokenjacc, bestcellval = 0, None
                # Compare attribute values to cell values
                for cellval in cell:
                    for linklabel in get_s_strs(v):
                        tokenjacc = util.tokenjaccard(linklabel, cellval)
                        if tokenjacc > besttokenjacc:
                            besttokenjacc, bestcellval = tokenjacc, cellval
                if bestcellval:
                    linkcode = '%d%s%d' % (p,'+' if d==1 else '-',v)
                    match = (colnr, rownr, s, s_uri, linkcode, linklabel, bestcellval, besttokenjacc, p*d)
                    matches.append( match )

#         for colnr,cell in enumerate(rowcells):
#             if webtable.col_isnumeric.get(colnr, False):
#                 continue # skip numeric cols
#             for cellval in cell:
#                 for tokenjacc, (d,p,v,linklabel) in lookup_s_index(s, cellval):
#                     linkcode = '%d%s%d' % (p,'+' if d==1 else '-',v)
#                     match = (colnr, rownr, s, s_uri, linkcode, linklabel, cellval, tokenjacc, p*d)
#                     matches.append( match )
#                     link_rows.setdefault((d,p,v), set()).add(rownr)
                
    print(file=sys.stderr)
    columns = ['col', 'row', 'i', 'uri', 'link', 'label', 'cell', 'tokenjaccard', 'p']
    matches = pd.DataFrame(matches, columns=columns)

    # Filter matches on (rowcount > 1)
    colpred_rowcount = matches[['col','row','p']].drop_duplicates().groupby(['col','p']).size().to_frame('rowcount')
    matches = matches.merge(colpred_rowcount[colpred_rowcount['rowcount']>1], on=['col','p'])

    # Filter links on (rowcount > 1) and (kbcount > 1)
    link_rows = {link:len(rows) for link, rows in link_rows.items()}
    linktable = pd.DataFrame([
        ('%d%s%d' % (p,'+' if d==1 else '-',v), g.lookup_str(p),g.lookup_str(v),count_link(d,p,v), rowcount)
            for (d,p,v), rowcount in link_rows.items()
            if rowcount>1 and count_link(d,p,v)>1
        ],                    
        columns=['link', 'p', 'v', 'count', 'rowcount']
    )

    # Filter links on (rowcount > 1) and (kbcount > 1)
    entlinktable = pd.DataFrame([
        (s,'%d%s%d' % (p,'+' if d==1 else '-',v), 1) 
            for s in set(matches['i']) for d,p,v in s_linkscache.get(s,[])
            if link_rows.get((d,p,v),0)>1 and count_link(d,p,v)>1
        ],
        columns = ['i', 'link', 'weight']
    )

    return matches, linktable, entlinktable

if __name__ == '__main__':
    import sys, os.path, json
    try:
        _, table_fnames, candidate_fnames, DB_PATH, INLINKS, outdir = sys.argv
        table_fnames = open(table_fnames).read().split()
        table_names = set([os.path.basename(fname) for fname in table_fnames])
        candidate_fnames = open(candidate_fnames).read().split()
        inlinks = bool(int(INLINKS))

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
    
    p_redir = {
        g.lookup_id('<http://dbpedia.org/ontology/wikiPageRedirects>'): label_id,
        g.lookup_id('<http://dbpedia.org/ontology/wikiPageDisambiguates>'): label_id,
        g.lookup_id('<http://dbpedia.org/ontology/wikiPageWikiLinkText>'): label_id
    }
    
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
#     ids = set(s for s in candidates['id'] if s != -1)
#     n = len(ids)
#     for i,s in enumerate(set(candidates['id'])):
#         if not i % 100: print(f'Filling string filter for entity {i:6d} / {n:6d}', end='\r', file=sys.stderr)
#         fill_s_index(s)
    sys.stdout.flush()
    
    ### MATCH CELLS TO LINKS ###
    gr = candidates.groupby('table')
    for tablename, pertable_idx in gr.indices.items():
        webtable = webtables.get(tablename, None)
        if not webtable:
            print('Missing table: %s' % row['table'])
            continue
        matchtable, linktable, entlinktable = match(tablename, webtable, gr.get_group(tablename).copy(), inlinks=inlinks, p_redir = p_redir)
        print('%50s : %4d rows; %4d candidates; %5d matches; %5d unique links; %5d entity-links' % 
              (tablename, len(webtable.rows), len(set(matchtable['i'])), len(matchtable), len(linktable), len(entlinktable)))
        
        matchtable.to_csv(os.path.join(matchdir, tablename), index=False)
        linktable.to_csv(os.path.join(linkdir, tablename), index=False)
        entlinktable.to_csv(os.path.join(entitydir, tablename), index=False)
