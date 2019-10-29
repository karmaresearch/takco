"""
Table Interpretation
Usage: match.py KBPATH DBPATH TABLEFILE KEYCOL OUTROWFILE OUTCOLFILE
"""
import os, sys, csv, sqlite3
import trident
import pandas as pd
import numpy as np

def query_index_key(q, label_p):
    q = q.replace('"','""')
    return pd.read_sql("""
    WITH
    LabelTriples AS (
        SELECT * FROM Triples WHERE (Triples.p = :label_p)
    ), 
    MatchTriples AS (
        SELECT l1.rank, l1.lbl AS key, t2.*
        FROM Labels AS l1
        JOIN LabelTriples AS t1 ON (l1.rowid = t1.o)
        JOIN Triples AS t2 ON (t1.s=t2.s)
        WHERE (l1.lbl MATCH :query)
    )
    SELECT m.*, l.lbl as context FROM MatchTriples AS m JOIN Labels AS l ON (m.o=l.rowid)
    UNION
    SELECT m.*, l.lbl as context FROM MatchTriples AS m JOIN LabelTriples AS t ON (m.o=t.s) JOIN Labels AS l ON (l.rowid=t.o)
    ORDER BY rank
    """, db, params={'query':f'"{q}"', 'label_p':label_p})

def tokenize(s):
    return s.lower().split()
def jacc(a,b):
    return len(a&b) / len(a|b)

def match_context(row, keycol, max_candidates, label_p):
    kmatch = query_index_key(row[keycol], label_p)
    cells = kmatch['context']
    c_toks = [set(tokenize(c) if c else []) for c in row]

    context_c_jacc = []
    for s in set(cells):
        t = set(tokenize(s))
        for i,c in enumerate(c_toks):
            j = jacc(t,c)
            if j:
                context_c_jacc.append((s, i, j))

    cmatch = pd.DataFrame(context_c_jacc, columns='context col jacc'.split())
    match = kmatch.merge(cmatch, on='context').sort_values('rank')    
    match['jacc'] *= 1 - np.exp(match['rank']+1)
    
    bestent = set(match.groupby('s')['jacc'].max().nlargest(max_candidates).index)
    match = match[match['s'].apply(lambda s:s in bestent)]
    return match

def get_linkscore(alls, link_ents, ent_rows, row_rowscore):
    linkscore = {}
    for (p,o),es in link_ents.items():
        es = list(es)
        linktotal = sum( max(rs.get((r,e),0) for e in es) 
                        for r,rs in row_rowscore.items() )
        nrows = len(set(r for e in es for r in ent_rows[e]))
        nkb = (g.n_s(p,o) if o else g.count_p(p))

        cover = linktotal / nrows
        salience = linktotal / nkb
        linkscore[(p,o)] = cover * salience
    return linkscore

def get_colscore(match):
    cellscore = match.groupby(['col','row','p']).mean() # over entities
    cellscore_total = cellscore.groupby(['col', 'p']).sum() # over rows
    return cellscore_total / cellscore_total.groupby(['col']).transform('sum')
        
def disambiguate(rows, keycol, max_candidates, label_p):
    # Get matches
    rowmatches = []
    for r, row  in enumerate(rows):
        match = match_context(row, keycol, max_candidates, label_p)
        if len(match):
            # enforce label p
            match = match[(match['col'] != keycol) | (match['p'] == label_p)]
            print(f'Matched row {r}: {len(match):>6d} matches', file=sys.stderr, end='\r')
            match['row'] = r
            rowmatches.append( match )
    rowmatches = pd.concat( rowmatches ).drop_duplicates()
    
    # Calculate entity-row score
    match = rowmatches.groupby(['col','row','s','p'])['jacc'].max() # over labels
    colscore = get_colscore(match)
    df = match.to_frame().join(colscore.to_frame('colscore'), on=['col', 'p'])
    rowscore = ((df['jacc'] * df['colscore']).drop_duplicates()
                .groupby(['col','row','s']).max()
                .groupby(['row','s']).mean())

    # Get link scores
    row_s = rowscore.to_frame().reset_index()[['row','s']]
    ent_rows = dict(row_s.groupby('s')['row'].agg(set))
    row_rowscore = dict(rowscore.groupby('row').agg(dict))
    alls = set(ent_rows)
    
    link_ents, ent_links = {}, {}
    for s in alls:
        for (p,o) in g.po(s):
            for l in [(p,o), (p,None)]:
                link_ents.setdefault( l, set() ).add(s)
                ent_links.setdefault( s, set() ).add(l)
    
    linkscore = get_linkscore(alls, link_ents, ent_rows, row_rowscore)
    
    # Get entity sims
    entsim = []
    for e1 in alls:
        for e2 in alls:
            if e1 != e2:
                sim = sum(linkscore[l] for l in ent_links[e1] | ent_links[e2] )
                entsim.append((e1, e2, sim))
    entsim = pd.DataFrame(entsim, columns=['s1', 's2', 'sim']).set_index(['s1','s2'])
    
    # Belief Propagation
    S = entsim.unstack().fillna(0)
    L = rowscore.unstack().fillna(0)
    L = (L.T / L.T.sum()).T
    q = (L.dot(S)+1).stack().groupby('s2').agg('prod')['sim']
    q /= q.max()
    L *= q
    
    rowscore = (L.T / L.T.sum()).T.stack()
    match = (match * rowscore).dropna()
    colscore = get_colscore(match)
    
    return rowscore, colscore

def bestcolscore(colscore):
    # Greedy best columns
    cs = colscore.to_frame('score').reset_index()
    cs = cs.sort_values(['col','score'], ascending=False)
    best = []
    while len(cs):
        _, bestcol = next(cs.nlargest(1, columns=['score']).iterrows())
        best.append(bestcol)
        cs = cs[(cs['col'] != bestcol.col) & (cs['p'] != bestcol.p)]
    best = pd.DataFrame(best).sort_values('col')
    best['p'] = [int(p) for p in best['p']]
    best['col'] = [int(p) for p in best['col']]
    best['uri'] = [g.lookup_str(p) for p in best['p']]
    return best

def bestrowscore(rowscore):
    best = rowscore[(rowscore>0) & (rowscore == rowscore.groupby('row').transform('max'))]
    best = best.to_frame('score').reset_index()
    best['uri'] = [g.lookup_str(s) for s in best['s']]
    return best

if __name__ == '__main__':
    try:
        _, KBPATH, DBPATH, TABLEFILE, KEYCOL, OUTROWFILE, OUTCOLFILE = sys.argv

        g = trident.Db(KBPATH)
        LABEL_P = g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>')

        db = sqlite3.connect(DBPATH)
        db.row_factory = lambda x,y: dict(sqlite3.Row(x,y))

        MAX_CANDIDATES = 100

        KEYCOL = int(KEYCOL)
        ROWS = list(csv.reader(open(TABLEFILE)))
        
        rowscore, colscore = disambiguate(ROWS, KEYCOL, MAX_CANDIDATES, LABEL_P)
        bestrowscore(rowscore).to_csv(OUTROWFILE, index=None, float_format='%.8f')
        bestcolscore(colscore).to_csv(OUTCOLFILE, index=None, float_format='%.8f')

    except Exception as e:
        print(__doc__)
        raise e
