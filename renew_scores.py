import pandas as pd

def ReNew(join, verbose=False):
    out = {}
    def frac(name, a,b):
        a = len(join[a])
        b = len(join[b])
        value = (a / b) if b else 0
        if verbose:
            print('%-20s: %-5d / %-5d = %.2f' % (name, a, b, value))
        out[name] = value
    frac('positive_redundancy',
        (join.kb==True) & (join.gold==True) & (join.pred==True),
        (join.gold==True) & (join.pred==True) )
    frac('negative_redundancy',
        (join.kb==True) & (join.gold==True) & (join.pred==False),
        (join.gold==True) & (join.pred==False) )
    frac('novel_recall',
        (join.kb==False) & (join.gold==True) & (join.pred==True),
        (join.kb==False) & (join.gold==True) )
    frac('redundant_recall',
        (join.kb==True) & (join.gold==True) & (join.pred==True),
        (join.kb==True) & (join.gold==True) )

    frac('novel_precision',
        (join.kb==False) & (join.gold==True) & (join.pred==True),
        (join.kb==False) & (join.pred==True) )
    return out

if __name__ == '__main__':
    import sys
    
    try:
        _, gold_triples, pred_triples = sys.argv
    except:
        print('Usage: python3 renew_scores.py <gold_triples.csv> <pred_triples.csv>')
        sys.exit(0)
        
    g = pd.read_csv(gold_triples)
    p = pd.read_csv(pred_triples)
    
    kbo = {} # kb objects
    # collect kb presence for gold, for UNIQUE triples
    g.columns = [c.replace(' ','_') for c in g.columns]
    g_map = {}
    g_tables = {}
    for r in g.itertuples():
        triple = (r.Subject_URI, r.Predicate_URI, r.Object_Value)
        g_map[triple] = r.Object_Value_matches_KB
        g_tables.setdefault(r.Table, set()).add(triple)
        kbo[triple] = (r.Object_Value_in_KB, r.Object_Value_Similarity)
    
    print('gold:', len(g_map))
    
    # collect kb presence for pred, for UNIQUE triples
    p.columns = [c.replace(' ','_') for c in p.columns]
    p_map = {}
    p_tables = {}
    for r in p.itertuples():
        triple = (r.Subject_URI, r.Predicate_URI, r.Object_Value)
        p_map[triple] = r.Object_Value_matches_KB
        p_tables.setdefault(r.Table, set()).add(triple)
        kbo[triple] = (r.Object_Value_in_KB, r.Object_Value_Similarity)
    
    print('pred:', len(p_map))
    
    # outer join
    j = pd.DataFrame([dict(
        triple=t, 
        gold=(t in g_map), 
        pred=(t in p_map), 
        kb=(g_map.get(t, False) or p_map.get(t, False)))
            for t in set(p_map)|set(g_map)])
    j.set_index('triple', inplace=True)

    # ReNew
    renew = ReNew(j, verbose=True)
    
    # get scores per table
    t = []
    for table in set(g_tables) | set(p_tables):
        tj = j.loc[list( g_tables.get(table, set()) | p_tables.get(table, set()) )]
        row = ReNew(tj, verbose=False)
        row['table'] = table
        t.append(row)
    t = pd.DataFrame(t)
    t.set_index('table', inplace=True)
    
    # Get mistakes (false positive) & misses (false negative)
    t['novel_f1'] = 2 / ((1/t['novel_recall']) + (1/t['novel_precision']))
    t.sort_values('novel_f1', inplace=True, ascending=True)
    
    grouped_fact_pos = []
    grouped_fact_neg = []
    for score in t.itertuples():
        table = score.Index
        # get joined triples for this table
        tj = j.loc[list( g_tables.get(table, set()) | p_tables.get(table, set()) )]
        tj = tj[(tj.kb==False) & ((tj.pred==False) | (tj.gold==False))]
        if len(tj):
            # sort by predicate
            _, tj['p'], _ = tuple(zip(*tj.index))
            tj.sort_values('p', inplace=True)
            for triple in tj.itertuples():
                s,p,o = triple.Index
                s = s.replace('http://dbpedia.org/resource/', '')
                p = p.replace('http://dbpedia.org/ontology/', '')
                o_kb, o_kb_sim = kbo.get(triple.Index, (None,None))
                row = (
                    table,
                    score.novel_precision, score.novel_recall, score.novel_f1,
                    s,p,o, 
                    o_kb, o_kb_sim
                )
                if triple.pred:
                    grouped_fact_pos.append( row )
                else:
                    grouped_fact_neg.append( row )

    col_names = 'table np nr nf1 s p o okb okb_sim'.split()
    
    grouped_fact_pos = pd.DataFrame(grouped_fact_pos, columns=col_names)
    grouped_fact_neg = pd.DataFrame(grouped_fact_neg, columns=col_names)
    
    import os
    dirname = os.path.dirname(pred_triples)
    basename = os.path.basename(pred_triples)
    grouped_fact_pos.to_csv( os.path.join(dirname, basename.replace('extracted_triples', 'falsepos')) )
    grouped_fact_neg.to_csv( os.path.join(dirname, basename.replace('extracted_triples', 'falseneg')) )
    
    
    
    
    