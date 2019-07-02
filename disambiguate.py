"""
Disambiguates row-entity candidate sets using cell-KB matches
Usage: python disambiguate.py [table fnames file] [bp iterations:int] [outdir]
"""

import os,re 
import pandas as pd
import numpy as np

def colscore(match_scores):
    cellscore = match_scores.groupby(level=['col','row','p']).agg('mean')
    score = cellscore.groupby(level=['col','p']).agg('sum') / cellscore.sum()
    return score.to_frame('colscore').reset_index()

def coherence(matches, entities, links, bp_it=1):
    matches = matches.copy()
    matches['labelscore'] = matches['tokenjaccard']
    
    for i in set(matches['i']) - set(entities['i']):
        matches = matches[matches['i'] != i]
    for i in set(entities['i']) - set(matches['i']):
        entities = entities[entities['i'] != i]
    
    matches['p'] = matches['link'].map(lambda f: ''.join(re.split('(\+|-)', f, 1)[0:2]) )

    # col score
    match_scores = matches.set_index(['col','row','p', 'i'])['labelscore']
    P = colscore(match_scores)
    matches = matches.merge(P, how='left')

    # label score
    matches['labelscore*colscore'] = matches['labelscore'] * matches['colscore']
    # L is sum (over cols) of max (over preds) of score * colscore
    L = (matches.groupby(['row','i','col'])['labelscore*colscore'].agg('max').groupby(level=['row','i']).agg('sum'))
    matches = matches.drop(columns=['labelscore']).merge(L.to_frame('labelscore').reset_index())

    # merge links / candidates
    entlink = entities.merge(L.to_frame('labelscore').reset_index())
    entlink['weight*labelscore'] = entlink['weight'] * entlink['labelscore']
    # F is sum (over rows) of max (over candidates) of score * labelscore
    F = entlink.groupby(['link','row'])['weight*labelscore'].agg('max').groupby(level=['link']).agg('sum')


    # weight links
    if 'linktotal' in links.columns:
        links.drop(columns=['linktotal'], inplace=True)
    links = links.merge(F.to_frame('linktotal').reset_index())
    links['cover'] = links['linktotal'] / links['rowcount']
    links['salience'] = links['linktotal'] / links['count']
    links['linkscore'] = links['cover'] * links['salience']

    ## Calculate coherence
    es = entities.set_index(['i','link'])['weight']
    fs = links.set_index(['link'])['linkscore']
    # Similarity Matrix: calculate entity similarities from weighted in-links and out-links
    S = es.unstack().fillna(0).dot( (es * fs).unstack().fillna(0).T ) 
    np.fill_diagonal( S.values, 0 )
    # Belief Propagation: calculate how similar each entity is to all others
    C = L.copy()
    for bp_i in range(bp_it):
        # align matrices
        C_, S_ = C.unstack().fillna(0), S.reindex(C.index.levels[1]).T.reindex(C.index.levels[1]).T
        q = (C_.dot(S_)+1).applymap(np.log).sum() # log-sum trick

        q /= q.max() # normalize
        C = L*q # multiply prior

        match_scores = matches.set_index(['col','row','p', 'i'])['labelscore']
        pred = C[C.groupby('row').transform('max') == C]
        m = match_scores.to_frame().reset_index().merge( pred.to_frame('C').reset_index() )
        P = colscore( m.set_index(['col','row','p', 'i'])['labelscore'] )

        yield bp_i, L, C, P, links.copy()
    matches = matches.drop(columns=['labelscore']).merge(C.to_frame('labelscore').reset_index())


if __name__ == '__main__':
    import sys, os.path, json
    try:
        _, table_fnames, N_BP_ITERATIONS, outdir = sys.argv
        N_BP_ITERATIONS = int(N_BP_ITERATIONS)
        table_fnames = open(table_fnames).read().split()
        table_names = set([os.path.basename(fname) for fname in table_fnames])

        def make_dir(name):
            path = os.path.join(outdir, name)
            os.makedirs(path, exist_ok=True)
            return path

        assert os.path.exists(outdir)

        linkdir = make_dir('links')
        entitydir = make_dir('entities')
        matchdir = make_dir('matches')

        instancedir = make_dir('instance')
        propertydir = make_dir('property')
        classdir = make_dir('class')

    except Exception as e:
        print(__doc__)
        raise e
    
    import glob
    for i, fname in enumerate(os.listdir(os.path.join(outdir, 'matches'))):
        if fname not in table_names:
            continue
        
        matches = pd.read_csv(os.path.join(outdir, 'matches', fname))
        if not os.path.exists(os.path.join(outdir, 'entities', fname)): continue
        entities = pd.read_csv(os.path.join(outdir, 'entities', fname))
        if not os.path.exists(os.path.join(outdir, 'links', fname)): continue
        links = pd.read_csv(os.path.join(outdir, 'links', fname))
        if not len(matches) or not len(entities) or not len(links):
            continue
        
        instances = matches[['row', 'i', 'uri']].drop_duplicates().set_index(['row', 'i'])
        
        try:
            iter_results = coherence(matches, entities, links, N_BP_ITERATIONS)
            for bp_i, L, C, P, F in iter_results:
                instances['labelscore_bp%d' % (bp_i)] = L
                instances['coherence_bp%d' % (bp_i)] = C
                instances['combined_bp%d' % (bp_i)] = C + L
        except Exception as e:
            print('Problem with table %s' % fname)
            raise e
            
        instances = instances.reset_index().drop(columns=['i'])
        instances['table'] = fname
        instances.to_csv(os.path.join(instancedir, fname), index=False)

        properties = links[['link', 'p']].rename(columns={'link':'p', 'p':'uri'})
        properties['p'] = properties['p'].map(lambda f: ''.join(re.split('(\+|-)', f, 1)[0:2]) )
        try:
            properties = properties.merge(P).drop_duplicates()
            properties['p'] = properties['p'].map(lambda x: x[-1])
            properties.sort_values('colscore', ascending=False)
        except Exception as e:
            print('Problem with table %s' % fname)
            raise e
            
        properties.to_csv(os.path.join(propertydir, fname), index=False)
        
        F.to_csv(os.path.join(outdir, 'links', fname), index=False)
        
        n_rows = len(set(matches['row']))
        
        print('%4d [%4d rows, %4d candidates, %4d NaN, %4d zero]  %s'%(i, n_rows, len(C), sum(C.isna()), sum(C==0), fname))

