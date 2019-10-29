"""
Calculates column-predicate prediction scores: precision, recall, f1, tradeoff plots
Usage: python prop-analysis.py [gold dir] [t2k output dir] [sti dir] [our dir]
"""

import pandas as pd
import util

if __name__ == '__main__':
    import sys, os, glob
    try:
        
        gold_path = sys.argv[1]
        t2k_path = sys.argv[2]
        sti_path = sys.argv[3]
        results_path = sys.argv[4]
        
        gold_path = os.path.realpath(os.path.join(gold_path, 'gs_property.csv'))
        t2k_path = os.path.realpath(os.path.join(t2k_path, 'schema_correspondences.csv'))
        sti_path = os.path.realpath(os.path.join(sti_path, 'property', '*.csv'))
        our_path = os.path.realpath(os.path.join(results_path, 'property', '*.csv'))

        
    except Exception as e:
        print(__doc__)
        raise e
        
    
    
    gs = pd.read_csv(gold_path, header=None, names=['col', 'uri', 'gold'])
    gs['table'] = gs['col'].map(lambda c: c.split('.csv~Col')[0])
    gs['colnr'] = gs['col'].map(lambda c: int(c.split('.csv~Col')[1]))
    gs['gold'] = gs['gold'].map(int)
    gs.drop(columns=['col'], inplace=True)
    
    # Limit the set of predicates to those that are valid in the set of annotations
    valid_uris = set(gs['uri'])
    
    t2k = pd.read_csv(t2k_path, header=None, names=['col', 'uri', 't2k-final'])
    t2k['table'] = t2k['col'].map(lambda c: c.split('.')[0])
    t2k['colnr'] = t2k['col'].map(lambda c: int(c.split('~Col')[1]) if len(c) else 0)
    t2k.drop(columns=['col'], inplace=True)
    t2k = t2k[t2k['uri'].map(lambda uri: uri in valid_uris)]
    print('t2k', len(t2k))
    if not len(t2k):
        t2k = pd.DataFrame([], columns=['table', 'colnr', 'uri', 't2k-final'])
    
    sti = pd.concat([pd.read_csv(fname) for fname in glob.glob(sti_path) if os.stat(fname).st_size>1], sort=True)
    sti = sti[sti['uri'].map(lambda uri: uri in valid_uris)]
    sti = sti[sti.groupby(['table', 'fromColumnIndex', 'toColumnIndex'])['score'].transform('max') == sti['score']]
    sti.set_index('table', inplace=True)
    key_cols = (gs['uri'] == 'http://www.w3.org/2000/01/rdf-schema#label')
    sti['keycol'] = gs[key_cols].set_index('table')['colnr']
    sti = sti[sti['fromColumnIndex'] == sti['keycol']].reset_index()
    sti = sti.rename(columns={'toColumnIndex':'colnr', 'score':'sti'})
    sti = sti[['table', 'colnr', 'uri', 'sti']]
    print('sti', len(sti))
    if not len(sti):
        sti = pd.DataFrame([], columns=['table', 'colnr', 'uri', 'sti'])
    
    
    our = []
    for fname in glob.glob(our_path):
        if os.stat(fname).st_size>1:
            df = pd.read_csv(fname)
            try:
                df = df[df['p'] == '+']
                df['uri'] = df['uri'].map(lambda x:x[1:-1])
                df = df[df['uri'].map(lambda uri: uri in valid_uris)]
                df = df[df.groupby('col')['colscore'].transform('max') == df['colscore']]
                df['table'] = os.path.basename(fname).replace('.csv','').replace('.json','')
                our.append(df.rename(columns={'col':'colnr', 'colscore':'our'}).drop(columns=['p']))
            except Exception as e:
                print(e)
    our = pd.concat(our, sort=True)
    
    
    score_tables = [gs, t2k, sti, our]
    from functools import reduce
    joined = reduce(lambda left, right: left.merge(right, how='outer', on=['table','colnr','uri']), score_tables)
    joined = joined.set_index(['table', 'colnr', 'uri']).fillna(0)
    
    joined['sti'] = joined['sti'] / joined['sti'].max()
    joined['our'] = joined['our'] / joined['our'].max()    
    joined.to_csv(os.path.join(results_path, 'joined_property.csv'))
    
    scores = {}
    grouped = joined.groupby(['table','colnr'])
    hasgold = grouped['gold'].transform('sum').map(bool)
    for s in joined.columns:
        if s is not 'gold':
            best_per_col = joined[hasgold & (grouped[s].transform('max') == joined[s])]
            scores[s] = util.get_pr_curve_scores(best_per_col, 'gold', s)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.subplots(1)
    for s, (p,r,f,ap, n_error, n_pred, p_curve, r_curve) in scores.items():
        print(s, 'p:%.2f r:%.2f f:%.2f ap:%.2f'%(p,r,f,ap))
        ax.step(r_curve[1:-1], p_curve[1:-1], where='post', label=s)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    # ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2))
    fig.savefig(os.path.join(results_path,'property_pr.png'), format='png')
    fig.savefig(os.path.join(results_path,'property_pr.pdf'), format='pdf')
    
    