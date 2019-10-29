"""
Calculates row-entity prediction scores: precision, recall, f1, tradeoff plots
Usage: python error-analysis.py [gold dir] [t2k instance dir] [sti dir] [our dir] [redir file]
"""

import os, glob, urllib.parse
import pandas as pd
import util

def load_gold(gold_path, redir={}):
    gold = pd.read_csv(os.path.join(gold_path,'gs_instance.csv'), header=None, names=['row', 'uri', 'gold'])
    gold['table'], gold['rownr'] = zip(*(s.split('.csv~Row') for s in gold['row']))
    gold['rownr'] = gold['rownr'].map(int)
    return gold[['table','rownr','uri','gold']]

def load_t2k(t2k_path, redir={}):
    t2k_glob = glob.glob(os.path.join(t2k_path,'*.csv'))
    t2k = pd.concat([pd.read_csv(fname) for fname in t2k_glob], sort=True)
    t2k['table'], t2k['rownr'] = zip(*(s.split('.csv~Row') for s in t2k['row']))
    t2k['rownr'] = t2k['rownr'].map(int)
    t2k = t2k.rename(columns={'entity':'uri', 'score':'t2k-final'})
    return t2k[['table','rownr','uri','t2k-final']]
    
def load_sti(sti_path, gold_path, redir={}):
    gold_prop_fname = os.path.join(gold_path,'gs_property.csv') # needed for gold subject column
    gs_property = pd.read_csv(gold_prop_fname, header=None, names=['col', 'uri', 'gold'])
    gs_property['table'], gs_property['columnIndex'] = zip(*(s.split('.csv~Col') for s in gs_property['col']))
    gs_property['columnIndex'] = gs_property['columnIndex'].map(int)
    key_cols = (gs_property['uri'] == 'http://www.w3.org/2000/01/rdf-schema#label')

    sti_glob = glob.glob(os.path.join(sti_path,'instance/*.xml.csv'))
    sti = pd.concat([pd.read_csv(fname) for fname in sti_glob if os.stat(fname).st_size>1], sort=True)
    sti = sti.merge(gs_property[key_cols][['table','columnIndex']]) # filter preds on gold key cols
    sti = sti.rename(columns={'rowIndex':'rownr', 'score':'sti'})
    return sti[['table', 'rownr', 'uri', 'sti']]

def load_our(our_path, redir={}):
    our_glob = glob.glob(os.path.join(our_path,'instance/*.csv'))
    our = pd.concat([pd.read_csv(fname) for fname in our_glob if os.stat(fname).st_size>1], sort=True)
    our['table'] = our['table'].map(lambda t: t.replace('.csv',''))
    our = our.rename(columns={'row':'rownr'})
    return our

from functools import reduce
def join_results(score_tables):
    join_instance = reduce(lambda left, right: left.merge(right, how='outer'), score_tables)
    join_instance['gold'].fillna(False, inplace=True)
    join_instance.fillna(0, inplace=True)
    return join_instance.set_index(['table', 'rownr', 'uri'])


import util

def get_table_scores(joined_instance, name):
    gr = joined_instance.groupby('table')
    table_scores = [] 
    for table, group in gr:
        grouped = group.groupby(['rownr'])
        hasgold = grouped['gold'].transform('sum').map(bool)
        
        best_per_row = group[(hasgold & (grouped[name].transform('max') == group[name])) | (group['gold']>0)].copy()
        best_per_row.loc[(best_per_row.groupby('rownr')[name].transform('max') != best_per_row[name]), name] = 0
        
        p,r,f,ap, n_correct,n_pred,n_gold, p_curve,r_curve = util.get_pr_curve_scores(best_per_row, 'gold', name)
        
        table_scores.append( (table, p,r,f,ap, n_correct,n_pred,n_gold, ) )
    return pd.DataFrame(table_scores, 
        columns=['table', 'precision', 'recall', 'f1', 'average_precision', 'n_correct', 'n_pred', 'n_gold'])


if __name__ == '__main__':
    import sys
    try:
        _, gold_path, t2k_path, sti_path, our_path, redir_file = sys.argv
        
    except Exception as e:
        print(__doc__)
        raise e
        
    from distutils.dir_util import copy_tree
    copy_tree(os.path.join(gold_path,"tables"), os.path.join(our_path, "tables"))
    
    redir = {}
    for i,line in enumerate(open(redir_file)):
        source, _, target = line.strip()[:-2].split(None, 2)
        source, target = urllib.parse.unquote(source[1:-1]), urllib.parse.unquote(target[1:-1])
        redir[source] = target
        if 0 == i % 100000:
            print('Loading redirects line %d    ' % i, end='\r', file=sys.stderr)
    
    ## INSTANCE SCORES ##
    print('loading row instance predictions...')
    gold = load_gold(gold_path)
    t2k = load_t2k(t2k_path)
    sti = load_sti(sti_path, gold_path)
    our = load_our(our_path)
    methods = [gold, t2k, sti, our]
    for m in methods:
        m['uri'] = m['uri'].map(lambda uri: redir.get(uri,uri))
    joined_instance = join_results(methods)
    joined_instance.fillna(0, inplace=True)
    
    
    cols = ['gold', 't2k-final', 'sti', 'labelscore_bp0', 'coherence_bp0', 'combined_bp0']
    joined_instance = joined_instance[[c for c in cols if c in joined_instance.columns] + 
                                      [c for c in joined_instance.columns if c not in cols]]
    
    
    joined_instance.to_csv(os.path.join(our_path, 'joined_instance.csv'))
    
    print('calculating precision & recall...')
    scores = {}
    grouped = joined_instance.groupby(['table','rownr'])
    print('%d groups' % len(grouped))
    hasgold = grouped['gold'].transform('sum').map(bool)
    for s in joined_instance.columns:
        if s is not 'gold':
            best_per_row = (grouped[s].transform('max') == joined_instance[s])
            best_per_row = joined_instance[(hasgold & best_per_row) | (joined_instance['gold']>0) ].copy()
            best_per_row.loc[(best_per_row.groupby(['table','rownr'])[s].transform('max') != best_per_row[s]), s] = 0
            scores[s] = util.get_pr_curve_scores(best_per_row, 'gold', s)
    
    # Save Precision-Recall graph
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.subplots(1)
    
    for s, (p,r,f,ap, n_correct,n_pred,n_gold, p_curve, r_curve) in scores.items():
        print(s, 'p:%.2f r:%.2f f:%.2f ap:%.2f'%(p,r,f,ap))
        ax.step(r_curve[1:-1], p_curve[1:-1], where='post', label=s)
    ax.legend()
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.5, 1.0])
    # ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2))
    fig.savefig(os.path.join(our_path,'instance_pr.png'), format='png')
    fig.savefig(os.path.join(our_path,'instance_pr.pdf'), format='pdf')
    
    score_table = pd.DataFrame([(name,p,r,f,ap, n_correct,n_pred,n_gold)
        for name,(p,r,f,ap, n_correct,n_pred,n_gold, _,_) in scores.items()],
        columns=['name', 'precision', 'recall', 'f1', 'average_precision', 'n_correct','n_pred','n_gold'])
#     score_table['n_gold'] = sum(joined_instance['gold'])
    score_table.to_csv(os.path.join(our_path,'instance_scores.csv'), index=False)
    import json
    json.dump({
        name:dict(
            p=p,r=r,f=f,ap=ap,
            n_correct=n_correct,n_pred=n_pred,n_gold=n_gold,
            p_curve=list(p_curve),r_curve=list(r_curve))
        for name,(p,r,f,ap, n_correct,n_pred,n_gold, p_curve,r_curve) in scores.items()
    }, open(os.path.join(our_path,'instance_scores.json'), 'w'))
    
    
    table_scores = pd.DataFrame({s: get_table_scores(joined_instance, s).set_index('table').stack()
                  for s in joined_instance.columns if s is not 'gold'}).unstack()

    gold_class_fname = os.path.join(gold_path,'gs_class.csv')
    if os.path.exists(os.path.join(gold_path,'gs_class.csv')):
        gs_class = pd.read_csv(gold_class_fname, header=None, 
                               names=['table', 'class', 'gold'])
        gs_class['table'] = gs_class['table'].map(lambda x: x.replace('.csv',''))
        gs_class.set_index('table', inplace=True)
        table_scores.insert(0, 'class', gs_class['class'])
    else:
        table_scores.insert(0, 'class', '?')
    
    table_scores.insert(1, 'n_gold', joined_instance.groupby('table')['gold'].agg('sum'))
    table_scores.to_csv(os.path.join(our_path,'instance_scores_pertable.csv'))

