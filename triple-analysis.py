import sys, os

try:
    _, results_dir = sys.argv
    results_dir = os.path.realpath(results_dir)
except Exception as e:
    print('Usage: triple-analysis.py [results_dir]')
    raise(e)
    
from functools import reduce
import pandas as pd


triple_dir = os.path.join(results_dir, 'triples')
triple_tables = []
methods = [m for m in os.listdir(triple_dir) if not m.endswith('.csv')]
for method in methods:
    t = pd.read_csv(os.path.join(results_dir, 'triples', method, 'extracted_triples-GOLD.csv'))
    t.rename(columns={'Subject Confidence':method}, inplace=True)
    triple_tables.append(t)
join_triples = reduce(lambda left, right: left.merge(right, how='outer'), triple_tables)
join_triples.fillna(0, inplace=True)
join_triples.to_csv(os.path.join(triple_dir, 'extracted_triples.csv'))


import collections
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

jt = join_triples.copy()
jt = jt.drop(columns=['URL', 'Table', 'Column Index']).drop_duplicates()

scores = {}
for redundant, ax in [(True, ax1), (False, ax2)]:
    for method in methods:
        if method != 'gold':
            j = jt[(jt['Object Value matches KB'] == redundant)]
            y_test = j['gold']
            y_score = j[method]
            p_curve, r_curve, _ = precision_recall_curve(list(y_test), list(y_score))
            ax.step(r_curve[1:-1], p_curve[1:-1], where='post', label='%s' % method)
            
#             f_curve = 2 / ((1/p_curve) + (1/r_curve))
#             t = f_curve[1:-1].argmax()+1
#             ax.plot([r_curve[t]], [p_curve[t]], marker='x', color="black")
#             ax.text(r_curve[t], p_curve[t], '$f_1$:%.2f' % f_curve[t])
            
            n_correct = sum((y_score > 0) & (y_test > 0))
            p = n_correct / sum(y_score > 0) if sum(y_score > 0) else 0
            r = n_correct / sum(y_test > 0) if sum(y_test > 0) else 0
            f = 2 / ((1/p) + (1/r)) if p and r else 0
            ap = average_precision_score(list(y_test), list(y_score))

            scores['%s %s'%(method, redundant)] = collections.OrderedDict(
                redundant=redundant, method=method,
                precision=p,recall=r,f1=f,ap=ap,
                p_curve=list(p_curve), r_curve=list(r_curve),
            )
            
    ax.set_title(['Novel', 'Redundant'][int(redundant)])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0,1])
ax.legend()



import json
json.dump(scores, open(os.path.join(results_dir,'triple_scores.json'), 'w'))

df = pd.DataFrame(list(scores.values()))
df = df[['redundant','method','precision','recall','f1','ap']]
df.columns = ['Redundant', 'System', 'Precision','Recall','$F_1$','AP']
df = df.pivot(index='System', columns='Redundant')
df.columns = df.columns.swaplevel(0, 1)
df.sortlevel(0, axis=1, inplace=True, sort_remaining=False)
df = df.reindex(index=df.index[::-1])

df.to_csv(os.path.join(results_dir,'triple_scores.csv'))
df.to_latex(os.path.join(results_dir,'triple_scores.tex'))

fig.savefig(os.path.join(results_dir,'triple_pr.png'), format='png')
fig.savefig(os.path.join(results_dir,'triple_pr.pdf'), format='pdf')


