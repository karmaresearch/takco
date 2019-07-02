EXP_DIR = './results/'


import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re

from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)

import util

import base64
from io import BytesIO

def plot_pr_curve(joined_instance):
    fig = plt.figure(figsize=(4,5))
    ax = plt.axes()
    
    grouped = joined_instance.groupby(['table','rownr'])
    hasgold = grouped['gold'].transform('sum').map(bool)
    for s in joined_instance.columns:
        if str(s) != 'gold':
            best_per_row = (grouped[s].transform('max') == joined_instance[s])
            best_per_row = joined_instance[hasgold & best_per_row]
            y_test = best_per_row['gold'].map(int)
            y_score = best_per_row[s]
            try:
                p_curve, r_curve, _ = precision_recall_curve(list(y_test), list(y_score))
                ax.step(r_curve[1:-1], p_curve[1:-1], where='post', label='%s' %s)
            except:
                pass
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.75, 1.0])
    ax.set_xlim([0.0, 1.0])
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0 + box.height*0.5,
#                      box.width, box.height*0.5])
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.2))
    
    figfile = BytesIO()
    fig.tight_layout()
    fig.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    return base64.b64encode(figfile.getvalue()).decode('utf8')



@app.route("/")
def index():
    experiments = []
    for fname in sorted(os.listdir(EXP_DIR)):
        try:
            n_tables = len(os.listdir(os.path.join(EXP_DIR, fname, 'instance')))
        except:
            n_tables = 0
        experiments.append(dict(
            name=fname, 
            n_tables=n_tables
        ))
    
    return render_template(
        'index.html', 
        experiments=experiments
    )

    
    
@app.route("/overview/<exp>")
def overview(exp):
    print(request.args)
    sort_score = request.args.get('sort_score', None)
    sort_method = request.args.get('sort_method', None)
    sort_asc = (request.args.get('sort_asc', None) == 'True') 
    
    exp_path = os.path.join(EXP_DIR, exp)
    
    joined_instance = pd.read_csv(os.path.join(exp_path, 'joined_instance.csv'), index_col=[0,1,2])
    score_table = pd.read_csv(os.path.join(exp_path,'instance_scores.csv'))
    score_plot_fname = os.path.join(exp_path,'instance_pr.png')
    table_instance_scores = pd.read_csv(os.path.join(exp_path,'instance_scores_pertable.csv'),
        header=[0,1],index_col=[0])
    
    # load results per table per method, extract class and ngold
    table_class = dict(table_instance_scores[('class','Unnamed: 1_level_1')])
    del table_instance_scores[('class','Unnamed: 1_level_1')]
    table_ngold = dict(table_instance_scores[('n_gold','Unnamed: 2_level_1')])
    del table_instance_scores[('n_gold','Unnamed: 2_level_1')]
    table_instance_scores.columns = table_instance_scores.columns.swaplevel(0,1)
    table_instance_scores_dict = {
        name: gr.reset_index().set_index('table').to_dict('index')
        for name, gr in table_instance_scores.stack().reset_index().groupby('level_1')
    }
    
    methods = [c for c in joined_instance.columns if c != 'gold']
    if sort_method and sort_score:
        df = pd.DataFrame.from_dict(table_instance_scores_dict[sort_method], orient='index')
        df['ap'] = df['average_precision']
        tables = df.sort_values(sort_score, ascending=(not sort_asc)).index # weird ascending behaviour: whatever
    else:
        tables = table_class
        
    tables = [t for t in tables if table_ngold.get(t,0)]
    
    return render_template(
        'overview.html',
        predpath=exp,
        score_plot=base64.b64encode(open(score_plot_fname, 'rb').read()).decode('utf8'),
        score_table=score_table.to_dict('records'),
        methods=methods,
        
        tables=list(tables),
        table_class=table_class,
        table_ngold=table_ngold,
        table_instance_scores=table_instance_scores_dict,
        sort_score=sort_score,
        sort_method=sort_method,
        sort_asc=sort_asc,
        
    )

import html, csv
def splitcells(cells):
    # T2KMatch table serialization reader
    cells = html.unescape(cells).replace('}','{').replace('\n', ' ').replace('\r', ' ')
    cells = [c.split('|') for c in next(csv.reader([cells], delimiter='|', quotechar='{'))]
    return cells


import json, urllib, csv
import numpy as np
def get_entropy(arr):
    arr = np.array([a for a in arr if a])
    arr = arr / arr.sum()
    return -sum(arr * np.log(arr))

@app.route('/table/<exp>/<table>')
def table(exp,table):
    exp_path = os.path.join(EXP_DIR, exp)
    display_no_gold = request.args.get('display_no_gold', False)
    
    # Preview
    header,*rows = csv.reader(open(os.path.join(exp_path, 'tables', '%s.csv'%table)))
    rows = [splitcells('|'.join(cells)) for cells in rows]
    
    # Instances
    joined_instance = pd.read_csv(os.path.join(exp_path, 'joined_instance.csv'))
    joined_instance_table = joined_instance[joined_instance['table'] == table]
    methods = [c for c in joined_instance_table.columns if c not in ['table', 'rownr', 'uri', 'gold']]
    print(request.args)
    method1 = request.args.get('method1', methods[-1]) # default combined
    method2 = request.args.get('method2', methods[0]) # default t2k
    scores = {}
    for m in methods:
        grouped = joined_instance_table.groupby(['rownr'])
        hasgold = grouped['gold'].transform('sum').map(bool)
        try:
            m_isbest = (grouped[m].transform('max') == joined_instance_table[m])
            best_per_row = joined_instance_table[(hasgold & m_isbest) | (joined_instance_table['gold']>0)].copy()
            best_per_row.loc[(best_per_row.groupby('rownr')[m].transform('max') != best_per_row[m]), m] = 0
            scores[m] = util.get_pr_curve_scores(best_per_row, 'gold', m)
        except:
            scores[m] = (0,0,0,0, 0,0, [], [])
    
    entropy = {}
    correct1, correct2 = {},{}
    for r,gr in joined_instance_table.groupby('rownr'):
        entropy[r] = {}
        i = gr[method1].idxmax()
        correct1[r] = (gr.loc[i][method1] > 0 and gr.loc[i]['gold']) if i else False
        entropy[r][method1] = np.exp(-get_entropy(gr[method1].fillna(0)))
        i = gr[method2].idxmax()
        correct2[r] = (gr.loc[i][method2] > 0 and gr.loc[i]['gold']) if i else False
        entropy[r][method2] = np.exp(-get_entropy(gr[method2].fillna(0)))
        
    print(len(joined_instance_table), 'joined instances', '%d gold' % (joined_instance_table['gold'].sum()))
    
    instance_table = pd.read_csv(os.path.join(exp_path, 'instance', '%s.csv'%table))
    instance_table['rownr'] = instance_table['row']
    instance_table['table'] = instance_table['table'].map(lambda x: x.replace('.csv', ''))
    instance_table = instance_table.merge(joined_instance_table, on=['rownr','uri'], suffixes=('','_'), how='outer')
    instance_table['gold'].fillna(False, inplace=True)
    print(len(instance_table), 'instances', '%d gold' % (instance_table['gold'].sum()))
    instance_table['name'] = instance_table['uri'].map(lambda u: u.split('/')[-1])
    candidates = {r: gr for r,gr in instance_table.groupby('rownr')}
    hasgold = instance_table.groupby('rownr')['gold'].agg(sum).to_dict()
    print('%d gold' % (sum(hasgold.values())))
    row_gold = {r: (gr.loc[gr['gold'].idxmax()]['uri'] if any(gr['gold']) else None) 
                for r,gr in instance_table.groupby('rownr')}
    missgold = {}
    
    
    ## links
    link_table = pd.read_csv(os.path.join(exp_path, 'links', '%s.csv'%table))
    print('link columns:', list(link_table.columns))
    print(len(link_table), 'links')
    link_table['v'].fillna('', inplace=True)
    link_table['predicate'] = link_table['p'].map(lambda u: u[1:-1])
    link_table['value'] = link_table['v'].map(lambda u: u[1:-1])
    link_table['predname'] = link_table['p'].map(lambda u: u[1:-1].split('/')[-1])
    link_table['valname'] = link_table['v'].map(lambda u: u[1:-1].split('/')[-1])
    link_table['p'] = link_table['link'].map(lambda f: ''.join(re.split('(\+|-)', f, 1)[0:2]) )
    links = link_table.sort_values('linkscore', ascending=False).to_dict('records')
    uri_p = {(f['predicate'], f['p'][-1]):f['p'] for f in links}
    
    ## Schema
    property_table = pd.read_csv(os.path.join(exp_path, 'property', '%s.csv'%table))
    print(len(property_table), 'properties')
    property_table['uri'] = property_table['uri'].map(lambda u: u[1:-1])
    property_table['name'] = property_table['uri'].map(lambda u: u.split('/')[-1])
    property_table['id'] = [uri_p.get((row['uri'],row['p']), None) for _,row in property_table.iterrows()]
    schema = {col:gr.sort_values('colscore',ascending=False)
              for col,gr in property_table.groupby('col')}
    
    ## Matches
    matches = pd.read_csv(os.path.join(exp_path, 'matches', '%s.csv'%table))
    # show top label matches
    matches = matches[(matches.groupby(['row', 'uri', 'col'])['tokenjaccard'].transform('max') == matches['tokenjaccard'])]
    # TODO: merge matches with schema
    i_uri = matches[['i','uri']].drop_duplicates().set_index('i')
    
    
    print(len(matches), 'matches')
    matches['p'] = matches['link'].map(lambda f: ''.join(re.split('(\+|-)', f, 1)[0:2]) )
    matches_ = {}
    for i, row in matches.iterrows():
        matches_.setdefault(row['row'], {}).setdefault(row['uri'], []).append(row.to_dict())
    matches = matches_
    
    entities = pd.read_csv(os.path.join(exp_path, 'entities', '%s.csv'%table))
    entities = entities.groupby('i').agg('count').merge(i_uri, left_index=True, right_index=True)
    n_links = entities.set_index('uri')['link'].to_dict()
    print('link examples:', list(n_links.items())[:3])

    
    return render_template(
        'table.html', 
        predpath=exp,
        table=table,
        
        header=header,
        rows=list(enumerate(rows)),
        
        candidates=candidates,
        schema=schema,
        links=links,
        
        methods=methods,
        method1=method1,
        method2=method2,
        scores=scores,
        correct1=correct1,
        correct2=correct2,
        entropy=entropy,
        
        hasgold=hasgold,
        missgold=missgold,
        row_gold=row_gold,
        
        n_links=n_links,
        
        matches=matches,
        fig = plot_pr_curve(joined_instance_table.set_index(['table', 'rownr', 'uri'])),
        
        display_no_gold=display_no_gold
    )