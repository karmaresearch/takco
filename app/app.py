import sys, os, glob, csv, json, html, urllib, itertools, collections
from flask import Flask, escape, request, render_template, g, redirect

import trident

from data import read_t2d, kbs, Db
from stats import get_kbgold, get_kbinfo, get_novelty

# Run Flask: env PYTHONPATH=src/ FLASK_ENV=development FLASK_APP=app/app.py DATADIR=data/ flask run --port=5555
app = Flask(__name__)


def get_datasets():
    if 'datasets' not in g:
        datadir = os.environ.get('DATADIR')
        datasets_file = os.path.join(datadir, 'datasets-cache.json')
        if not os.path.exists(datasets_file):
            print('re-loading datasets...')
            datasets = {
                't2d-v1': list( read_t2d( os.path.join(datadir, 'raw', 't2d-v1'), v=1 ) ),
                't2d-v2': list( read_t2d( os.path.join(datadir, 'raw', 't2d-v2'), v=2 ) ),
    #             't2d-fixed': list( read_t2d( os.path.join(datadir, 'raw', 'TAIPAN-Datasets', 'T2D'), v=2 ) ),
    #             't2d-star': list( read_t2d( os.path.join(datadir, 'raw', 'TAIPAN-Datasets', 'T2DStar'), v=2 ) ),
            }
            with open(datasets_file, 'w') as fw:
                json.dump(datasets, fw)
        g.datasets = json.load(open(datasets_file))
    return g.datasets

def save_datasets():
    datadir = os.environ.get('DATADIR')
    datasets_file = os.path.join(datadir, 'datasets-cache.json')
    with open(datasets_file, 'w') as fw:
        json.dump(g.datasets, fw)

def get_kbs():
    global kbs
    return {'kbid': request.cookies.get('kbid'), 'kbs':kbs}

def get_kb():
    global kbs
    if 'kb' not in g:
        defaultkb = kbs
        kbdir = os.environ.get('KBDIR')
        kbid = get_kbs().get('kbid')
        if kbid in get_kbs().get('kbs'):
            kbpath = os.path.realpath(os.path.join(kbdir, *kbs[kbid]['path']))
            if os.path.exists(kbpath):
                g.kb = Db(kbpath, **kbs[kbid])
            else:
                raise Exception('Bad KB path!')
        else:
            g.kb = None
    return g.kb

@app.route('/setcookies', methods=['POST'])
def setcookies():
    resp = redirect(request.referrer)
    for name in ['kbid']:
        val = request.form.get(name, default = None, type = str)
        if val is not None:
            resp.set_cookie(name, val)
    return resp

@app.route('/')
def home():
    datasets = get_datasets()
    stats = {}
    for datasetname, data in datasets.items():
        stats[datasetname] = dict(
            numtables = len(data),
            numrows = sum(1 for d in data if d['rows'] for _ in d['rows']),
            numcols = sum(1 for d in data if d['rows'] for _ in zip(*d['rows'])),
            numents = sum(1 for d in data if d['entities'] for _ in d['entities']),
            numclass = sum(1 for d in data if d['class']),
            numprops = sum(1 for d in data if d['properties'] for _ in d['properties']),
        )
    
    return render_template(
        'home.html', 
        tablenames={name:sorted(t['name'] for t in tables) for name, tables in datasets.items()},
        stats=stats,
        **get_kbs(),
    )

@app.route('/table')
def table():
    dataset = request.args.get('dataset', None, type=str)
    tablename = request.args.get('table', None, type=str)
    nocache = request.args.get('nocache', False, type=bool)
    tables = {t['name']:t for t in get_datasets()[dataset]}
    assert tablename in tables
    table = tables[tablename]
    
    tablenames = sorted(tables)
    table_index = tablenames.index(tablename)
    prev_table = tablenames[table_index-1]
    next_table = tablenames[table_index+1 if table_index+1<len(tablenames) else 0]
    
    kb = get_kb()
    if kb:
        kblinks = table.get('kblinks', {}).get(kb.name)
        if nocache or not kblinks:
            kblinks, kbinfo = get_kbgold(kb, table)
            table.setdefault('kblinks', {}).setdefault(kb.name, kblinks)
            save_datasets()
        else:
            kbinfo = get_kbinfo(kb, kblinks)
        novelty = get_novelty(kb, kblinks)
    
    return render_template(
        'table.html', 
        dataset=dataset,
        tablename=tablename,
        tablenames=tablenames,
        prev_table=prev_table,
        table_index=table_index,
        next_table=next_table,
        table=table,
        kbinfo=kbinfo,
        kblinks=kblinks,
        novelty=novelty,
        **get_kbs(),
        
    )

@app.route('/dataset')
def dataset():
    dataset = request.args.get('dataset', None, type=str)
    nocache = request.args.get('nocache', False, type=bool)
    
    tables = {t['name']:t for t in get_datasets()[dataset]}
    
    table_novelty = {}
    kb = get_kb()
    if kb:
        kbinfo = {}
        for i, (name, table) in enumerate( tables.items() ):
            kblinks = table.get('kblinks', {}).get(kb.name)
            if nocache or not kblinks:
                print(f'{i:4d} / {len(tables)} {name}')
                kblinks, kbinfo_ = get_kbgold(kb, table)
                kbinfo.update( kbinfo_ )
                table.setdefault('kblinks', {}).setdefault(kb.name, kblinks)
            table_novelty[name] = table['kblinks'][kb.name]['novelty']
            
        save_datasets()
        
    novelty = {}
    for kind in ['lbl', 'cls', 'prop']:
        novelty[kind] = sum(n[kind] for n in table_novelty.values())
        novelty[kind + '_nomatch'] = sum(n.get(kind + '_nomatch', 0) for n in table_novelty.values())
        novelty[kind + '_redundant'] = sum(n.get(kind + '_redundant', 0) for n in table_novelty.values())
        novelty[kind + '_total'] = sum(n[kind + '_total'] for n in table_novelty.values())
    novelty.update({
        'lbl_pct': (novelty['lbl'] / novelty['lbl_total']) if novelty['lbl_total'] else 0,
        'cls_pct': (novelty['cls'] / novelty['cls_total']) if novelty['cls_total'] else 0,
        'prop_pct': (novelty['prop'] / novelty['prop_total']) if novelty['prop_total'] else 0,
    })
    novelty.update({
        'lbl_val_pct': (novelty['lbl_nomatch'] / novelty['lbl_total']) if novelty['lbl_total'] else 0,
        'prop_val_pct': (novelty['prop_nomatch'] / novelty['prop_total']) if novelty['prop_total'] else 0,
    })
    
    return render_template(
        'dataset.html', 
        dataset=dataset,
        tables=tables,
        table_novelty=table_novelty,
        novelty=novelty,
        **get_kbs(),
    )