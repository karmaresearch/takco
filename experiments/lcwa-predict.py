import sys, os.path, json
try:
    outdir = sys.argv[1]
    N_FEATURE_ITERATIONS, N_BP_ITERATIONS = int(sys.argv[2]), int(sys.argv[3])
    infiles = sys.argv[4:]
    assert(infiles)
    
    def make_dir(name):
        path = os.path.join(outdir, name)
        os.makedirs(path, exist_ok=True)
        return path
    
    os.makedirs(outdir, exist_ok=True)
    
    featdir = make_dir('features')
    entitydir = make_dir('entities')
    
    instancedir = make_dir('instance')
    propertydir = make_dir('property')
    classdir = make_dir('class')
    
    errordir = make_dir('errors')
    simdir = make_dir('sims')
    
except Exception as e:
    print(e)
    print('Usage: python LCWA-predict.py [outdir]')
    sys.exit(0)

# DB_PATH = './data/dbpedia_subset/db/'
DB_PATH = './data/dbpedia_part/db_infloc/'

LOAD_SURFACE = True
RELOAD_TABLES = True
USE_REDIR = True

N_WORKERS = 8

# N_FEATURE_ITERATIONS, N_BP_ITERATIONS = 1,1


candidate_dir = './data_t2k/output-v1-goldCandidates/instance_candidates/candidate/'
table_fnames = set(infiles)
table_names = set([os.path.basename(fname) for fname in table_fnames])

gold_file = './data_t2k/v1/gs_instance.csv'

surface_file = './data_t2k/db/surfaceforms.txt'
redir_file = './data_t2k/db/redirects.txt'
transitive_redirects_en_file = './data/transitive_redirects_en.ttl'


### LOAD TABLE CELLS ###
rowcontents = {}
table_col_isnumeric = {}
import os, csv, html, re
def splitcells(cells):
    # T2KMatch table serialization reader
    cells = html.unescape(cells).replace('}','{')
    cells = [c.split('|') for c in next(csv.reader([cells], delimiter='|', quotechar='{'))]
    cells = [set([tokenize(c) for c in cs]) for cs in cells]
    return cells
NUM = re.compile('[-0-9.]')
def isnumeric(col):
    numratio = [(sum(1 for _ in NUM.finditer(c)) / len(c)) if c else 0 for c in col]
    return (((sum(numratio) / len(col)) if col else 0) > 0.5)
WHITESPACE = re.compile('\s+')
for fname in table_fnames:
    name = os.path.basename(fname)
    header, *rows = list(csv.reader(open(fname,'r')))
    rows = [[WHITESPACE.sub(' ',c) for c in r] for r in rows]
    cols = zip(*rows)
    table_col_isnumeric[name] = {i: isnumeric(col) for i,col in enumerate(cols)}
    for i,row in enumerate(rows):
        rowcontents['%s~Row%d' % (name, i)] = '|'.join(row)


### LOAD CANDIDATES ###
import pandas as pd
import numpy as np
import os, glob, sys
import urllib

candidate_fnames = glob.glob(os.path.join(candidate_dir, '*.*'))

candidates = pd.concat([pd.read_csv(fname) for fname in candidate_fnames], ignore_index=True)
candidates['table'] = candidates.row.map(lambda row: row.split('~')[0])
candidates = candidates.loc[candidates.table.map(lambda x: x in table_names)]
candidates['entity'] = candidates.entity.map(lambda uri: uri.replace('/page/', '/resource/'))
print('candidates:', len(candidates), list(candidates.columns))
print('%d unique rows' % len(set(candidates.row)), '; %d tables' % len(set(candidates.table)))

gold = pd.read_csv(gold_file, header=None, names=['row', 'entity', 'gold'])
gold['table'] = gold.row.map(lambda row: row.split('~')[0])
gold['entity'] = gold.entity.map(lambda uri: uri.replace('/page/', '/resource/'))
gold = gold.loc[gold.table.map(lambda x: x in table_names)]
print('gold:', len(gold), list(gold.columns))
print('%s unique rows' % len(set(gold.row)), '; %d tables' % len(set(candidates.table)))

sys.stdout.flush()

redir = {}
if USE_REDIR:
    for i,line in enumerate(open(transitive_redirects_en_file)):
        source, _, target = line.strip()[:-2].split(' ', 2)
        source, target = urllib.parse.unquote(source[1:-1]), urllib.parse.unquote(target[1:-1])
        redir[source] = target
        if 0 == i % 100000:
            print('Loading redirects line %d    ' % i, end='\r', file=sys.stderr)

## Lookup entitites in KB
import trident
# trident.setLoggingLevel(0)
g = trident.Db(DB_PATH)
type_id = g.lookup_id('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')
label_id = g.lookup_id('<http://www.w3.org/2000/01/rdf-schema#label>')
assert label_id != None
print('%d terms, %d triples' % (g.n_terms(), g.n_triples()))

def try_lookup(uri):
    uri = redir.get(uri, uri)
    e = g.lookup_id('<%s>' % uri)
    if not e:
        e = g.lookup_id('<%s>' % urllib.parse.quote(uri, safe='/:'))
    if not e:
        e = g.lookup_id('<%s>' % urllib.parse.unquote(uri))
    return e or -1

candidates['id'] = candidates.entity.map(try_lookup)
print('%d candidate entities not in KB' % len([i for _,i in set(zip(candidates.entity,candidates.id)) if i == -1]))
print('  e.g.', list(candidates.entity[(candidates.id == -1)])[:5])
gold['id'] = gold.entity.map(try_lookup)
print('%d gold entities not in KB' % len([i for _,i in set(zip(gold.entity,gold.id)) if i == -1]))
print('  e.g.', list(gold.entity[(gold.id == -1)])[:5])

sys.stdout.flush()


### LOAD SURFACE FORMS ###

n_surface = None if LOAD_SURFACE else 0 
import urllib
from itertools import islice
surface_sets = {}
lookup_surface = {}
for i,line in islice(enumerate(open(surface_file)), n_surface):
    row = urllib.parse.unquote(bytes(line, "utf-8").decode("unicode_escape")).strip().split('\t')
    surface_sets[i] = row
    for r in row:
        lookup_surface.setdefault(r, set()).add(i)
    if 0 == i % 100000:
        print('Loading surface forms line %d    '% i, end='\r', file=sys.stderr)
print('Got %d surface form sets' % len(surface_sets))


import urllib
from itertools import islice
lookup_redir = {}
for i,line in islice(enumerate(open(redir_file)), n_surface):
    row = urllib.parse.unquote(bytes(line, "utf-8").decode("unicode_escape")).strip().split('\t', 1)
    if len(row) == 2:
        source, target = row
        if source != target:
            lookup_redir.setdefault(target, set()).add(source)
    if 0 == i % 100000:
        print('Loading redirected surface forms line %d    ' % i, end='\r', file=sys.stderr)
print('Got %d redirects' % len(lookup_redir))
def get_all_redir(t, t_set=set()):
    yield t
    if t not in t_set:
        for s in lookup_redir.get(t, []):
            yield from get_all_redir(s, t_set|set([t]) )
print('atheism -> ', list(get_all_redir('atheism')))
print('at&t -> ', list(get_all_redir('at&t')))


import re
bracketed = re.compile(' \([^)]*\)')

def try_bracket(s):
    yield s
    s, n = bracketed.subn('', s)
    if n:
        yield s

def expand_surface(s):
    s = s.lower()
    for r in get_all_redir(s):
        yield from try_bracket(r)
        for i in lookup_surface.get(r, []):
            for j in surface_sets.get(i, []):
                yield from try_bracket(j)

print(set(expand_surface('American Automobile Association')))
# print(sorted(set(expand_surface('United States'))))



import re
token_pattern = re.compile(r"(?u)\b\w+\b")
def tokenize(s):
    return tuple(token_pattern.findall(s.lower()))

def datatype(node_id):
    # RDF datatype
    node = g.lookup_str(node_id)
    if not node:
        return '', None
    elif node[-1] == '>':
        if node[0] == '<':
            return node[1:-1], 'uri' # uri
        elif node[0] == '"' and '^^' in node:
            lit, dtype = node.rsplit('^^', 1)
            return lit[1:-1], dtype[1:-1] # typed literal
    elif node[0] == '"':
        if node[-1] == '"':
            return node[1:-1], 'str' # string
        elif '@' in node:
            lit, lang = node.rsplit('@', 1)
            return lit[1:-1], '@%s'%lang # language string


strs_cache = {}
def strs(i):
    if i in strs_cache:
        return strs_cache[i]
    ss = set()
    # Entity strings
    node, dtype = datatype(i)
    if dtype == 'uri':
        
        # HACK FOR TESTING: also use dbpedia url name
        s = node.replace('http://dbpedia.org/resource/', '').replace('_',' ')
        if s and (s != node):
            ss.update( set([tokenize(s_) for s_ in expand_surface(s)]) )
        # END HACK
            
        for l in g.o(i, label_id):
            s = datatype(l)[0]
            ss.update( set([tokenize(s_) for s_ in expand_surface(s)]) )
    elif dtype == 'str':
        s = node
        ss.update( set([tokenize(s_) for s_ in expand_surface(s)]) )
    strs_cache[i] = ss
    return ss


# What is shared by the gold entities?
from collections import Counter
from functools import reduce
from itertools import islice
pd.set_option('display.width', 1000)

# LOOKUP FEATURES
all_id_feats = {}
def features(i):
    if i in all_id_feats:
        return all_id_feats[i]
    # The feature set consists of the inlinks, outlinks, inpredicates and outpredicates of an entity
    # Features are normalized by number of subjects/objects that appear with predicate!
        # because...
    # yields (((direction, predicate), value), count)
    po, ps = set(g.po(i)), set(g.ps(i))
    poc, psc = Counter(p for p,_ in po), Counter(p for p,_ in ps)
    yield from ( ((( 1, p),  o), 1/poc[p]) for p,o in po)
    yield from ( (((-1, p),  s), 1/psc[p]) for p,s in ps) # inverse direction
    yield from ( ((( 1, p), -1), 1) for p in poc)
    yield from ( (((-1, p), -1), 1) for p in psc) # inverse direction
    
    # HACK FOR TESTING: also use dbpedia url name
    yield ((( 1, label_id), i), 1/(poc[label_id] or 1))



all_feat_counts = {}
def count_feature(dp,v):
    if (dp,v) in all_feat_counts:
        return all_feat_counts[(dp,v)]
    # Count inlinks/outlinks or inpredicates/outpredicates in KB
    d,p = dp # predicate, direction
    assert p != None, 'p is None'
    assert v != None, 'v is None'
    return g.count_p(p) if v == -1 else (g.n_s(p,v) if d == 1 else g.n_o(v,p))

def jaccard(a,b):
    a,b = set(a), set(b)
    return len(a&b) / len(a|b) if a|b else 0

p_prior = {p: g.count_p(p)/g.n_terms() for p in g.all_p()}

# Format direction and predicate for debugging
def fmt_dp(d,p):
    return '%21.21s %s' % ((g.lookup_str(p) or '').split('/')[-1][:-1], '>' if d==1 else '<')
def fmt_dpv(dp,v):
    return '%s %-21.21s' % (fmt_dp(*dp), (g.lookup_str(v) or '').split('/')[-1][:-1])


m = pd.merge(gold, candidates, how='outer', on=['table', 'row', 'entity', 'id'])
m.score.fillna(0, inplace=True)
m['rownr'] = m.row.map(lambda r: int(r.split('~Row')[1]))
m['score'] /= m.groupby('row').score.transform('sum')
m['name'] = m.entity.map(lambda s: s.replace('http://dbpedia.org/resource/', ''))
m.gold.fillna(False, inplace=True)
if rowcontents:
    m.rowcontent = m.row.map(lambda r: rowcontents.get(r, ''))
m.rowcontent.fillna('', inplace=True)
m['strscore'] = 0
m['simscore'] = 0


# LOAD FEATURES
all_features = set()
all_ids = set(m.id)
n = len(all_ids)
for i,e in enumerate(all_ids):
    if e > 0:
        for dpv,c in features(e):
            all_features.add(dpv)
    if 0 == i % 100:
        print('Getting entity features %6d / %6d' % (i,n), end='\r')
print()

all_vs = set()
n = len(all_features)
for i,(dp,v) in enumerate(all_features):
    count_feature(dp,v)
    all_vs.add(v)
    if 0 == i % 10000:
        print('Getting feature counts %9d / %9d' % (i,n), end='\r')
print()

n = len(all_vs)
for i,v in enumerate(all_vs):
    strs(v)
    if 0 == i % 10000:
        print('Getting labels %9d / %9d' % (i,n), end='\r')
print()


def link(table, pertable_idx, pertable):
    nrows = len(set(pertable.row))
    t = pertable
    print('Processing table %s (%d rows, %d candidates)' % (table, nrows, len(t)), file=sys.stderr)

    ids = set([i for i in pertable.id if i>0])
    id_feats = {i: Counter(dict(features(i))) for i in ids}
    
    feats = set.union(*map(set,id_feats.values()))
    feat_count = Counter({dpv: (count_feature(*dpv) or 1) for dpv in feats})
    feat_rowcount = Counter()
    for row, group in pertable.groupby('row'):
        feat_rowcount.update(set(dpv for i in group.id for dpv in id_feats.get(i, {})))
    
    id_pred_val = {}
    for e, fs in id_feats.items():
        for dp, v in fs: # aggregate over predicates
            id_pred_val.setdefault(e, {}).setdefault(dp, set()).add(v)

    # Get feature string scores, per (column, row, entity, predicate) [bottleneck: row loops]
    row_cells = {}
    col_row_id_pred_scores = {}
    for i, (row, group) in enumerate(pertable.groupby('row')):
        cells = splitcells(next((s for s in set(group.rowcontent) if s), ''))
        row_cells[row] = cells
        for e in group.id:
            for col, cell in enumerate(cells):
                if table_col_isnumeric.get(table, {}).get(col, False):
                    continue
                pred_scores = col_row_id_pred_scores.setdefault(col, {}).setdefault(row, {}).setdefault(e, Counter())
                for dp, vs in id_pred_val.get(e,{}).items(): # maximize over predicates later
                    ss = set(s for v in vs for s in set(strs(v)))
                    j = max((jaccard(s,c) for c in cell for s in ss), default=0)
                    if j:
                        pred_scores[dp] = j

    # Make column-relation scores: Average per row & features
    col_pred_scores = {}
    max_row_cover = {}
    for col, rips in col_row_id_pred_scores.items():
        if table_col_isnumeric.get(table, {}).get(col, False):
            continue
        pred_scores = Counter()
        covered_rows = set()
        for row,ips in rips.items():
            n_rowcandidates = len(ips)
            for e,ps in ips.items():
                n_feats = len(ps)
                for dp,j in ps.items():
                    pred_scores[dp] += j * (1 / n_rowcandidates)
                    covered_rows.add(row)
        if pred_scores:
            top_s = sum(pred_scores.values())
            max_row_cover[col] = len(covered_rows) / nrows
            # Column predicate score = predscore * rowcover
            # Do not normalize predscore: that way columns with many matches are more important than those with few
            col_pred_scores[col] = Counter({(d,p):s * max_row_cover[col] # * (1/top_s) # do not normalize predscore!
                                            for (d,p),s in pred_scores.items()})
    col_best_confidence = sum(max(ps.values()) for ps in col_pred_scores.values())

    schemastats = pd.DataFrame.from_records([
        (col, p,d, max_row_cover[col], c)
        for col,ps in col_pred_scores.items()
        for (d,p),c in ps.items()
    ], columns=['column', 'predicate', 'direction', 'rowcover', 'score']
    ).sort_values(['column', 'score'], ascending=False)


    # calculate string scores
    score = {}
    max_e_score = {}
    tot_e_score = {}
    label_score_numer = {}
    label_score_denom = {}
    for idx in pertable_idx:
        e = pertable.loc[idx, 'id']
        row = pertable.loc[idx, 'row']
        if e > 0:
            # potential = max possible given predicates
            potential = sum(max((s for p,s in ps.items() if p in id_pred_val.get(e, {})), default=0)
                            for col,ps in col_pred_scores.items())
                        
            col_ps = {col: col_row_id_pred_scores[col].get(row,{}).get(e, Counter()) for col in col_pred_scores}
            # sum of column-predicate scores
            tot = sum(max((col_pred_scores[col][dp]*ps[dp] for dp in ps), default=0)
                      for col,ps in col_ps.items())
            label_score_numer[(row,e)] = tot
            label_score_denom[(row,e)] = potential
            score[(row,e)] = ( (tot / potential) if potential else 0 )
            
            max_e_score[e] = max(max_e_score.get(e,0), score[(row,e)])
            tot_e_score[e] = tot_e_score.get(e, 0) + score[(row,e)]
    t.loc[pertable_idx, 'strscore_numer'] = [label_score_numer.get((row,e), 0) for row,e in zip(pertable.row, pertable.id)]
    t.loc[pertable_idx, 'strscore_denom'] = [label_score_denom.get((row,e), 0) for row,e in zip(pertable.row, pertable.id)]
#     t.loc[pertable_idx, 'strscore'] = [score.get((row,e), 0)/(tot_e_score.get(e,1) or 1) for row,e in zip(pertable.row, pertable.id)]
    t.loc[pertable_idx, 'strscore'] = [score.get((row,e), 0) for row,e in zip(pertable.row, pertable.id)]
    
    
    score = label_score_numer # Try non-lCWA
    
    # Filter feats
    feats = set([f for f in feats if feat_count[f]>1 and feat_rowcount[f]>1])
    feat_count = Counter({f:c for f,c in feat_count.items() if f in feats})
    feat_rowcount = Counter({f:c for f,c in feat_rowcount.items() if f in feats})
    id_feats = {i:Counter({f:s for f,s in fs.items() if f in feats }) for i,fs in id_feats.items()}
    feat_rowcovertable = Counter({dpv: (c/nrows) for dpv,c in feat_rowcount.items()}) # fraction of rows the feature covers
    feat_rowcoverdb = Counter({dpv: (c/feat_count[dpv]) for dpv,c in feat_rowcount.items()}) # fraction of entities the rows cover
    t['n_feats'] = [len(id_feats[e]) if e>0 else 0 for e in t.id]
    
    for feature_loop_i in range(N_FEATURE_ITERATIONS):
        print('feature_loop_i', feature_loop_i)
        
        # calculate feature scores
        featscore = Counter()
        for row, group in list(pertable.groupby('row')):
            group_featscore = Counter()
            for e in group.id:
                for dpv,c in id_feats.get(e, {}).items():
                    # FILTER FEATURES
                    # filter by frequency: only feats that could occur in any row are valid
                    # filter by overlap: only feats that cover sqrt(nrows) rows
                    if True: #feat_count[dpv] > nrows and feat_rowcount[dpv] > nrows**0.5:
                        # normalize by entity score, feature frequency and feature cover
                        group_featscore[dpv] = max(group_featscore[dpv], c * score.get((row,e),0))
            featscore += group_featscore

        featscore = Counter({dpv: (featscore[dpv] / feat_count[dpv]) * (featscore[dpv] / feat_rowcount[dpv])
                             for dpv in featscore})


        # Calculate entity similarities
        sims = {}
        valid_es = set(e for e in set(pertable.id) if e > 0 and max_e_score.get(e,0) > 0)
        for e1 in valid_es:
            e1_feats = Counter({dpv: id_feats[e1][dpv] * featscore[dpv] for dpv in id_feats.get(e1, Counter()) & featscore})
#             e1_feats = Counter({dpv: id_feats[e1][dpv] for dpv in id_feats.get(e1, Counter())}) # without feature weights
            for e2 in valid_es:
                s = sum(e1_feats[dpv]*id_feats[e2][dpv] for dpv in e1_feats)
                sims[(e1,e2)] = s
                sims[(e2,e1)] = s
        if not sims:
            print('The similarity matrix has disappeared!! Table %s' % table)
            break
        
        # Perform Belief Propagation
        e_i = {e:i for i,e in enumerate(valid_es)}
        i_e = {i:e for i,e in enumerate(valid_es)}
        import scipy.sparse
        data, i, j = zip(*[(s, e_i[e1], e_i[e2]) for (e1,e2),s in sims.items() if e1 in e_i and e2 in e_i if e1!=e2 if s>0])
        D = scipy.sparse.coo_matrix((data, (i, j)), shape=(len(e_i), len(e_i)))
    #     print(D.shape)
        data,i,j = zip(*((s, r, e_i[e]) for r,e,s in zip(t.rownr, t.id, t.strscore) if e in e_i))
        P = scipy.sparse.coo_matrix((data,(i,j)),shape=(max(t.rownr)+1, len(e_i)))
    #     print(P.shape)

        import numpy as np
        M = P
        for _ in range(N_BP_ITERATIONS):
            K = np.array(scipy.log1p(M.dot(D).todense()).sum(axis=0))[0]
            K = K / K.max()
            M = scipy.sparse.coo_matrix(P.toarray()*K)

        score = {('%s~Row%d'%(table,r),i_e[i]):s for r,i,s in zip(*M.nonzero(), M.data)}
        
    t = t.merge(pd.DataFrame([(r,e,s) for (r,e),s in score.items()], 
                                columns=['row', 'id', 'coherence']), how='outer')
    t['coherence'].fillna(0, inplace=True)
    t['sim'] = t['coherence']
    
    t['simscore'] = t['coherence'] + t['strscore'] # Add-combine
    
    
    
    feat_i = {dpv:i for i,dpv in enumerate(featscore) }
    featstats = pd.DataFrame.from_records(
        [tuple([
            feat_i[((d,p),v)], p, d, v, *[b.get(((d,p),v), None) 
                for b in [feat_count, feat_rowcount, feat_rowcovertable, feat_rowcoverdb, featscore]]
            ])
        for ((d,p),v) in featscore],
        columns = ['feature','predicate','direction','value','count','rows','rowcover','featcover','score']
        ).sort_values('score', ascending=False)
    
    id_feat_table = pd.DataFrame.from_records(
        [(i,feat_i[f],c) for i,fc in id_feats.items() for f,c in fc.items() if f in feat_i],
        columns = ['entity','feature','score'])
    
    
    return table, t, row_cells, id_feats, id_feat_table, id_pred_val, col_pred_scores, featscore, featstats, schemastats, sims


# Logging all stats for analysis in GUI later
def analysis(table, t, row_cells, id_feats, id_pred_val, col_pred_scores, featscore):
    analysis_gold, analysis_pred = {}, {}
    for row, group in list(t.groupby('row')):
        
        best = group.loc[group.simscore.idxmax()]
        if best.gold != True:
            
            real = next(iter(group[group.gold == True].itertuples()), None)
            if real:
                cells = row_cells.get(row, [])
                if real.id in id_feats:

                    def get_feature_scores(cell, e):
                        for dp, vs in id_pred_val.get(e, {}).items(): # maximize over predicates
                            ss = set(s for v in vs for s in set(strs(v)))
                            sc = max(((s,c) for c in cell for s in ss), default=0, key=lambda sc:jaccard(*sc))
                            if sc and jaccard(*sc):
                                s,c = sc
                                yield c, dp, s, jaccard(s,c)

                    def show_candidate(name, i):
                        analysis = {}
                        e = i['id']
                        analysis['uri'] = g.lookup_str(e)
                        analysis['score'] = i['simscore']

                        # str
                        potential = sum(max((s for p,s in ps.items() if p in id_pred_val.get(e, {})), default=0)
                                        for ps in col_pred_scores.values())
                        col_fs = {col: (cell, list(get_feature_scores(cell, e))) for col,cell in enumerate(cells)}
                        n_colmatch = sum(1 for _,fs in col_fs.values() if fs)
                        tot = sum(max(col_pred_scores.get(col,{}).get(dp,0)*j for _,dp,_,j in fs)
                                  for col, (_,fs) in col_fs.items() if fs)
                        for col, (cell, fs) in col_fs.items():
                            if fs:
                                c, (d,p), s, j = max(fs, key=lambda x:col_pred_scores.get(col,{}).get(x[1],0)*x[3])
                                cs = col_pred_scores.get(col,{}).get((d,p),0)
                        analysis['str'] = {
                            'score':i['strscore'], 'n_colmatch':n_colmatch, 'tot':tot, 'potential':potential, 
                            'matches':[dict(cell=c, label=s, jac=j, pred=g.lookup_str(p), dir=d, col=col)
                                       for col, (cell, fs) in col_fs.items() if fs
                                       for c, (d,p), s, j in [max(fs, key=lambda x:col_pred_scores.get(col,{}).get(x[1],0)*x[3])]]
                        }

                        # sim
                        preds = id_pred_val.get(e, {})
                        potential = sum(featscore[(dp,v)] for dp,v in featscore if dp in preds)

                        n_potential = len(set([dp for dp,v in featscore if dp in preds])) # potential preds
                        feats = id_feats.get(e, Counter())

                        sim = sum(featscore.get(dpv, 0) for dpv in feats)
                        if feats and potential:
                            for dpv in sorted(feats&featscore, key=lambda dpv: -(feats[dpv]*featscore[dpv]))[:5]:
                                dp,v = dpv
                        analysis['sim'] = {
                            'score':i['sim'],
                            'n_feats':len(feats), 'n_potential':n_potential, 'sim':sim, 'potential':potential,
                            'n_used_feats': sum(int(dpv in featscore) for dpv in feats),
                            'matches':[dict(predicate=g.lookup_str(p), direction=d, value=g.lookup_str(v), feat=feats[((d,p),v)], featscore=featscore[((d,p),v) ])
                                       for (d,p),v in sorted(feats&featscore, key=lambda dpv: -(feats[dpv]*featscore[dpv]))]
                        }




                        return analysis


                    analysis_gold[row] = show_candidate('gold', real._asdict())
                    analysis_pred[row] = show_candidate('best', best)
            else:
                pass # gold not in kb
    return analysis_gold, analysis_pred

import concurrent.futures
def link_all(m):
    table_typ = []
    gr = m.groupby('table')
    groups = list(gr.indices.items())
    n_tasks = len(groups)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:        
#         tasks = { executor.submit(link, table, pertable_idx, gr.get_group(table).copy())
#                   for table, pertable_idx in groups }
#         for i, res_task in enumerate(concurrent.futures.as_completed(tasks)):
#             res = res_task.result()
    for i,res in enumerate(link(table, pertable_idx, gr.get_group(table).copy()) for table, pertable_idx in groups):
            table, t, row_cells, id_feats, id_feat_table, id_pred_val, col_pred_scores, featscore, featstats, schemastats, sims = res
            analysis_gold, analysis_pred = analysis(table, t, row_cells, id_feats, id_pred_val, col_pred_scores, featscore)
            print('%4d / %4d tables processed' % (i,n_tasks), file=sys.stderr)

            predfile = os.path.join(instancedir, table)
            t.to_csv(predfile)
            
            featstats['predicate'] = featstats['predicate'].map(lambda p: g.lookup_str(p))
            featstats['value'] = featstats['value'].map(lambda v: g.lookup_str(v))
            featstats.to_csv(os.path.join(featdir, '%s.tsv'%table), sep='\t', float_format='%.5f', index=False)
            id_feat_table.to_csv(os.path.join(entitydir, '%s.tsv'%table), sep='\t', float_format='%.5f', index=False)

            schemastats['predicate'] = schemastats['predicate'].map(lambda p: g.lookup_str(p))
            schemastats.to_csv(os.path.join(propertydir, '%s.tsv'%table), sep='\t', float_format='%.2f', index=False)

            with open(os.path.join(errordir, '%s.json'%table), 'w') as fw:
                json.dump(dict(gold=analysis_gold, pred=analysis_pred), fw, indent=2)

            typ_uri = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
            topclasses = [(table, (g.lookup_str(v) or '')[1:-1],c) for ((d,p),v),c in featscore.most_common() if g.lookup_str(p) == typ_uri]
            classfile = os.path.join(classdir, table)
            pd.DataFrame(topclasses, columns=['table', 'class', 'score']).to_csv(classfile)
            
            strscore = dict(zip(t.id, t.groupby('id').simscore.transform('max')))
            valid_es = set(e for e in id_feats if strscore.get(e,0) > 0)
            with open(os.path.join(simdir, '%s.tsv'%table), 'w') as fw:
                for (e1,e2), s in sims.items():
                    if e1 in valid_es and e2 in valid_es and s > 0:
                        print(e1,e2,'%.3f' % (-np.log(s)), sep='\t', file=fw)

link_all(m)